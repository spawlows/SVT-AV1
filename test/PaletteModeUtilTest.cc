/*
 * Copyright(c) 2019 Netflix, Inc.
 * SPDX - License - Identifier: BSD - 2 - Clause - Patent
 */

/******************************************************************************
 * @file PaletteModeUtilTest.cc
 *
 * @brief Unit test for util functions in palette mode:
 * - eb_av1_count_colors
 * - av1_count_colors_highbd
 * - av1_k_means_dim1
 * - av1_k_means_dim2
 *
 * @author Cidana-Edmond
 *
 ******************************************************************************/
#include <vector>
#include "gtest/gtest.h"
// workaround to eliminate the compiling warning on linux
// The macro will conflict with definition in gtest.h
#ifdef __USE_GNU
#undef __USE_GNU  // defined in EbThreads.h
#endif
#ifdef _GNU_SOURCE
#undef _GNU_SOURCE  // defined in EbThreads.h
#endif
#include "EbDefinitions.h"
#include "EbUtility.h"
#include "random.h"
#include "util.h"
#include "EbTime.h"
#include "aom_dsp_rtcd.h"

using std::tuple;
using std::vector;
using svt_av1_test_tool::SVTRandom;

#ifdef __cplusplus
extern "C" {
#endif

int av1_get_palette_color_index_context(const uint8_t *color_map, int stride,
                                        int r, int c, int palette_size,
                                        uint8_t *color_order, int *color_idx);

#ifdef __cplusplus
}
#endif

namespace {

extern "C" int eb_av1_count_colors(const uint8_t *src, int stride, int rows,
                                   int cols, int *val_count);
extern "C" int av1_count_colors_highbd(uint16_t *src, int stride, int rows,
                                       int cols, int bit_depth, int *val_count);

/**
 * @brief Unit test for counting colors:
 * - eb_av1_count_colors
 * - av1_count_colors_highbd
 *
 * Test strategy:
 * Feeds the random value both into test function and the vector without
 * duplicated, then compares the count of result and the individual item count
 * in vector.
 *
 * Expected result:
 * The count numbers from test function and vector are the same.
 *
 * Test coverage:
 * The input can be 8-bit and 8-bit/10-bit/12-bit for HBD cases
 */
template <typename Sample>
class ColorCountTest : public ::testing::Test {
  protected:
    ColorCountTest() : rnd_(16, false) {
        input_ =
            (Sample *)eb_aom_memalign(32, MAX_PALETTE_SQUARE * sizeof(Sample));
        bd_ = 8;
        ref_.clear();
        val_count_ = nullptr;
    }

    ~ColorCountTest() {
        if (input_) {
            eb_aom_free(input_);
            input_ = nullptr;
        }
        aom_clear_system_state();
    }

    void prepare_data() {
        memset(input_, 0, MAX_PALETTE_SQUARE * sizeof(Sample));
        ref_.clear();
        const int32_t mask = (1 << bd_) - 1;
        for (size_t i = 0; i < MAX_PALETTE_SQUARE; i++) {
            input_[i] = rnd_.random() & mask;
            /** put same value into a vector for reference */
            ref_.push_back(input_[i]);
        }
        /** remove all duplicated items */
        std::sort(ref_.begin(), ref_.end());
        vector<int>::iterator it = std::unique(ref_.begin(), ref_.end());
        ref_.erase(it, ref_.end());
    }

    void run_test(size_t times) {
        const int max_colors = (1 << bd_);
        val_count_ = (int *)eb_aom_memalign(32, max_colors * sizeof(int));
        for (size_t i = 0; i < times; i++) {
            prepare_data();
            ASSERT_EQ(count_color(), ref_.size())
                << "color count failed at: " << i;
        }
        if (val_count_) {
            eb_aom_free(val_count_);
            val_count_ = nullptr;
        }
    }

    virtual unsigned int count_color() = 0;

  protected:
    SVTRandom rnd_;
    Sample *input_;
    uint8_t bd_;
    vector<int> ref_;
    int *val_count_;
};

class ColorCountLbdTest : public ColorCountTest<uint8_t> {
  protected:
    unsigned int count_color() override {
        const int max_colors = (1 << bd_);
        memset(val_count_, 0, max_colors * sizeof(int));
        unsigned int colors =
            (unsigned int)eb_av1_count_colors(input_, 64, 64, 64, val_count_);
        return colors;
    }
};

TEST_F(ColorCountLbdTest, MatchTest) {
    run_test(1000);
}

class ColorCountHbdTest : public ColorCountTest<uint16_t> {
  protected:
    unsigned int count_color() override {
        const int max_colors = (1 << bd_);
        memset(val_count_, 0, max_colors * sizeof(int));
        unsigned int colors = (unsigned int)av1_count_colors_highbd(
            input_, 64, 64, 64, bd_, val_count_);
        return colors;
    }
};

TEST_F(ColorCountHbdTest, MatchTest8Bit) {
    bd_ = 8;
    run_test(1000);
}

TEST_F(ColorCountHbdTest, MatchTest10Bit) {
    bd_ = 10;
    run_test(1000);
}

TEST_F(ColorCountHbdTest, MatchTest12Bit) {
    bd_ = 12;
    run_test(1000);
}

extern "C" void av1_k_means_dim1_c(const int *data, int *centroids,
                                 uint8_t *indices, int n, int k, int max_itr);
extern "C" void av1_k_means_dim2_c(const int *data, int *centroids,
                                 uint8_t *indices, int n, int k, int max_itr);
static const int MaxItr = 50;
/**
 * @brief Unit test for kmeans functions:
 * - av1_k_means_dim1
 * - av1_k_means_dim2
 *
 * Test strategy:
 * Feeds the plane buffer with random colors into kmeans function and get the
 * centroids and indices, verifies each color being the closest to the centroid
 * in all candidates.
 *
 * Expected result:
 * Every pixels are closest to their centroid in all candidates
 *
 * Test coverage:
 * Tests for K from PALETTE_MIN_SIZE to PALETTE_MAX_SIZE
 */
class KMeansTest : public ::testing::TestWithParam<int> {
  protected:
    KMeansTest() : rnd_(8, false), palette_rnd_(2, 64) {
        k_ = GetParam();
        data_ = new int[2 * MAX_PALETTE_SQUARE];
    }

    ~KMeansTest() {
        if (data_) {
            delete[] data_;
            data_ = nullptr;
        }
    }

    /** functions for 1d test */
    int prepare_data(const int max_colors) {
        memset(data_, 0, MAX_PALETTE_SQUARE * sizeof(int));
        uint8_t *palette = new uint8_t[max_colors];
        for (int i = 0; i < max_colors; i++)
            palette[i] = rnd_.random();
        uint8_t tmp[MAX_PALETTE_SQUARE] = {0};
        for (size_t i = 0; i < MAX_PALETTE_SQUARE; i++)
            data_[i] = tmp[i] = palette[rnd_.random() % max_colors];
        delete[] palette;
        int val_count[MAX_PALETTE_SQUARE] = {0};
        return eb_av1_count_colors(tmp, 64, 64, 64, val_count);
    }

    void run_test(size_t times) {
        uint8_t indices[MAX_PALETTE_SQUARE] = {0};
        for (size_t i = 0; i < times; i++) {
            const int max_colors = palette_rnd_.random();
            const int colors = prepare_data(max_colors);
            int centroids[PALETTE_MAX_SIZE] = {0};
            int k = AOMMIN(colors, k_);
            av1_k_means_dim1_c(
                data_, centroids, indices, MAX_PALETTE_SQUARE, k, MaxItr);
            check_output(centroids, k, data_, indices, MAX_PALETTE_SQUARE);
        }
    }

    void check_output(const int *centroids, const int k, const int *data,
                      const uint8_t *indices, const int n) {
        for (int i = 0; i < n; i++) {
            const int min_delta = abs(data[i] - centroids[indices[i]]);
            for (int j = 0; j < k; j++) {
                const int delta = abs(data[i] - centroids[j]);
                ASSERT_GE(delta, min_delta)
                    << "index error at " << i << ", value is " << data[i]
                    << ", distance to centroid( " << centroids[indices[i]]
                    << ") is greater than to " << centroids[j];
            }
        }
    }

    /** functions for 2d test */
    int prepare_data_2d(const int max_colors) {
        memset(data_, 0, 2 * MAX_PALETTE_SQUARE * sizeof(int));
        uint16_t *palette = new uint16_t[max_colors];
        for (int i = 0; i < max_colors; i++)
            palette[i] = (rnd_.random() << 8) + rnd_.random();
        vector<uint16_t> val_vec;
        for (size_t i = 0; i < MAX_PALETTE_SQUARE; i++) {
            uint16_t tmp = palette[rnd_.random() % max_colors];
            data_[2 * i] = tmp >> 8;
            data_[2 * i + 1] = tmp & 0xFF;
            val_vec.push_back(tmp);
        }
        delete[] palette;
        std::sort(val_vec.begin(), val_vec.end());
        vector<uint16_t>::iterator it =
            std::unique(val_vec.begin(), val_vec.end());
        val_vec.erase(it, val_vec.end());
        return (const int)val_vec.size();
    }

    void run_test_2d(size_t times) {
        uint8_t indices[2 * MAX_PALETTE_SQUARE] = {0};
        for (size_t i = 0; i < times; i++) {
            const int max_colors = palette_rnd_.random();
            const int colors = prepare_data_2d(max_colors);
            int centroids[2 * PALETTE_MAX_SIZE] = {0};
            int k = AOMMIN(colors, k_);
            av1_k_means_dim2_c(
                data_, centroids, indices, MAX_PALETTE_SQUARE, k, MaxItr);
            check_output_2d(centroids, k, data_, indices, MAX_PALETTE_SQUARE);
        }
    }

    static double distance_2d(int x1, int y1, int x2, int y2) {
        int x_d = x1 - x2;
        int y_d = y1 - y2;
        return sqrt(x_d * x_d + y_d * y_d);
    }

    void check_output_2d(const int *centroids, const int k, const int *data,
                         const uint8_t *indices, const int n) {
        for (int i = 0; i < n; i++) {
            const double min_delta = distance_2d(data[2 * i],
                                                 data[2 * i + 1],
                                                 centroids[2 * indices[i]],
                                                 centroids[2 * indices[i] + 1]);
            for (int j = 0; j < k; j++) {
                const double delta = distance_2d(data[2 * i],
                                                 data[2 * i + 1],
                                                 centroids[2 * j],
                                                 centroids[2 * j + 1]);
                ASSERT_GE(delta, min_delta)
                    << "index error at " << i << ", value is " << data[i]
                    << ", distance to centroid( " << centroids[indices[i]]
                    << ") is greater than to " << centroids[j];
            }
        }
    }

  protected:
    int *data_;
    int k_;
    SVTRandom rnd_;
    SVTRandom palette_rnd_;
};

TEST_P(KMeansTest, CheckOutput) {
    run_test(1000);
};

TEST_P(KMeansTest, CheckOutput2D) {
    run_test_2d(1000);
};

INSTANTIATE_TEST_CASE_P(PalleteMode, KMeansTest,
                        ::testing::Range(PALETTE_MIN_SIZE, PALETTE_MAX_SIZE));


typedef void (*av1_k_means_func)(const int *data, int *centroids,
                      uint8_t *indices, int n, int k, int max_itr);

#define MAX_BLOCK_SIZE (MAX_SB_SIZE * MAX_SB_SIZE)
typedef std::tuple<int, int> BlockSize;
typedef enum { MIN, MAX, RANDOM } TestPattern;
BlockSize TEST_BLOCK_SIZES[] = {BlockSize(4, 4),
                                BlockSize(4, 8),
                                BlockSize(8, 8),
                                BlockSize(8, 16),
                                BlockSize(8, 32),
                                BlockSize(16, 4),
                                BlockSize(16, 16),
                                BlockSize(16, 32),
                                BlockSize(32, 8),
                                BlockSize(32, 32),
                                BlockSize(32, 64),
                                BlockSize(64, 16),
                                BlockSize(64, 64),
                                BlockSize(64, 128),
                                BlockSize(128, 128)};
TestPattern TEST_PATTERNS[] = {MIN, MAX, RANDOM};

static void av1_calc_indices_dim1_c_wrap(const int *data, int *centroids,
                                    uint8_t *indices, int n, int k,
                                    int max_itr)
{
    (void)max_itr;
    av1_calc_indices_dim1_c(data, centroids, indices, n, k);
}

static void av1_calc_indices_dim1_avx2_wrap(const int *data, int *centroids,
                                    uint8_t *indices, int n, int k,
                                    int max_itr)
{
    (void)max_itr;
    av1_calc_indices_dim1_avx2(data, centroids, indices, n, k);
}

static void av1_calc_indices_dim2_c_wrap(const int *data, int *centroids,
                                         uint8_t *indices, int n, int k,
                                         int max_itr) {
    (void)max_itr;
    av1_calc_indices_dim2_c(data, centroids, indices, n, k);
}

static void av1_calc_indices_dim2_avx2_wrap(const int *data, int *centroids,
                                            uint8_t *indices, int n, int k,
                                            int max_itr) {
    (void)max_itr;
    av1_calc_indices_dim2_avx2(data, centroids, indices, n, k);
}

typedef std::tuple<av1_k_means_func, av1_k_means_func> FuncPair;
FuncPair TEST_FUNC_PAIRS[] = {
    FuncPair(av1_calc_indices_dim1_c_wrap, av1_calc_indices_dim1_avx2_wrap),
    FuncPair(av1_k_means_dim1_c, av1_k_means_dim1_avx2),
    FuncPair(av1_calc_indices_dim2_c_wrap, av1_calc_indices_dim2_avx2_wrap),
    FuncPair(av1_k_means_dim2_c, av1_k_means_dim2_avx2)
};

typedef std::tuple<TestPattern, BlockSize, FuncPair> Av1KMeansDimParam;

class Av1KMeansDim : public ::testing::WithParamInterface<Av1KMeansDimParam>,
                     public ::testing::Test {
  public:
    Av1KMeansDim() {
        rnd8_ = new SVTRandom(0, ((1 << 8) - 1));
        rnd32_ = new SVTRandom(-((1 << 14) - 1), ((1 << 14) - 1));
        pattern_ = TEST_GET_PARAM(0);
        block_ = TEST_GET_PARAM(1);
        func_ref_ = std::get<0>(TEST_GET_PARAM(2));
        func_tst_ = std::get<1>(TEST_GET_PARAM(2));

        n_ = std::get<0>(block_) * std::get<1>(block_);

        // Additonal *2 to account possibility of write into extra memory
        centroids_size_ = 2 * PALETTE_MAX_SIZE * 2;
        indices_size_ = MAX_SB_SQUARE * 2;

        //*2 to account of AV1_K_MEANS_DIM = 2
        data_ = new int[n_ * 2];
        centroids_ref_ = new int[centroids_size_];
        centroids_tst_ = new int[centroids_size_];
        indices_ref_ = new uint8_t[indices_size_];
        indices_tst_ = new uint8_t[indices_size_];
    }

    void TearDown() override {
        if (rnd32_)
            delete rnd32_;
        if (rnd8_)
            delete rnd8_;
        if (data_)
            delete data_;
        if (centroids_ref_)
            delete centroids_ref_;
        if (centroids_tst_)
            delete centroids_tst_;
        if (indices_ref_)
            delete indices_ref_;
        if (indices_tst_)
            delete indices_tst_;
    }

  protected:
    void prepare_data() {
        if (pattern_ == MIN) {
            memset(data_, 0, n_ * sizeof(int) * 2);
            memset(centroids_ref_, 0, centroids_size_ * sizeof(int));
            memset(centroids_tst_, 0, centroids_size_ * sizeof(int));
            memset(indices_ref_, 0, indices_size_ * sizeof(uint8_t));
            memset(indices_tst_, 0, indices_size_ * sizeof(uint8_t));
        } else if (pattern_ == MAX) {
            memset(data_, 0xff, n_ * sizeof(int) * 2);
            memset(centroids_ref_, 0xff, centroids_size_ * sizeof(int));
            memset(centroids_tst_, 0xff, centroids_size_ * sizeof(int));
            memset(indices_ref_, 0xff, indices_size_ * sizeof(uint8_t));
            memset(indices_tst_, 0xff, indices_size_ * sizeof(uint8_t));
        } else {  // pattern_ == RANDOM
            for (int i = 0; i < n_ * 2; i++)
                data_[i] = rnd32_->random();
            for (size_t i = 0; i < centroids_size_; i++)
                centroids_ref_[i] = centroids_tst_[i] = rnd32_->random();
            for (size_t i = 0; i < indices_size_; i++)
                indices_ref_[i] = indices_tst_[i] = rnd8_->random();
        }
    }

    void check_output() {
        int res = memcmp(
            centroids_ref_, centroids_tst_, centroids_size_ * sizeof(int));
        ASSERT_EQ(res, 0) << "Compare Centroids array error";

        res =
            memcmp(indices_ref_, indices_tst_, indices_size_ * sizeof(uint8_t));
        ASSERT_EQ(res, 0) << "Compare indices array error";
    }

    void run_test() {
        size_t test_num = 100;
        if (pattern_ == MIN || pattern_ == MAX)
            test_num = 1;

        for (int k = PALETTE_MIN_SIZE; k <= PALETTE_MAX_SIZE; k++) {
            for (size_t i = 0; i < test_num; i++) {
                prepare_data();
                func_ref_(data_, centroids_ref_, indices_ref_, n_, k, MaxItr);
                func_tst_(data_, centroids_tst_, indices_tst_, n_, k, MaxItr);
                check_output();
            }
        }
    }

    void speed() {
        const uint64_t num_loop = 200000 / (n_ >> 3);
        double time_c, time_o;
        uint64_t start_time_seconds, start_time_useconds;
        uint64_t middle_time_seconds, middle_time_useconds;
        uint64_t finish_time_seconds, finish_time_useconds;

        prepare_data();

        eb_start_time(&start_time_seconds, &start_time_useconds);

        for (uint64_t i = 0; i < num_loop; i++) {
            for (int k = PALETTE_MIN_SIZE; k <= PALETTE_MAX_SIZE; k++) {
                func_ref_(data_, centroids_ref_, indices_ref_, n_, k, MaxItr);
            }
        }

        eb_start_time(&middle_time_seconds, &middle_time_useconds);

        for (uint64_t i = 0; i < num_loop; i++) {
            for (int k = PALETTE_MIN_SIZE; k <= PALETTE_MAX_SIZE; k++) {
                func_tst_(data_, centroids_tst_, indices_tst_, n_, k, MaxItr);
            }
        }

        eb_start_time(&finish_time_seconds, &finish_time_useconds);

        check_output();

        eb_compute_overall_elapsed_time_ms(start_time_seconds,
                                      start_time_useconds,
                                      middle_time_seconds,
                                      middle_time_useconds,
                                      &time_c);
        eb_compute_overall_elapsed_time_ms(middle_time_seconds,
                                      middle_time_useconds,
                                      finish_time_seconds,
                                      finish_time_useconds,
                                      &time_o);

        printf("    speedup %5.2fx\n", time_c / time_o);
    }

  protected:
    SVTRandom *rnd32_;
    SVTRandom *rnd8_;
    av1_k_means_func func_ref_;
    av1_k_means_func func_tst_;
    int *data_;
    int *centroids_ref_;
    int *centroids_tst_;
    uint8_t *indices_ref_;
    uint8_t *indices_tst_;

    uint32_t centroids_size_;
    uint32_t indices_size_;
    TestPattern pattern_;
    BlockSize block_;
    int n_;
};

TEST_P(Av1KMeansDim, RunCheckOutput) {
    run_test();
};

TEST_P(Av1KMeansDim, DISABLED_speed) {
    speed();
};

INSTANTIATE_TEST_CASE_P(
    Av1KMeansDim, Av1KMeansDim,
    ::testing::Combine(::testing::ValuesIn(TEST_PATTERNS),
                       ::testing::ValuesIn(TEST_BLOCK_SIZES),
                       ::testing::ValuesIn(TEST_FUNC_PAIRS)));

/////////

#define MAX_COLOR_CONTEXT_HASH 8
int palette_color_index_context_lookup[MAX_COLOR_CONTEXT_HASH + 1] = {
    -1, -1, 0, -1, -1, 4, 3, 2, 1};

#define NUM_PALETTE_NEIGHBORS 3  // left, top-left and top.
int av1_get_palette_color_index_context_ref(const uint8_t *color_map, int stride,
                                        int r, int c, int palette_size,
                                        uint8_t *color_order, int *color_idx) {
    assert(palette_size <= PALETTE_MAX_SIZE);
    assert(r > 0 || c > 0);

    // Get color indices of neighbors.
    //int color_neighbors[NUM_PALETTE_NEIGHBORS];
    //color_neighbors[0] = (c - 1 >= 0) ? color_map[r * stride + c - 1] : -1;
    //color_neighbors[1] =
    //    (c - 1 >= 0 && r - 1 >= 0) ? color_map[(r - 1) * stride + c - 1] : -1;
    //color_neighbors[2] = (r - 1 >= 0) ? color_map[(r - 1) * stride + c] : -1;


    //int color_neighbors0 = (c - 1 >= 0) ? color_map[r * stride + c - 1] : -1;
    //int color_neighbors1 =
    //    (c - 1 >= 0 && r - 1 >= 0) ? color_map[(r - 1) * stride + c - 1] : -1;
    //int color_neighbors2 =
    //    (r - 1 >= 0) ? color_map[(r - 1) * stride + c] : -1;


    // The +10 below should not be needed. But we get a warning "array subscript
    // is above array bounds [-Werror=array-bounds]" without it, possibly due to
    // this (or similar) bug: https://gcc.gnu.org/bugzilla/show_bug.cgi?id=59124
    int scores[PALETTE_MAX_SIZE + 10] = {0};
    int i;
   // static const int weights[NUM_PALETTE_NEIGHBORS] = {2, 1, 2};
    //for (i = 0; i < NUM_PALETTE_NEIGHBORS; ++i) {
    //    if (color_neighbors[i] >= 0) {
    //        scores[color_neighbors[i]] += weights[i];
    //    }
    //}

        if ((c - 1 >= 0)) {
        scores[color_map[r * stride + c - 1]] += 2;
        }

        if ((c - 1 >= 0 && r - 1 >= 0)) {
            scores[color_map[(r - 1) * stride + c - 1]] += 1;
        }

        if ((r - 1 >= 0)) {
            scores[color_map[(r - 1) * stride + c]] += 2;
        }
    


      /*  
        if (color_neighbors0 >= 0) {
            scores[color_neighbors0] += 2;
        }

        if (color_neighbors1 >= 0) {
            scores[color_neighbors1] += 1;
        }

        if (color_neighbors2 >= 0) {
            scores[color_neighbors2] += 2;
        }*/
    


    int inverse_color_order[PALETTE_MAX_SIZE] = {0, 1, 2, 3, 4, 5, 6, 7};
    for (i = 0; i < PALETTE_MAX_SIZE; ++i) {
        color_order[i] = i;
    }

    //// Get the top NUM_PALETTE_NEIGHBORS scores (sorted from large to small).
    //for (i = 0; i < NUM_PALETTE_NEIGHBORS; ++i) {
    //    int max = scores[i];
    //    int max_idx = i;
    //    for (int j = i + 1; j < 3; ++j) {
    //        if (scores[j] > max) {
    //            max = scores[j];
    //            max_idx = j;
    //        }
    //    }
    //    // Move the score at index 'max_idx' to index 'i', and shift the
    //    // scores from 'i' to 'max_idx - 1' by 1.
    //    const int max_score = scores[max_idx];
    //    const uint8_t max_color_order = color_order[max_idx];
    //    for (int k = max_idx; k > i; --k) {
    //        scores[k] = scores[k - 1];
    //        color_order[k] = color_order[k - 1];
    //        inverse_color_order[color_order[k]] = k;
    //    }
    //    scores[i] = max_score;
    //    color_order[i] = max_color_order;
    //    inverse_color_order[color_order[i]] = i;
    //}




      // Get the top NUM_PALETTE_NEIGHBORS scores (sorted from large to small).
    //0
        //int max = scores[0];
        //int max_idx = 0;


        //
        //if (!(color_neighbors0 >= 0)) {
        //    if ((color_neighbors2 >= 0)) {
        //        max = scores[2];
        //        max_idx = 2;
        //    } else if ((color_neighbors1 >= 0)) {
        //        max = scores[1];
        //        max_idx = 1;
        //    }
        //}

        ////for (int j = i + 1; j < 3; ++j) {
        ////    if (scores[j] > max) {
        ////        max = scores[j];
        ////        max_idx = j;
        ////    }
        ////}
        //// Move the score at index 'max_idx' to index 'i', and shift the
        //// scores from 'i' to 'max_idx - 1' by 1.
        //const int max_score = scores[max_idx];
        //const uint8_t max_color_order = color_order[max_idx];
        //for (int k = max_idx; k > i; --k) {
        //    scores[k] = scores[k - 1];
        //    color_order[k] = color_order[k - 1];
        //    inverse_color_order[color_order[k]] = k;
        //}
        //scores[i] = max_score;
        //color_order[i] = max_color_order;
        //inverse_color_order[color_order[i]] = i;

         // Get the top NUM_PALETTE_NEIGHBORS scores (sorted from large to
        // small).
       // for (i = 0; i <1; ++i) {

            int max = scores[0];
            int max_idx = 0;


 /*   if (!(color_neighbors0 >= 0)) {
        if ((color_neighbors2 >= 0)) {
            max = scores[2];
            max_idx = 2;
        } else if ((color_neighbors1 >= 0)) {
            max = scores[1];
            max_idx = 1;
        }
        }*/


            for (int j = 0 + 1; j < 3; ++j) {
                if (scores[j] > max) {
                    max = scores[j];
                    max_idx = j;
                }
            }
            // Move the score at index 'max_idx' to index 'i', and shift the
            // scores from 'i' to 'max_idx - 1' by 1.
            const int max_score = scores[max_idx];
            const uint8_t max_color_order = color_order[max_idx];
            for (int k = max_idx; k >0; --k) {
                scores[k] = scores[k - 1];
                color_order[k] = color_order[k - 1];
                inverse_color_order[color_order[k]] = k;
            }
            scores[0] = max_score;
            color_order[0] = max_color_order;
            inverse_color_order[color_order[0]] = 0;
       // }


      // Get the top NUM_PALETTE_NEIGHBORS scores (sorted from large to small).
    for (i = 1; i < NUM_PALETTE_NEIGHBORS; ++i) {
        int max = scores[i];
        int max_idx = i;
        for (int j = i + 1; j < 3; ++j) {
            if (scores[j] > max) {
                max = scores[j];
                max_idx = j;
            }
        }
        // Move the score at index 'max_idx' to index 'i', and shift the
        // scores from 'i' to 'max_idx - 1' by 1.
        const int max_score = scores[max_idx];
        const uint8_t max_color_order = color_order[max_idx];
        for (int k = max_idx; k > i; --k) {
            scores[k] = scores[k - 1];
            color_order[k] = color_order[k - 1];
            inverse_color_order[color_order[k]] = k;
        }
        scores[i] = max_score;
        color_order[i] = max_color_order;
        inverse_color_order[color_order[i]] = i;
    }















    if (color_idx != NULL)
        *color_idx = inverse_color_order[color_map[r * stride + c]];

    // Get hash value of context.
    int color_index_ctx_hash = scores[0] * 1 + scores[1] * 2 + scores[2] * 2;

    assert(color_index_ctx_hash > 0);
    assert(color_index_ctx_hash <= MAX_COLOR_CONTEXT_HASH);

    // Lookup context from hash.
    const int color_index_ctx =
        palette_color_index_context_lookup[color_index_ctx_hash];
    assert(color_index_ctx >= 0);
    assert(color_index_ctx < PALETTE_COLOR_INDEX_CONTEXTS);
    return color_index_ctx;
}

/* clang-format off */
 typedef AomCdfProb(*MapCdf)[PALETTE_COLOR_INDEX_CONTEXTS]
     [CDF_SIZE(PALETTE_COLORS)];
 typedef const int(*ColorCost)[PALETTE_SIZES][PALETTE_COLOR_INDEX_CONTEXTS]
     [PALETTE_COLORS];
/* clang-format on */


typedef struct {
    int rows;
    int cols;
    int n_colors;
    int plane_width;
    uint8_t *color_map;
    MapCdf map_cdf;
    ColorCost color_cost;
} Av1ColorMapParam;


static int cost_and_tokenize_map_ref(Av1ColorMapParam *param, TOKENEXTRA **t,
                                 int plane, int calc_rate, int allow_update_cdf,
                                 MapCdf map_pb_cdf) {
    const uint8_t *const color_map = param->color_map;
    MapCdf map_cdf = param->map_cdf;
    ColorCost color_cost = param->color_cost;
    const int plane_block_width = param->plane_width;
    const int rows = param->rows;
    const int cols = param->cols;
    const int n = param->n_colors;
    const int palette_size_idx = n - PALETTE_MIN_SIZE;
    int this_rate = 0;
    uint8_t color_order[PALETTE_MAX_SIZE];

    (void)plane;

    for (int k = 1; k < rows + cols - 1; ++k) {
        for (int j = AOMMIN(k, cols - 1); j >= AOMMAX(0, k - rows + 1); --j) {
            int i = k - j;
            int color_new_idx;
            const int color_ctx =
                av1_get_palette_color_index_context(color_map,
                                                    plane_block_width,
                                                    i,
                                                    j,
                                                    n,
                                                    color_order,
                                                    &color_new_idx);
            assert(color_new_idx >= 0 && color_new_idx < n);
            if (calc_rate) {
                this_rate +=
                    (*color_cost)[palette_size_idx][color_ctx][color_new_idx];
            } else {
                (*t)->token = color_new_idx;
                (*t)->color_map_cdf = map_pb_cdf[palette_size_idx][color_ctx];
                ++(*t);
                if (allow_update_cdf)
                    update_cdf(
                        map_cdf[palette_size_idx][color_ctx], color_new_idx, n);
#if CONFIG_ENTROPY_STATS
                if (plane) {
                    ++counts->palette_uv_color_index[palette_size_idx]
                                                    [color_ctx][color_new_idx];
                } else {
                    ++counts->palette_y_color_index[palette_size_idx][color_ctx]
                                                   [color_new_idx];
                }
#endif
            }
        }
    }
    if (calc_rate)
        return this_rate;
    return 0;
}




class PaletteColor : public ::testing::TestWithParam<int> {
  public:
    PaletteColor() {
        rnd8_ = new SVTRandom(0, ((1 << 8) - 1));
        //rnd32_ = new SVTRandom(-((1 << 14) - 1), ((1 << 14) - 1));
        //pattern_ = TEST_GET_PARAM(0);
        //block_ = TEST_GET_PARAM(1);
        //func_ref_ = std::get<0>(TEST_GET_PARAM(2));
        //func_tst_ = std::get<1>(TEST_GET_PARAM(2));

        //n_ = std::get<0>(block_) * std::get<1>(block_);

        //// Additonal *2 to account possibility of write into extra memory
        //centroids_size_ = 2 * PALETTE_MAX_SIZE * 2;
        //indices_size_ = MAX_SB_SQUARE * 2;

        ////*2 to account of AV1_K_MEANS_DIM = 2
        //data_ = new int[n_ * 2];
        //centroids_ref_ = new int[centroids_size_];
        //centroids_tst_ = new int[centroids_size_];
        //indices_ref_ = new uint8_t[indices_size_];
        //indices_tst_ = new uint8_t[indices_size_];
    }

    void TearDown() override {
        if (rnd8_)
            delete rnd8_;
       /* if (rnd32_)
            delete rnd32_;
        
        if (data_)
            delete data_;
        if (centroids_ref_)
            delete centroids_ref_;
        if (centroids_tst_)
            delete centroids_tst_;
        if (indices_ref_)
            delete indices_ref_;
        if (indices_tst_)
            delete indices_tst_;*/
    }

  protected:
    void prepare_data() {
        //if (pattern_ == MIN) {
        //    memset(data_, 0, n_ * sizeof(int) * 2);
        //    memset(centroids_ref_, 0, centroids_size_ * sizeof(int));
        //    memset(centroids_tst_, 0, centroids_size_ * sizeof(int));
        //    memset(indices_ref_, 0, indices_size_ * sizeof(uint8_t));
        //    memset(indices_tst_, 0, indices_size_ * sizeof(uint8_t));
        //} else if (pattern_ == MAX) {
        //    memset(data_, 0xff, n_ * sizeof(int) * 2);
        //    memset(centroids_ref_, 0xff, centroids_size_ * sizeof(int));
        //    memset(centroids_tst_, 0xff, centroids_size_ * sizeof(int));
        //    memset(indices_ref_, 0xff, indices_size_ * sizeof(uint8_t));
        //    memset(indices_tst_, 0xff, indices_size_ * sizeof(uint8_t));
        //} else {  // pattern_ == RANDOM
        //    for (int i = 0; i < n_ * 2; i++)
        //        data_[i] = rnd32_->random();
        //    for (size_t i = 0; i < centroids_size_; i++)
        //        centroids_ref_[i] = centroids_tst_[i] = rnd32_->random();
        //    for (size_t i = 0; i < indices_size_; i++)
        //        indices_ref_[i] = indices_tst_[i] = rnd8_->random();
        //}
    }

    void check_output() {
        /*int res = memcmp(
            centroids_ref_, centroids_tst_, centroids_size_ * sizeof(int));
        ASSERT_EQ(res, 0) << "Compare Centroids array error";

        res =
            memcmp(indices_ref_, indices_tst_, indices_size_ * sizeof(uint8_t));
        ASSERT_EQ(res, 0) << "Compare indices array error";*/
    }

    void run_test(int speed_check) {

         const uint64_t num_loop = speed_check?50000000:1;
        double time_c, time_o;
        uint64_t start_time_seconds, start_time_useconds;
        uint64_t middle_time_seconds, middle_time_useconds;
        uint64_t finish_time_seconds, finish_time_useconds;


       /* size_t test_num = 100;
        if (pattern_ == MIN || pattern_ == MAX)
            test_num = 1;

        for (int k = PALETTE_MIN_SIZE; k <= PALETTE_MAX_SIZE; k++) {
            for (size_t i = 0; i < test_num; i++) {
                prepare_data();
                func_ref_(data_, centroids_ref_, indices_ref_, n_, k, MaxItr);
                func_tst_(data_, centroids_tst_, indices_tst_, n_, k, MaxItr);
                check_output();
            }
        }*/
       uint8_t color_order[PALETTE_MAX_SIZE];
       uint8_t color_order_ref[PALETTE_MAX_SIZE];
       int color_idx;
       int color_idx_ref;

       int palette_size = 8;  // 1..8
       uint8_t color_map[64 * 64];
       int c = 32;
       int r = 32;
       int stride = 32;

       for (int i = 0; i < 64 * 64; i++) {
           color_map[i] = rnd8_->random()%6;
       }

       eb_start_time(&start_time_seconds, &start_time_useconds);


        int ret = 0;
       
        
          for (uint64_t i = 0; i < num_loop; i++) {
        
        ret = av1_get_palette_color_index_context(
           color_map,
                                               stride,
                                               r,
                                               c,
                                               palette_size,
                                               color_order,
                                               &color_idx);
        }

          eb_start_time(&middle_time_seconds, &middle_time_useconds);

          int ret_ref;

            for (uint64_t i = 0; i < num_loop; i++) {

        ret_ref = av1_get_palette_color_index_context_ref(color_map,
                                        stride,
                                        r,
                                        c,
                                        palette_size,
                                        color_order_ref,
                                        &color_idx_ref);
        }

        eb_start_time(&finish_time_seconds, &finish_time_useconds);

          int res = memcmp(color_order_ref, color_order, sizeof(color_order));
        ASSERT_EQ(res, 0) << "Compare Centroids array error";


        ASSERT_EQ(color_idx_ref, color_idx)
            << "Compare Centroids array error2";

                ASSERT_EQ(ret_ref, ret) << "Compare Centroids array error3";

                if (speed_check) {
                    eb_compute_overall_elapsed_time_ms(start_time_seconds,
                                                       start_time_useconds,
                                                       middle_time_seconds,
                                                       middle_time_useconds,
                                                       &time_c);
                    eb_compute_overall_elapsed_time_ms(middle_time_seconds,
                                                       middle_time_useconds,
                                                       finish_time_seconds,
                                                       finish_time_useconds,
                                                       &time_o);

                    printf("    speedup %5.2fx\n", time_c / time_o);
                }


    }

    
    void run_test2(int speed_check) {
        const uint64_t num_loop = speed_check ? 50000000 : 1;
        double time_c, time_o;
        uint64_t start_time_seconds, start_time_useconds;
        uint64_t middle_time_seconds, middle_time_useconds;
        uint64_t finish_time_seconds, finish_time_useconds;


       

        uint8_t color_order[PALETTE_MAX_SIZE];
        uint8_t color_order_ref[PALETTE_MAX_SIZE];
        int color_idx;
        int color_idx_ref;



      

        int palette_size = 8;  // 1..8
        uint8_t color_map[64 * 64];
        int c = 32;
        int r = 32;
        int stride = 32;

        for (int i = 0; i < 64 * 64; i++) {
            color_map[i] = rnd8_->random() % 7;
        }


        Av1ColorMapParam param, param_ref;
        memset(&param, 0, sizeof(param));

        int* ColorCost_org[PALETTE_SIZES][PALETTE_COLOR_INDEX_CONTEXTS][PALETTE_COLORS];
        int* ColorCost_ref[PALETTE_SIZES][PALETTE_COLOR_INDEX_CONTEXTS][PALETTE_COLORS];
        memset(ColorCost_org, 0, sizeof(ColorCost_org));
        memset(ColorCost_ref, 0, sizeof(ColorCost_org));


        TOKENEXTRA t1, t2;
        TOKENEXTRA *pt1 = &t1;
        TOKENEXTRA *pt2 = &t2;


        param.color_map = color_map;
        param.rows = 8;
        param.cols = 16;
        param.plane_width = param.cols;
        param.n_colors = 8; //max 1..8????


        param_ref = param;
      /*  param.color_cost = ColorCost_org;
        param_ref.color_cost = ColorCost_ref;*/

       // MapCdf map_cdf = param->map_cdf;
       // ColorCost color_cost = param->color_cost;

         int calc_rate = 0; //0, 1
        int allow_update_cdf = 0;  // 0, 1 jak calc_rate 0

        MapCdf map_pb_cdf_a, map_pb_cdf_b;

      //  memset(map_pb_cdf_a, 0, sizeof(*map_pb_cdf_a));
        map_pb_cdf_a = NULL;

        //AomCdfProb (
        //     *MapCdf)[PALETTE_COLOR_INDEX_CONTEXTS][CDF_SIZE(PALETTE_COLORS)];

        eb_start_time(&start_time_seconds, &start_time_useconds);

        int ret = 0;

        for (uint64_t i = 0; i < num_loop; i++) {
            ret = av1_get_palette_color_index_context(
                color_map, stride, r, c, palette_size, color_order, &color_idx);

            cost_and_tokenize_map_ref(&param,
                                      &pt1,
                                     -1,
                                      calc_rate,
                                      allow_update_cdf, map_pb_cdf_a);


        }

        eb_start_time(&middle_time_seconds, &middle_time_useconds);

        int ret_ref;

        for (uint64_t i = 0; i < num_loop; i++) {
            ret_ref = av1_get_palette_color_index_context_ref(color_map,
                                                              stride,
                                                              r,
                                                              c,
                                                              palette_size,
                                                              color_order_ref,
                                                              &color_idx_ref);
        }

        eb_start_time(&finish_time_seconds, &finish_time_useconds);

        int res = memcmp(color_order_ref, color_order, sizeof(color_order));
        ASSERT_EQ(res, 0) << "Compare Centroids array error";

        ASSERT_EQ(color_idx_ref, color_idx) << "Compare Centroids array error2";

        ASSERT_EQ(ret_ref, ret) << "Compare Centroids array error3";

        if (speed_check) {
            eb_compute_overall_elapsed_time_ms(start_time_seconds,
                                               start_time_useconds,
                                               middle_time_seconds,
                                               middle_time_useconds,
                                               &time_c);
            eb_compute_overall_elapsed_time_ms(middle_time_seconds,
                                               middle_time_useconds,
                                               finish_time_seconds,
                                               finish_time_useconds,
                                               &time_o);

            printf("    speedup %5.2fx\n", time_c / time_o);
        }
    }

    void speed() {
       /* const uint64_t num_loop = 200000 / (n_ >> 3);
        double time_c, time_o;
        uint64_t start_time_seconds, start_time_useconds;
        uint64_t middle_time_seconds, middle_time_useconds;
        uint64_t finish_time_seconds, finish_time_useconds;

        prepare_data();

        eb_start_time(&start_time_seconds, &start_time_useconds);

        for (uint64_t i = 0; i < num_loop; i++) {
            for (int k = PALETTE_MIN_SIZE; k <= PALETTE_MAX_SIZE; k++) {
                func_ref_(data_, centroids_ref_, indices_ref_, n_, k, MaxItr);
            }
        }

        eb_start_time(&middle_time_seconds, &middle_time_useconds);

        for (uint64_t i = 0; i < num_loop; i++) {
            for (int k = PALETTE_MIN_SIZE; k <= PALETTE_MAX_SIZE; k++) {
                func_tst_(data_, centroids_tst_, indices_tst_, n_, k, MaxItr);
            }
        }

        eb_start_time(&finish_time_seconds, &finish_time_useconds);

        check_output();

        eb_compute_overall_elapsed_time_ms(start_time_seconds,
                                           start_time_useconds,
                                           middle_time_seconds,
                                           middle_time_useconds,
                                           &time_c);
        eb_compute_overall_elapsed_time_ms(middle_time_seconds,
                                           middle_time_useconds,
                                           finish_time_seconds,
                                           finish_time_useconds,
                                           &time_o);

        printf("    speedup %5.2fx\n", time_c / time_o);*/
    }

  protected:
    SVTRandom *rnd8_;
   /* SVTRandom *rnd32_;
    
    av1_k_means_func func_ref_;
    av1_k_means_func func_tst_;
    int *data_;
    int *centroids_ref_;
    int *centroids_tst_;
    uint8_t *indices_ref_;
    uint8_t *indices_tst_;

    uint32_t centroids_size_;
    uint32_t indices_size_;
    TestPattern pattern_;
    BlockSize block_;
    int n_;*/
};

//TEST_P(PaletteColor, RunCheckOutput) {
//    run_test(0);
//};
//////_get_palette_color_index
//
//TEST_P(PaletteColor, /*DISABLED_*/speed) {
//    run_test(1);
//};

TEST_P(PaletteColor, RunCheckOutput2) {
    run_test2(0);
};
////_get_palette_color_index

TEST_P(PaletteColor, /*DISABLED_*/ speed2) {
    run_test2(1);
};

//INSTANTIATE_TEST_CASE_P(
//    PaletteColor, PaletteColor,
//    ::testing::Combine(::testing::ValuesIn(TEST_PATTERNS),
//                       ::testing::ValuesIn(TEST_BLOCK_SIZES),
//                       ::testing::ValuesIn(TEST_FUNC_PAIRS)));

INSTANTIATE_TEST_CASE_P(PaletteColor, PaletteColor,
                        /*::testing::Range(PALETTE_MIN_SIZE, PALETTE_MAX_SIZE)*/
                        ::testing::Values(1));

}  // namespace
