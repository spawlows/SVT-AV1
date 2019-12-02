#ifdef __USE_GNU
#undef __USE_GNU  // defined in EbThreads.h
#endif
#ifdef _GNU_SOURCE
#undef _GNU_SOURCE  // defined in EbThreads.h
#endif
#include "aom_dsp_rtcd.h"
#include "util.h"
#include "EbUtility.h"
#include "gtest/gtest.h"
#include "random.h"

using std::make_tuple;
using std::tuple;
using svt_av1_test_tool::SVTRandom;

#ifndef AV1_K_MEANS_RENAME
#define AV1_K_MEANS_RENAME(func, dim) func##_dim##dim
#endif
void AV1_K_MEANS_RENAME(av1_k_means, 1)(const int* data, int* centroids,
                                        uint8_t* indices, int n, int k,
                                        int max_itr);
void AV1_K_MEANS_RENAME(av1_k_means, 2)(const int* data, int* centroids,
                                        uint8_t* indices, int n, int k,
                                        int max_itr);
#define DIVIDE_AND_ROUND(x, y) (((x) + ((y) >> 1)) / (y))

static INLINE unsigned int lcg_rand16(unsigned int* state) {
    *state = (unsigned int)(*state * 1103515245ULL + 12345);
    return *state / 65536 % 32768;
}

#define AV1_K_MEANS_DIM 1
#include "k_means_template.h"
#undef AV1_K_MEANS_DIM
#define AV1_K_MEANS_DIM 2
#include "k_means_template.h"
#undef AV1_K_MEANS_DIM

/***************************************************************************************/
void av1_calc_indices_1(const int* data, const int* centroids,
                              uint8_t* indices, int n, int k) {
    for (int i = 0; i < n; ++i) {
        int min_dist = (int)SQR((int)data[i] - centroids[0]);
        indices[i] = 0;
        for (int j = 1; j < k; ++j) {
            const int this_dist = (int)SQR((int)data[i] - centroids[j]);
            if (this_dist < min_dist) {
                min_dist = this_dist;
                indices[i] = j;
            }
        }
    }
}

static int64_t calc_total_dist_1(const int* data, const int* centroids,
                                       const uint8_t* indices, int n, int k) {
    int64_t dist = 0;
    (void)k;
    for (int i = 0; i < n; ++i) {
        dist += (int)SQR((int)data[i] - centroids[indices[i]]);
    }
    return dist;
}

static void calc_centroids_1(const int* data, int* centroids,
                                   const uint8_t* indices, int n, int k) {
    int i;
    int count[PALETTE_MAX_SIZE] = {0};
    unsigned int rand_state = (unsigned int)data[0];
    assert(n <= 32768);
    memset(centroids, 0, sizeof(centroids[0]) * k);

    for (i = 0; i < n; ++i) {
        const int index = indices[i];
        assert(index < k);
        ++count[index];
        centroids[index] += data[i];
    }

    for (i = 0; i < k; ++i) {
        if (count[i] == 0) {
            memcpy(centroids + i,
                   data + (lcg_rand16(&rand_state) % n), sizeof(centroids[0]));
        } else {
            centroids[i] = DIVIDE_AND_ROUND(centroids[i], count[i]);
        }
    }
}

void av1_k_means_dim1_avx2(const int* data, int* centroids, uint8_t* indices, int n,
                      int k, int max_itr) {
    int pre_centroids[2 * PALETTE_MAX_SIZE];
    uint8_t pre_indices[MAX_SB_SQUARE];

    av1_calc_indices_1(data, centroids, indices, n, k);
    int64_t this_dist = calc_total_dist_1(data, centroids, indices, n, k);

    for (int i = 0; i < max_itr; ++i) {
        const int64_t pre_dist = this_dist;
        memcpy(pre_centroids, centroids, sizeof(pre_centroids[0]) * k);
        memcpy(pre_indices, indices, sizeof(pre_indices[0]) * n);

        calc_centroids_1(data, centroids, indices, n, k);
        av1_calc_indices_1(data, centroids, indices, n, k);
        this_dist = calc_total_dist_1(data, centroids, indices, n, k);

        if (this_dist > pre_dist) {
            memcpy(centroids, pre_centroids, sizeof(pre_centroids[0]) * k);
            memcpy(indices, pre_indices, sizeof(pre_indices[0]) * n);
            break;
        }
        if (!memcmp(centroids, pre_centroids, sizeof(pre_centroids[0]) * k))
            break;
    }
}
/***************************************************************************************/

namespace {

    using av1_k_means_func = void (*)(const int* data, int* centroids,
                                      uint8_t* indices, int n, int k, int max_itr);

    #define MAX_BLOCK_SIZE (MAX_SB_SIZE * MAX_SB_SIZE)
    typedef std::tuple<int, int> BlockSize;
    typedef enum { MIN, MAX, RANDOM } TestPattern;
    BlockSize TEST_BLOCK_SIZES[] = {
        BlockSize(4, 4),    BlockSize(4, 8),    BlockSize(4, 16),
        BlockSize(8, 4),    BlockSize(8, 8),    BlockSize(8, 16),
        BlockSize(8, 32),   BlockSize(16, 4),   BlockSize(16, 8),
        BlockSize(16, 16),  BlockSize(16, 32),  BlockSize(16, 64),
        BlockSize(32, 8),   BlockSize(32, 16),  BlockSize(32, 32),
        BlockSize(32, 64),  BlockSize(64, 16),  BlockSize(64, 32),
        BlockSize(64, 64),  BlockSize(64, 128), BlockSize(128, 64),
        BlockSize(128, 128)};
    TestPattern TEST_PATTERNS[] = {MIN, MAX, RANDOM};

    typedef std::tuple<av1_k_means_func, av1_k_means_func> FuncPair;
    FuncPair TEST_FUNC_PAIRS[] = {FuncPair(AV1_K_MEANS_RENAME(av1_k_means, 1), av1_k_means_dim1_avx2),
                                  FuncPair(AV1_K_MEANS_RENAME(av1_k_means, 2),
                                           AV1_K_MEANS_RENAME(av1_k_means, 2))};

    typedef std::tuple<TestPattern, BlockSize, FuncPair> Av1KMeansDimParam;

    class Av1KMeansDim : public ::testing::WithParamInterface<Av1KMeansDimParam>,
                         public ::testing::Test {
      public:
        Av1KMeansDim() {
            rnd8_ = new SVTRandom(0, ((1 << 8) - 1));
            rnd32_ = new SVTRandom(-((1 << 30) - 1), ((1 << 30) - 1));
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
                memset(data_, 0, n_ * sizeof(int));
                memset(centroids_ref_, 0, centroids_size_ * sizeof(int));
                memset(centroids_tst_, 0, centroids_size_ * sizeof(int));
                memset(indices_ref_, 0, indices_size_ * sizeof(uint8_t));
                memset(indices_tst_, 0, indices_size_ * sizeof(uint8_t));
            }
            else if (pattern_ == MAX) {
                memset(data_, 0xff, n_ * sizeof(int));
                memset(centroids_ref_, 0xff, centroids_size_ * sizeof(int));
                memset(centroids_tst_, 0xff, centroids_size_ * sizeof(int));
                memset(indices_ref_, 0xff, indices_size_ * sizeof(uint8_t));
                memset(indices_tst_, 0xff, indices_size_ * sizeof(uint8_t));
            }
            else { //pattern_= == RANDOM
                for (size_t i = 0; i < n_; i++)
                    data_[i] = rnd32_->random();
                for (size_t i = 0; i < centroids_size_; i++)
                    centroids_ref_[i] = centroids_tst_[i] = rnd32_->random();
                for (size_t i = 0; i < indices_size_; i++)
                    indices_ref_[i] = indices_tst_[i] = rnd8_->random();
            }
        }

        void check_output() {
            int res = memcmp(centroids_ref_, centroids_tst_, centroids_size_* sizeof(int));
            ASSERT_EQ(res, 0) << "Compare Centroids array error";

            res = memcmp(indices_ref_, indices_tst_, indices_size_* sizeof(uint8_t));
            ASSERT_EQ(res, 0) << "Compare indices array error";
        }

        void run_test() {
            size_t test_num = 1000;
            if (pattern_ == MIN || pattern_ == MAX)
                test_num = 1;

            for (size_t k = 2; k <= 8; k++) {
                for (size_t i = 0; i < test_num; i++) {
                    prepare_data();
                    func_ref_(data_, centroids_ref_, indices_ref_, n_, k, max_itr_);
                    func_tst_(data_, centroids_tst_, indices_tst_, n_, k, max_itr_);
                    check_output();
                }
            }
        }

      protected:
        SVTRandom* rnd32_;
        SVTRandom* rnd8_;
        av1_k_means_func func_ref_;
        av1_k_means_func func_tst_;
        int* data_;
        int* centroids_ref_;
        int* centroids_tst_;
        uint8_t* indices_ref_;
        uint8_t* indices_tst_;

        uint32_t centroids_size_;
        uint32_t indices_size_;
        TestPattern pattern_;
        BlockSize block_;

        const int max_itr_ = 50;
        int n_;
    };

    TEST_P(Av1KMeansDim, RunCheckOutput) {
        run_test();
    };

    INSTANTIATE_TEST_CASE_P(
        Av1KMeansDim, Av1KMeansDim,
        ::testing::Combine(::testing::ValuesIn(TEST_PATTERNS),
                           ::testing::ValuesIn(TEST_BLOCK_SIZES),
                           ::testing::ValuesIn(TEST_FUNC_PAIRS)));

}  // namespace