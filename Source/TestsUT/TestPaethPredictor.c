// INTEL CONFIDENTIAL
// Copyright © 2018 Intel Corporation.
//
// This software and the related documents are Intel copyrighted materials, 
// and your use of them is governed by the express license under which they were provided to you.
// Unless the License provides otherwise, you may not use, modify, copy, publish, distribute, disclose or transmit 
// this software or the related documents without Intel's prior written permission.
// This software and the related documents are provided as is, with no express or implied warranties, 
// other than those that are expressly stated in the License.

// main.cpp
//  -Contructs the following resources needed during the encoding process
//      -memory
//      -threads
//      -semaphores
//  -Configures the encoder
//  -Calls the encoder via the API (Static Library)
//  -Destructs the resources
//
/***************************************
 * Includes
 ***************************************/
#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#ifdef _WIN32
#include <Windows.h>
#endif
#include <immintrin.h>

#include "aom_dsp_rtcd.h"
#include "TestCore.h"

typedef void(*fnhighbd_paeth_predictor)(
    uint16_t *dst,
    ptrdiff_t stride,
    int bw,
    int bh,
    const uint16_t *above,
    const uint16_t *left,
    int bd);

static INLINE int abs_diff(int a, int b) { return (a > b) ? a - b : b - a; }
static INLINE uint16_t paeth_predictor_single(uint16_t left, uint16_t top,
    uint16_t top_left) {
    const int base = top + left - top_left;
    const int p_left = abs_diff(base, left);
    const int p_top = abs_diff(base, top);
    const int p_top_left = abs_diff(base, top_left);

    // Return nearest to base of left, top and top_left.
    return (p_left <= p_top && p_left <= p_top_left)
        ? left
        : (p_top <= p_top_left) ? top : top_left;
}

void highbd_paeth_predictor_c(uint16_t *dst, ptrdiff_t stride,
    int bw, int bh, const uint16_t *above,
    const uint16_t *left, int bd) {
    int r, c;
    const uint16_t ytop_left = above[-1];
    (void)bd;

    for (r = 0; r < bh; r++) {
        for (c = 0; c < bw; c++)
            dst[c] = paeth_predictor_single(left[r], above[c], ytop_left);
        dst += stride;
    }
}

#define EP(x) [x] = #x  /* ENUM PRINT */
enum highbd_paeth_enum {
    paeth_2x2=0,
    paeth_4x4,
    paeth_8x8,
    paeth_16x16,
    paeth_32x32,
    paeth_64x64,
    paeth_4x8,
    paeth_4x16,
    paeth_8x4,
    paeth_8x16,
    paeth_8x32,
    paeth_16x4,
    paeth_16x8,
    paeth_16x32,
    paeth_16x64,
    paeth_32x8,
    paeth_32x16,
    paeth_32x64,
    paeth_64x16,
    paeth_64x32,
    paeth_size
};
const char* Names[] = { EP(paeth_2x2), EP(paeth_4x4),EP(paeth_8x8),
EP(paeth_16x16), EP(paeth_32x32),EP(paeth_64x64),
EP(paeth_4x8), EP(paeth_4x16),EP(paeth_8x4),
EP(paeth_8x16), EP(paeth_8x32),EP(paeth_16x4),
EP(paeth_16x8), EP(paeth_16x32),EP(paeth_16x64),
EP(paeth_32x8), EP(paeth_32x16),EP(paeth_32x64),
EP(paeth_64x16), EP(paeth_64x32)};

//*******************************funcitons AVX2*******************************
static INLINE __m256i paeth_pred(const __m256i *left, const __m256i *top,
    const __m256i *topleft) {
    const __m256i base =
        _mm256_sub_epi16(_mm256_add_epi16(*top, *left), *topleft);

    __m256i pl = _mm256_abs_epi16(_mm256_sub_epi16(base, *left));
    __m256i pt = _mm256_abs_epi16(_mm256_sub_epi16(base, *top));
    __m256i ptl = _mm256_abs_epi16(_mm256_sub_epi16(base, *topleft));

    __m256i mask1 = _mm256_cmpgt_epi16(pl, pt);
    mask1 = _mm256_or_si256(mask1, _mm256_cmpgt_epi16(pl, ptl));
    __m256i mask2 = _mm256_cmpgt_epi16(pt, ptl);

    pl = _mm256_andnot_si256(mask1, *left);

    ptl = _mm256_and_si256(mask2, *topleft);
    pt = _mm256_andnot_si256(mask2, *top);
    pt = _mm256_or_si256(pt, ptl);
    pt = _mm256_and_si256(mask1, pt);

    return _mm256_or_si256(pt, pl);
}

void aom_highbd_paeth_predictor_16x4_avx2(uint16_t *dst, ptrdiff_t stride,
    int bw, int bh, const uint16_t *above, const uint16_t *left, int bd) {
    const __m256i tl16 = _mm256_set1_epi16(above[-1]);
    const __m256i top = _mm256_loadu_si256((const __m256i *)above);

    __m256i l16, row;
    int i;
    for (i = 0; i < 4; ++i) {
        l16 = _mm256_set1_epi16(left[i]);
        row = paeth_pred(&l16, &top, &tl16);
        _mm256_storeu_si256((__m256i *)dst, row);
        dst += stride;
    }
}

void aom_highbd_paeth_predictor_16x8_avx2(uint16_t *dst, ptrdiff_t stride,
    int bw, int bh, const uint16_t *above, const uint16_t *left, int bd) {
    const __m256i tl16 = _mm256_set1_epi16(above[-1]);
    const __m256i top = _mm256_loadu_si256((const __m256i *)above);

    __m256i l16, row;
    int i;
    for (i = 0; i < 8; ++i) {
        l16 = _mm256_set1_epi16(left[i]);
        row = paeth_pred(&l16, &top, &tl16);
        _mm256_storeu_si256((__m256i *)dst, row);
        dst += stride;
    }
}

void aom_highbd_paeth_predictor_16x16_avx2(uint16_t *dst, ptrdiff_t stride,
    int bw, int bh, const uint16_t *above, const uint16_t *left, int bd) {
    const __m256i tl16 = _mm256_set1_epi16(above[-1]);
    const __m256i top = _mm256_loadu_si256((const __m256i *)above);

    __m256i l16, row;
    int i;
    for (i = 0; i < 16; ++i) {
        l16 = _mm256_set1_epi16(left[i]);
        row = paeth_pred(&l16, &top, &tl16);
        _mm256_storeu_si256((__m256i *)dst, row);
        dst += stride;
    }
}

void aom_highbd_paeth_predictor_16x32_avx2(uint16_t *dst, ptrdiff_t stride,
    int bw, int bh, const uint16_t *above, const uint16_t *left, int bd) {
    const __m256i tl16 = _mm256_set1_epi16(above[-1]);
    const __m256i top = _mm256_loadu_si256((const __m256i *)above);

    __m256i l16, row;
    int i;
    for (i = 0; i < 32; ++i) {
        l16 = _mm256_set1_epi16(left[i]);
        row = paeth_pred(&l16, &top, &tl16);
        _mm256_storeu_si256((__m256i *)dst, row);
        dst += stride;
    }
}

void aom_highbd_paeth_predictor_16x64_avx2(uint16_t *dst, ptrdiff_t stride,
    int bw, int bh, const uint16_t *above, const uint16_t *left, int bd) {
    const __m256i tl16 = _mm256_set1_epi16(above[-1]);
    const __m256i top = _mm256_loadu_si256((const __m256i *)above);

    __m256i l16, row;
    int i;
    for (i = 0; i < 64; ++i) {
        l16 = _mm256_set1_epi16(left[i]);
        row = paeth_pred(&l16, &top, &tl16);
        _mm256_storeu_si256((__m256i *)dst, row);
        dst += stride;
    }
}


void aom_highbd_paeth_predictor_32x8_avx2(uint16_t *dst, ptrdiff_t stride,
    int bw, int bh, const uint16_t *above, const uint16_t *left, int bd) {
    const __m256i t0 = _mm256_loadu_si256((const __m256i *)above);
    const __m256i t1 = _mm256_loadu_si256((const __m256i *)(above + 16));
    const __m256i tl = _mm256_set1_epi16(above[-1]);

    __m256i l16, row;
    int i;
    for (i = 0; i < 8; ++i) {
        l16 = _mm256_set1_epi16(left[i]);

        row = paeth_pred(&l16, &t0, &tl);
        _mm256_storeu_si256((__m256i *)dst, row);

        row = paeth_pred(&l16, &t1, &tl);
        _mm256_storeu_si256((__m256i *)(dst + 16), row);

        dst += stride;
    }
}

void aom_highbd_paeth_predictor_32x16_avx2(uint16_t *dst, ptrdiff_t stride,
    int bw, int bh, const uint16_t *above, const uint16_t *left, int bd) {
    const __m256i t0 = _mm256_loadu_si256((const __m256i *)above);
    const __m256i t1 = _mm256_loadu_si256((const __m256i *)(above+16));
    const __m256i tl = _mm256_set1_epi16(above[-1]);

    __m256i l16, row;
    int i;
    for (i = 0; i < 16; ++i) {
        l16 = _mm256_set1_epi16(left[i]);

        row = paeth_pred(&l16, &t0, &tl);
        _mm256_storeu_si256((__m256i *)dst, row);

        row = paeth_pred(&l16, &t1, &tl);
        _mm256_storeu_si256((__m256i *)(dst+16), row);

        dst += stride;
    }
}

void aom_highbd_paeth_predictor_32x32_avx2(uint16_t *dst, ptrdiff_t stride,
    int bw, int bh, const uint16_t *above, const uint16_t *left, int bd) {
    const __m256i t0 = _mm256_loadu_si256((const __m256i *)above);
    const __m256i t1 = _mm256_loadu_si256((const __m256i *)(above + 16));
    const __m256i tl = _mm256_set1_epi16(above[-1]);

    __m256i l16, row;
    int i;
    for (i = 0; i < 32; ++i) {
        l16 = _mm256_set1_epi16(left[i]);

        row = paeth_pred(&l16, &t0, &tl);
        _mm256_storeu_si256((__m256i *)dst, row);

        row = paeth_pred(&l16, &t1, &tl);
        _mm256_storeu_si256((__m256i *)(dst + 16), row);

        dst += stride;
    }
}

void aom_highbd_paeth_predictor_32x64_avx2(uint16_t *dst, ptrdiff_t stride,
    int bw, int bh, const uint16_t *above, const uint16_t *left, int bd) {
    const __m256i t0 = _mm256_loadu_si256((const __m256i *)above);
    const __m256i t1 = _mm256_loadu_si256((const __m256i *)(above + 16));
    const __m256i tl = _mm256_set1_epi16(above[-1]);

    __m256i l16, row;
    int i;
    for (i = 0; i < 64; ++i) {
        l16 = _mm256_set1_epi16(left[i]);

        row = paeth_pred(&l16, &t0, &tl);
        _mm256_storeu_si256((__m256i *)dst, row);

        row = paeth_pred(&l16, &t1, &tl);
        _mm256_storeu_si256((__m256i *)(dst + 16), row);

        dst += stride;
    }
}

void aom_highbd_paeth_predictor_64x16_avx2(uint16_t *dst, ptrdiff_t stride,
    int bw, int bh, const uint16_t *above, const uint16_t *left, int bd) {
    const __m256i t0 = _mm256_loadu_si256((const __m256i *)above);
    const __m256i t1 = _mm256_loadu_si256((const __m256i *)(above + 16));
    const __m256i t2 = _mm256_loadu_si256((const __m256i *)(above + 32));
    const __m256i t3 = _mm256_loadu_si256((const __m256i *)(above + 48));
    const __m256i tl = _mm256_set1_epi16(above[-1]);

    __m256i l16, row;
    int i;
    for (i = 0; i < 16; ++i) {
        l16 = _mm256_set1_epi16(left[i]);

        row = paeth_pred(&l16, &t0, &tl);
        _mm256_storeu_si256((__m256i *)dst, row);

        row = paeth_pred(&l16, &t1, &tl);
        _mm256_storeu_si256((__m256i *)(dst + 16), row);

        row = paeth_pred(&l16, &t2, &tl);
        _mm256_storeu_si256((__m256i *)(dst + 32), row);

        row = paeth_pred(&l16, &t3, &tl);
        _mm256_storeu_si256((__m256i *)(dst + 48), row);

        dst += stride;
    }
}

void aom_highbd_paeth_predictor_64x32_avx2(uint16_t *dst, ptrdiff_t stride,
    int bw, int bh, const uint16_t *above, const uint16_t *left, int bd) {
    const __m256i t0 = _mm256_loadu_si256((const __m256i *)above);
    const __m256i t1 = _mm256_loadu_si256((const __m256i *)(above + 16));
    const __m256i t2 = _mm256_loadu_si256((const __m256i *)(above + 32));
    const __m256i t3 = _mm256_loadu_si256((const __m256i *)(above + 48));
    const __m256i tl = _mm256_set1_epi16(above[-1]);

    __m256i l16, row;
    int i;
    for (i = 0; i < 32; ++i) {
        l16 = _mm256_set1_epi16(left[i]);

        row = paeth_pred(&l16, &t0, &tl);
        _mm256_storeu_si256((__m256i *)dst, row);

        row = paeth_pred(&l16, &t1, &tl);
        _mm256_storeu_si256((__m256i *)(dst + 16), row);

        row = paeth_pred(&l16, &t2, &tl);
        _mm256_storeu_si256((__m256i *)(dst + 32), row);

        row = paeth_pred(&l16, &t3, &tl);
        _mm256_storeu_si256((__m256i *)(dst + 48), row);

        dst += stride;
    }
}


void aom_highbd_paeth_predictor_64x64_avx2(uint16_t *dst, ptrdiff_t stride,
    int bw, int bh, const uint16_t *above, const uint16_t *left, int bd) {
    const __m256i t0 = _mm256_loadu_si256((const __m256i *)above);
    const __m256i t1 = _mm256_loadu_si256((const __m256i *)(above + 16));
    const __m256i t2 = _mm256_loadu_si256((const __m256i *)(above + 32));
    const __m256i t3 = _mm256_loadu_si256((const __m256i *)(above + 48));
    const __m256i tl = _mm256_set1_epi16(above[-1]);

    __m256i l16, row;
    int i;
    for (i = 0; i < 64; ++i) {
        l16 = _mm256_set1_epi16(left[i]);

        row = paeth_pred(&l16, &t0, &tl);
        _mm256_storeu_si256((__m256i *)dst, row);

        row = paeth_pred(&l16, &t1, &tl);
        _mm256_storeu_si256((__m256i *)(dst + 16), row);

        row = paeth_pred(&l16, &t2, &tl);
        _mm256_storeu_si256((__m256i *)(dst + 32), row);

        row = paeth_pred(&l16, &t3, &tl);
        _mm256_storeu_si256((__m256i *)(dst + 48), row);

        dst += stride;
    }
}

void aom_highbd_paeth_predictor_8x4_avx2(uint16_t *dst, ptrdiff_t stride,
    int bw, int bh, const uint16_t *above, const uint16_t *left, int bd) {
    const __m128i t = _mm_loadu_si128((const __m128i *)above);
    const __m256i t0 = _mm256_setr_m128i(t, t);

    const __m256i tl = _mm256_set1_epi16(above[-1]);

    __m256i l16, row;

    int i;
    for (i = 0; i < 4; i+=2) {
        l16 = _mm256_setr_m128i(_mm_set1_epi16(left[i]), _mm_set1_epi16(left[i+1]));

        row = paeth_pred(&l16, &t0, &tl);
        _mm_storeu_si128((__m128i *)dst, _mm256_extractf128_si256(row,0));
        dst += stride;
        _mm_storeu_si128((__m128i *)dst, _mm256_extractf128_si256(row, 1));
        dst += stride;
    }
}

void aom_highbd_paeth_predictor_8x8_avx2(uint16_t *dst, ptrdiff_t stride,
    int bw, int bh, const uint16_t *above, const uint16_t *left, int bd) {
    const __m128i t = _mm_loadu_si128((const __m128i *)above);
    const __m256i t0 = _mm256_setr_m128i(t, t);

    const __m256i tl = _mm256_set1_epi16(above[-1]);

    __m256i l16, row;

    int i;
    for (i = 0; i < 8; i += 2) {
        l16 = _mm256_setr_m128i(_mm_set1_epi16(left[i]), _mm_set1_epi16(left[i + 1]));

        row = paeth_pred(&l16, &t0, &tl);
        _mm_storeu_si128((__m128i *)dst, _mm256_extractf128_si256(row, 0));
        dst += stride;
        _mm_storeu_si128((__m128i *)dst, _mm256_extractf128_si256(row, 1));
        dst += stride;
    }
}
void aom_highbd_paeth_predictor_8x16_avx2(uint16_t *dst, ptrdiff_t stride,
    int bw, int bh, const uint16_t *above, const uint16_t *left, int bd) {
    const __m128i t = _mm_loadu_si128((const __m128i *)above);
    const __m256i t0 = _mm256_setr_m128i(t, t);

    const __m256i tl = _mm256_set1_epi16(above[-1]);

    __m256i l16, row;

    int i;
    for (i = 0; i < 16; i += 2) {
        l16 = _mm256_setr_m128i(_mm_set1_epi16(left[i]), _mm_set1_epi16(left[i + 1]));

        row = paeth_pred(&l16, &t0, &tl);
        _mm_storeu_si128((__m128i *)dst, _mm256_extractf128_si256(row, 0));
        dst += stride;
        _mm_storeu_si128((__m128i *)dst, _mm256_extractf128_si256(row, 1));
        dst += stride;
    }
}

void aom_highbd_paeth_predictor_8x32_avx2(uint16_t *dst, ptrdiff_t stride,
    int bw, int bh, const uint16_t *above, const uint16_t *left, int bd) {
    const __m128i t = _mm_loadu_si128((const __m128i *)above);
    const __m256i t0 = _mm256_setr_m128i(t, t);

    const __m256i tl = _mm256_set1_epi16(above[-1]);

    __m256i l16, row;

    int i;
    for (i = 0; i < 32; i += 2) {
        l16 = _mm256_setr_m128i(_mm_set1_epi16(left[i]), _mm_set1_epi16(left[i + 1]));

        row = paeth_pred(&l16, &t0, &tl);
        _mm_storeu_si128((__m128i *)dst, _mm256_extractf128_si256(row, 0));
        dst += stride;
        _mm_storeu_si128((__m128i *)dst, _mm256_extractf128_si256(row, 1));
        dst += stride;
    }
}

void aom_highbd_paeth_predictor_4x4_avx2(uint16_t *dst, ptrdiff_t stride,
    int bw, int bh, const uint16_t *above, const uint16_t *left, int bd) {
    const __m256i t0 = _mm256_set1_epi64x(((uint64_t*)above)[0]);

    const __m256i tl = _mm256_set1_epi16(above[-1]);

    __m256i l16, row;

    int i;
    for (i = 0; i < 4; i += 4) {
        l16 = _mm256_setr_epi16(left[i], left[i], left[i], left[i],
            left[i+1], left[i + 1], left[i + 1], left[i + 1],
            left[i+2], left[i + 2], left[i + 2], left[i + 2],
            left[i+3], left[i + 3], left[i + 3], left[i + 3]);
        /*l16 = _mm256_setr_epi64x(_mm_set1_pi16(left[i]), _mm_set1_pi16(left[i + 1]),
            _mm_set1_pi16(left[i+2]), _mm_set1_pi16(left[i+3]));*/
        row = paeth_pred(&l16, &t0, &tl);

        ((uint64_t*)dst)[0] = _mm256_extract_epi64(row,0);
        dst += stride;
        ((uint64_t*)dst)[0] = _mm256_extract_epi64(row,1);
        dst += stride;
        ((uint64_t*)dst)[0] = _mm256_extract_epi64(row,2);
        dst += stride;
        ((uint64_t*)dst)[0] = _mm256_extract_epi64(row,3);
        dst += stride;
    }
}
void aom_highbd_paeth_predictor_4x8_avx2(uint16_t *dst, ptrdiff_t stride,
    int bw, int bh, const uint16_t *above, const uint16_t *left, int bd) {
    const __m256i t0 = _mm256_set1_epi64x(((uint64_t*)above)[0]);

    const __m256i tl = _mm256_set1_epi16(above[-1]);

    __m256i l16, row;

    int i;
    for (i = 0; i < 8; i += 4) {
        l16 = _mm256_setr_epi16(left[i], left[i], left[i], left[i],
            left[i + 1], left[i + 1], left[i + 1], left[i + 1],
            left[i + 2], left[i + 2], left[i + 2], left[i + 2],
            left[i + 3], left[i + 3], left[i + 3], left[i + 3]);
        /*l16 = _mm256_setr_epi64x(_mm_set1_pi16(left[i]), _mm_set1_pi16(left[i + 1]),
            _mm_set1_pi16(left[i+2]), _mm_set1_pi16(left[i+3]));*/
        row = paeth_pred(&l16, &t0, &tl);

        ((uint64_t*)dst)[0] = _mm256_extract_epi64(row, 0);
        dst += stride;
        ((uint64_t*)dst)[0] = _mm256_extract_epi64(row, 1);
        dst += stride;
        ((uint64_t*)dst)[0] = _mm256_extract_epi64(row, 2);
        dst += stride;
        ((uint64_t*)dst)[0] = _mm256_extract_epi64(row, 3);
        dst += stride;
    }
}
void aom_highbd_paeth_predictor_4x16_avx2(uint16_t *dst, ptrdiff_t stride,
    int bw, int bh, const uint16_t *above, const uint16_t *left, int bd) {
    const __m256i t0 = _mm256_set1_epi64x(((uint64_t*)above)[0]);

    const __m256i tl = _mm256_set1_epi16(above[-1]);

    __m256i l16, row;

    int i;
    for (i = 0; i < 16; i += 4) {
        l16 = _mm256_setr_epi16(left[i], left[i], left[i], left[i],
            left[i + 1], left[i + 1], left[i + 1], left[i + 1],
            left[i + 2], left[i + 2], left[i + 2], left[i + 2],
            left[i + 3], left[i + 3], left[i + 3], left[i + 3]);
        /*l16 = _mm256_setr_epi64x(_mm_set1_pi16(left[i]), _mm_set1_pi16(left[i + 1]),
            _mm_set1_pi16(left[i+2]), _mm_set1_pi16(left[i+3]));*/
        row = paeth_pred(&l16, &t0, &tl);

        ((uint64_t*)dst)[0] = _mm256_extract_epi64(row, 0);
        dst += stride;
        ((uint64_t*)dst)[0] = _mm256_extract_epi64(row, 1);
        dst += stride;
        ((uint64_t*)dst)[0] = _mm256_extract_epi64(row, 2);
        dst += stride;
        ((uint64_t*)dst)[0] = _mm256_extract_epi64(row, 3);
        dst += stride;
    }
}
//**************************END funciton AVX2*********************************
int TestCase_highbd_paeth_predictor(void** context, enum TEST_STAGE stage, int test_id, int verbose)
{
    struct contextX {

        uint16_t dst_src[100 * 100]; //Size to cover all tests
        uint16_t dst_cpy[100 * 100]; //Size to cover all tests
        ptrdiff_t stride;
        int bw; 
        int bh;
        uint16_t above[100];//Size to cover all tests
        uint16_t left[100];//Size to cover all tests
        int bd;
      

        uint32_t tx_type;
        fnhighbd_paeth_predictor a;
        fnhighbd_paeth_predictor b;
        int rand;
    } *cnt;

    if (stage == STAGE_GET_ID_MAX) {
        return paeth_size;  //Once test id
    }

    if (stage == STAGE_CREATE) {
        *context = malloc(sizeof(struct contextX));
        cnt = (struct contextX*)*context;

        memset(cnt->dst_src, 0, sizeof(cnt->dst_src));
        memset(cnt->dst_cpy, 0, sizeof(cnt->dst_cpy));
        cnt->stride = 100;


        printf("Create test: %s case %i/%i %s\n", __FUNCTION__, test_id + 1, (TestCase_highbd_paeth_predictor)(NULL, STAGE_GET_ID_MAX, 0, 0), Names[test_id]);


        cnt->a = highbd_paeth_predictor_c;

        cnt->rand = 0;
        cnt->tx_type = test_id;

        switch (cnt->tx_type) {
        case paeth_2x2:
            cnt->b = highbd_paeth_predictor_c;
            cnt->bw = 2;
            cnt->bh = 2;
            break;
        case paeth_4x4:
            cnt->b = aom_highbd_paeth_predictor_4x4_avx2;
            cnt->bw = 4;
            cnt->bh = 4;
            break;
        case paeth_8x8:
            cnt->b = aom_highbd_paeth_predictor_8x8_avx2;
            cnt->bw = 8;
            cnt->bh = 8;
            break;
        case paeth_16x16:
            cnt->b = aom_highbd_paeth_predictor_16x16_avx2;
            cnt->bw = 16;
            cnt->bh = 16;
            break;
        case paeth_32x32:
            cnt->b = aom_highbd_paeth_predictor_32x32_avx2;
            cnt->bw = 32;
            cnt->bh = 32;
            break;
        case paeth_64x64:
            cnt->b = aom_highbd_paeth_predictor_64x64_avx2;
            cnt->bw = 64;
            cnt->bh = 64;
            break;
        case paeth_4x8:
            cnt->b = aom_highbd_paeth_predictor_4x8_avx2;
            cnt->bw = 4;
            cnt->bh = 8;
            break;
        case paeth_4x16:
            cnt->b = aom_highbd_paeth_predictor_4x16_avx2;
            cnt->bw = 4;
            cnt->bh = 16;
            break;
        case paeth_8x4:
            cnt->b = aom_highbd_paeth_predictor_8x4_avx2;
            cnt->bw = 8;
            cnt->bh = 4;
            break;
        case paeth_8x16:
            cnt->b = aom_highbd_paeth_predictor_8x16_avx2;
            cnt->bw = 8;
            cnt->bh = 16;
            break;
        case paeth_8x32:
            cnt->b = aom_highbd_paeth_predictor_8x32_avx2;
            cnt->bw = 8;
            cnt->bh = 32;
            break;
        case paeth_16x4:
            cnt->b = aom_highbd_paeth_predictor_16x4_avx2;
            cnt->bw = 16;
            cnt->bh = 4;
            break;
        case paeth_16x8:
            cnt->b = aom_highbd_paeth_predictor_16x8_avx2;
            cnt->bw = 16;
            cnt->bh = 8;
            break;
        case paeth_16x32:
            cnt->b = aom_highbd_paeth_predictor_16x32_avx2;
            cnt->bw = 16;
            cnt->bh = 32;
            break;
        case paeth_16x64:
            cnt->b = aom_highbd_paeth_predictor_16x64_avx2;
            cnt->bw = 16;
            cnt->bh = 64;
            break;
        case paeth_32x8:
            cnt->b = aom_highbd_paeth_predictor_32x8_avx2;
            cnt->bw = 32;
            cnt->bh = 8;
            break;
        case paeth_32x16:
            cnt->b = aom_highbd_paeth_predictor_32x16_avx2;
            cnt->bw = 32;
            cnt->bh = 16;
            break;
        case paeth_32x64:
            cnt->b = aom_highbd_paeth_predictor_32x64_avx2;
            cnt->bw = 32;
            cnt->bh = 64;
            break;
        case paeth_64x16:
            cnt->b = aom_highbd_paeth_predictor_64x16_avx2;
            cnt->bw = 64;
            cnt->bh = 16;
            break;
        case paeth_64x32:
            cnt->b = aom_highbd_paeth_predictor_64x32_avx2;
            cnt->bw = 64;
            cnt->bh = 32;
            break;
        }


        return 0;
    }

    assert(*context);
    cnt = (struct contextX*)*context;

    switch (stage) {
    case STAGE_RAND_VALUES: {
        if (cnt->rand == 0) {

            for (int i = 0; i < 100; ++i) {
                cnt->above[i] = 1;
                cnt->left[i] = 2;
            }


        }
        else if (cnt->rand == 1) {

            for (int i = 0; i < 100; ++i) {
                cnt->above[i] = 1;
                cnt->left[i] = i;
            }

        }
        else if (cnt->rand == 2) {
  
            for (int i = 0; i < 100; ++i) {
                cnt->above[i] = i;
                cnt->left[i] = (2*i)%1023;
            }

        }
        else {

            for (int i = 0; i < 100; ++i) {
                cnt->above[i] = rand() % 1023;
                cnt->left[i] = rand() % 1023;
            }

            
        }


        ++cnt->rand;
        return 0;
    }
    case STAGE_EXECUTE_A: {
        //function  use index -1 for cnt->above array
        cnt->a(cnt->dst_src, cnt->stride, cnt->bw, cnt->bh, &cnt->above[1], cnt->left, cnt->bd);
        return 0;
    }
    case STAGE_EXECUTE_B: {
        cnt->b(cnt->dst_cpy, cnt->stride, cnt->bw, cnt->bh, &cnt->above[1], cnt->left, cnt->bd);
        return 0;
    }
    case STAGE_CHECK: {

        if (memcmp(cnt->dst_src, cnt->dst_cpy, sizeof(cnt->dst_src))) {
            printf("Invalid dst buffers!!! [ERROR]\n");
            return -1;
        }else if (!verbose) {
            printf("Correct dst buffers. [OK]\n");
        }


        return 0;

        break;
    }
    case STAGE_DESTROY: {

        free(*context);
        *context = NULL;
        return 0;
        break;
    }
    default:




        assert(0);
    }

    return -1;
}
