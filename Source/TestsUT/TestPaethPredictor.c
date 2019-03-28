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
#include <stdint.h>
#include <assert.h>

#include "TestCore.h"
#include "EbDefinitions.h"


///////////////////////////fill rect
typedef void(*fnfill_rect)(uint16_t *dst, int32_t dstride, int32_t v, int32_t h,
    uint16_t x);

static INLINE void fill_rect(uint16_t *dst, int32_t dstride, int32_t v, int32_t h,
    uint16_t x) {
    for (int32_t i = 0; i < v; i++) {
        for (int32_t j = 0; j < h; j++) {
            dst[i * dstride + j] = x;
        }
    }
}

static /*INLINE*/ void  fill_rect_avx2(uint16_t *dst, int32_t dstride, int32_t v, int32_t h,
    uint16_t x) {
    int32_t i, j;
    __m256i x_avx2 = _mm256_set1_epi16(x);
    __m128i x_avx = _mm_set1_epi16(x);
    __m128i mask = _mm_set_epi64x(0, UINT64_MAX);

    //if (h >= 32) { //uncoment this section for good performance
    //    for (i = 0; i < v; i++) { //wrong code here, fix it
    //        for (j=0; j < h; j++) {
    //            dst[i * dstride + j] = x;
    //        }
    //    }
    //}

    if (h >= 16) {
        for (i = 0; i < v; i++) {
            _mm256_storeu_si256((__m256i*)(dst + i * dstride), x_avx2);
            for (j=16; j < h; j++) {
                dst[i * dstride + j] = x;
            }

        }
    }
    else if(h >= 8){
        for (i = 0; i < v/2; i++) {
            _mm256_storeu2_m128i((__m128i*)(dst + i*2 * dstride),
                (__m128i*)(dst + (i*2+1) * dstride), x_avx2);
            for (j=8; j < h; j++) {
                dst[i*2 * dstride + j] = x;
                dst[(i*2+1) * dstride + j] = x;
            }
        }
        if (v % 2) {
            i*=2;
             _mm_storeu_si128((__m128i*)(dst+ i * dstride), x_avx);
            for (j=8; j < h; j++) {
                dst[i * dstride + j] = x;
            }
        }
    }
    else if (h >= 4) {
        for (i = 0; i < v; i++) {
            _mm_maskstore_epi64((uint64_t*)(dst + i * dstride), mask, x_avx);
            for (j=4; j < h; j++) {
                dst[i * dstride + j] = x;
            }
        }
    }
    else {
        for (i = 0; i < v; i++) {
            for (j = 0; j < h; j++) {
                dst[i * dstride + j] = x;
            }
        }
    }
}


static /*INLINE*/ void  fill_rect_avx2_fast(uint16_t *dst, int32_t dstride, int32_t v, int32_t h,
    uint16_t x) {
    int32_t i, j;
    __m256i x_avx2 = _mm256_set1_epi16(x);
    __m128i x_avx = _mm_set1_epi16(x);
    __m128i mask = _mm_set_epi64x(0, UINT64_MAX);

    if (h >= 32) { //uncoment this section for good performance
        for (i = 0; i < v; i++) { //wrong code here, fix it
            for (j=0; j < h; j++) {
                dst[i * dstride + j] = x;
            }
        }
    }

    if (h >= 16) {
        for (i = 0; i < v; i++) {
            _mm256_storeu_si256((__m256i*)(dst + i * dstride), x_avx2);
            for (j = 16; j < h; j++) {
                dst[i * dstride + j] = x;
            }

        }
    }
    else if (h >= 8) {
        for (i = 0; i < v / 2; i++) {
            _mm256_storeu2_m128i((__m128i*)(dst + i * 2 * dstride),
                (__m128i*)(dst + (i * 2 + 1) * dstride), x_avx2);
            for (j = 8; j < h; j++) {
                dst[i * 2 * dstride + j] = x;
                dst[(i * 2 + 1) * dstride + j] = x;
            }
        }
        if (v % 2) {
            i *= 2;
            _mm_storeu_si128((__m128i*)(dst + i * dstride), x_avx);
            for (j = 8; j < h; j++) {
                dst[i * dstride + j] = x;
            }
        }
    }
    else if (h >= 4) {
        for (i = 0; i < v; i++) {
            _mm_maskstore_epi64((uint64_t*)(dst + i * dstride), mask, x_avx);
            for (j = 4; j < h; j++) {
                dst[i * dstride + j] = x;
            }
        }
    }
    else {
        for (i = 0; i < v; i++) {
            for (j = 0; j < h; j++) {
                dst[i * dstride + j] = x;
            }
        }
    }
}


int TestCase_fill_rect(void** context, enum TEST_STAGE stage, int test_id, int verbose)
{
    struct contextX {

        uint16_t dst_src[100 * 100]; //Size to cover all tests
        uint16_t dst_cpy[100 * 100]; //Size to cover all tests
        ptrdiff_t stride;
        int bw;
        int bh;

        uint16_t x;

        int bd;


        uint32_t tx_type;
        fnfill_rect a;
        fnfill_rect b;
        int rand;
    } *cnt;

    if (stage == STAGE_GET_ID_MAX) {
        return 2;  //Once test id
    }

    if (stage == STAGE_CREATE) {
        *context = malloc(sizeof(struct contextX));
        cnt = (struct contextX*)*context;

        memset(cnt->dst_src, 0, sizeof(cnt->dst_src));
        memset(cnt->dst_cpy, 0, sizeof(cnt->dst_cpy));
        cnt->stride = 100;


        printf("Create test: %s case %i/%i %s\n", __FUNCTION__, test_id + 1, (TestCase_fill_rect)(NULL, STAGE_GET_ID_MAX, 0, 0), "AA");


        cnt->a = fill_rect_avx2;
        cnt->b = fill_rect_avx2_fast;

        cnt->rand = 0;
        cnt->tx_type = test_id;

        cnt->bw = 81;
        cnt->bh = 17;

        return 0;
    }

    assert(*context);
    cnt = (struct contextX*)*context;

    switch (stage) {
    case STAGE_RAND_VALUES: {
        if (cnt->rand == 0) {
                cnt->x = rand();
        }
        else if (cnt->rand == 1) {
            cnt->x = rand();
        }
        else if (cnt->rand == 2) {
            cnt->x = rand();
        }
        else {
            cnt->x = rand();
        }


        ++cnt->rand;
        return 0;
    }
    case STAGE_EXECUTE_A: {
        cnt->a(cnt->dst_src, cnt->stride, cnt->bw, cnt->bh, cnt->x);
        return 0;
    }
    case STAGE_EXECUTE_B: {
        cnt->b(cnt->dst_cpy, cnt->stride, cnt->bw, cnt->bh, cnt->x);
        return 0;
    }
    case STAGE_CHECK: {

        if (memcmp(cnt->dst_src, cnt->dst_cpy, sizeof(cnt->dst_src))) {
            printf("Invalid dst buffers!!! [ERROR]\n");
            return -1;
        }
        else if (!verbose) {
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