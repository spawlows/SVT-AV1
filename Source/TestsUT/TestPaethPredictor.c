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
            cnt->b = highbd_paeth_predictor_c;
            cnt->bw = 4;
            cnt->bh = 4;
            break;
        case paeth_8x8:
            cnt->b = highbd_paeth_predictor_c;
            cnt->bw = 8;
            cnt->bh = 8;
            break;
        case paeth_16x16:
            cnt->b = highbd_paeth_predictor_c;
            cnt->bw = 16;
            cnt->bh = 16;
            break;
        case paeth_32x32:
            cnt->b = highbd_paeth_predictor_c;
            cnt->bw = 32;
            cnt->bh = 32;
            break;
        case paeth_64x64:
            cnt->b = highbd_paeth_predictor_c;
            cnt->bw = 64;
            cnt->bh = 64;
            break;
        case paeth_4x8:
            cnt->b = highbd_paeth_predictor_c;
            cnt->bw = 4;
            cnt->bh = 8;
            break;
        case paeth_4x16:
            cnt->b = highbd_paeth_predictor_c;
            cnt->bw = 4;
            cnt->bh = 16;
            break;
        case paeth_8x4:
            cnt->b = highbd_paeth_predictor_c;
            cnt->bw = 8;
            cnt->bh = 4;
            break;
        case paeth_8x16:
            cnt->b = highbd_paeth_predictor_c;
            cnt->bw = 8;
            cnt->bh = 16;
            break;
        case paeth_8x32:
            cnt->b = highbd_paeth_predictor_c;
            cnt->bw = 8;
            cnt->bh = 32;
            break;
        case paeth_16x4:
            cnt->b = highbd_paeth_predictor_c;
            cnt->bw = 16;
            cnt->bh = 4;
            break;
        case paeth_16x8:
            cnt->b = highbd_paeth_predictor_c;
            cnt->bw = 16;
            cnt->bh = 8;
            break;
        case paeth_16x32:
            cnt->b = highbd_paeth_predictor_c;
            cnt->bw = 16;
            cnt->bh = 32;
            break;
        case paeth_16x64:
            cnt->b = highbd_paeth_predictor_c;
            cnt->bw = 16;
            cnt->bh = 64;
            break;
        case paeth_32x8:
            cnt->b = highbd_paeth_predictor_c;
            cnt->bw = 32;
            cnt->bh = 8;
            break;
        case paeth_32x16:
            cnt->b = highbd_paeth_predictor_c;
            cnt->bw = 32;
            cnt->bh = 16;
            break;
        case paeth_32x64:
            cnt->b = highbd_paeth_predictor_c;
            cnt->bw = 32;
            cnt->bh = 64;
            break;
        case paeth_64x16:
            cnt->b = highbd_paeth_predictor_c;
            cnt->bw = 64;
            cnt->bh = 16;
            break;
        case paeth_64x32:
            cnt->b = highbd_paeth_predictor_c;
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
