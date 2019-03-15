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
//#include "EbTime.h"
#ifdef _WIN32
#include <Windows.h>
#endif
#include <immintrin.h>

#include "aom_dsp_rtcd.h"
#include "TestCore.h"


void av1_inv_txfm2d_add_16x16_avx2(const int32_t *coeff, uint16_t *output,
  int stride, TxType tx_type, int bd);


typedef void(*fnInv32x32)(const int32_t *input, uint16_t *output, int stride, TxType tx_type, int bd);

void av1_inv_txfm2d_add_16x16_c_CPY(const int32_t *input, uint16_t *output,
	int stride, TxType tx_type, int bd);


//static INLINE int64_t clamp64(int64_t value, int64_t low, int64_t high) {
//    return value < low ? low : (value > high ? high : value);
//}
static INLINE int32_t clamp_value(int32_t value, int8_t bit) {
    if (bit <= 0) return value;  // Do nothing for invalid clamp bit.
    const int64_t max_value = (1LL << (bit - 1)) - 1;
    const int64_t min_value = -(1LL << (bit - 1));
    return (int32_t)clamp64(value, min_value, max_value);
}

static INLINE void clamp_buf(int32_t *buf, int32_t size, int8_t bit) {
    for (int i = 0; i < size; ++i) {
        int32_t aa = buf[i];
        int32_t bb = clamp_value(buf[i], bit);
        if (aa != bb) {
            buf[i] = bb;
            //      printf("Change clam buffer!!! value: %d %d bit %d\n", aa, buf[i], bit);
        }
    }
}

static INLINE void abs_buf(int32_t *buf, int32_t size) {
    for (int i = 0; i < size; ++i) {
        if (buf[i] < 0) {
            buf[i] = -buf[i];
        }
    }
}

static INLINE void small_buf(int32_t *buf, int32_t size) {
    int x = 2896;
    x = 64;
    for (int i = 0; i < size; ++i) {
        if (buf[i] > (32767 / 2896)) {
            buf[i] /= x;
        }
    }
}


//Calim should be ignore in code after this operation
static void normalize_random_buf(int32_t *buf, int32_t size, const int bd) {
    abs_buf(buf, size);
    clamp_buf(buf, size, (int8_t)(bd + 8));
    clamp_buf(buf, size, (int8_t)(AOMMAX(bd + 6, 16)));
    small_buf(buf, size);

    unsigned int aa = 0;
    for (int i = 0; i < size; ++i) {
        aa += buf[i];
        aa += 1;
    }
    //  printf("SUM CRC: %u\n", aa);
}

int TestCaseInv16x16(void** context, enum TEST_STAGE stage, int test_id, int verbose)
{
    struct contextX {
        TxType tx_type;
        int bd;
        int stride;
        int dest_size;
        uint16_t *dest_c;
        uint16_t *dest_opt;
        int src_size;
        int32_t *src_org;
        int32_t *src_cpy;
        fnInv32x32 a;
        fnInv32x32 b;
        int rand;
    } *cnt;


    if (stage == STAGE_GET_ID_MAX) {
        return 7;  //Once test id
    }

    if (stage == STAGE_CREATE) {
        *context = malloc(sizeof(struct contextX));
        cnt = (struct contextX*)*context;

#define TEST_BITS 16


		char *name = "Unknown";
		cnt->tx_type = IDTX;
		switch (test_id) {
		case 0: cnt->tx_type = IDTX; name = "IDTX"; break;
		case 1: cnt->tx_type = V_DCT; name = "V_DCT"; break;
		case 2: cnt->tx_type = H_DCT; name = "H_DCT"; break;
		case 3: cnt->tx_type = V_ADST; name = "V_ADST"; break;
		case 4: cnt->tx_type = H_ADST; name = "H_ADST"; break;
		case 5: cnt->tx_type = V_FLIPADST; name = "V_FLIPADST"; break;
		case 6: cnt->tx_type = H_FLIPADST; name = "H_FLIPADST"; break;
		default:
				break;
		}


				//	

		printf("Create test: %s case %i/%i %s\n", __FUNCTION__, test_id + 1, (TestCaseInv16x16)(NULL, STAGE_GET_ID_MAX, 0, 0), name);




		//cnt->tx_type = IDTX;

        cnt->bd = 10; //8 //Check 8 in feature
      //  cnt->bd = 8; //8 //Check 8 in feature
        cnt->stride = 1234; //Some value not important biggest that TEST_BITS

        cnt->stride = TEST_BITS + 8;/*+ 8*/; //min 32// 1234; //Some value not important biggest that TEST_BITS
    //    cnt->stride = 64; //min 32// 1234; //Some value not important biggest that TEST_BITS

        cnt->dest_size = (TEST_BITS * TEST_BITS +  (TEST_BITS -1) * (cnt->stride - TEST_BITS) );
        cnt->dest_c = malloc(cnt->dest_size * sizeof(uint16_t));
        cnt->dest_opt = malloc(cnt->dest_size * sizeof(uint16_t));
        cnt->src_size = TEST_BITS * TEST_BITS * sizeof(int32_t);
        cnt->src_org = malloc(cnt->src_size);
        cnt->src_cpy = malloc(cnt->src_size);
        cnt->a = av1_inv_txfm2d_add_16x16_c;
       // cnt->a = av1_inv_txfm2d_add_16x16_c_CPY;
        //cnt->b = av1_inv_txfm2d_add_16x16_c;
	  //	cnt->b = av1_inv_txfm2d_add_16x16_c_CPY;
       cnt->b = av1_inv_txfm2d_add_16x16_avx2;
        cnt->rand = 0;
        return 0;
    }

    assert(*context);
    cnt = (struct contextX*)*context;

    switch (stage) {
    case STAGE_RAND_VALUES: {
        if (cnt->rand == 0) {
            for (int i = 0; i < TEST_BITS * TEST_BITS; ++i) {
                cnt->src_org[i] =0;
            }
        }
        else if (cnt->rand == 1) {
            for (int i = 0; i < TEST_BITS * TEST_BITS; ++i) {
                cnt->src_org[i] = 1;
            }
        } else  {
            for (int i = 0; i < cnt->src_size/ sizeof(int32_t); ++i) {
                //((uint8_t*)src_org)[i] = rand() & (256 -1);
                (cnt->src_org)[i] = rand() & (1024 - 1);
                //  ((uint8_t*)src_cpy)[i] = 0xAA;
            }

            //Normalize buffer to not go around clame
            normalize_random_buf(cnt->src_org, cnt->src_size / sizeof(int32_t), cnt->bd);

        }

        memcpy(cnt->src_cpy, cnt->src_org, cnt->src_size);

		//Operacja out of range testuje tylko clamp
		for (int i = 0; i < cnt->dest_size; ++i) {
			cnt->dest_c[i] = 0xabcd;
		}
		////Ustawia wszystko poza stride
		for (int i = 0; i < TEST_BITS; ++i) {
			for (int j = 0; j < TEST_BITS; ++j) {
				if (i < TEST_BITS / 2)
				{
					cnt->dest_c[i* cnt->stride + j] = i * TEST_BITS + j;//0x0 ;
					//cnt->dest_c[i* cnt->stride + j] = i * TEST_BITS + j;//0x0 ;
				}
			}
		}
		memcpy(cnt->dest_opt, cnt->dest_c, cnt->dest_size * sizeof(uint16_t));

        ++cnt->rand;
        return 0;
    }
    case STAGE_EXECUTE_A: {
        cnt->a(cnt->src_org, /*CONVERT_TO_SHORTPTR(*/cnt->dest_c/*)*/, cnt->stride, cnt->tx_type, cnt->bd);
        return 0;
    }
    case STAGE_EXECUTE_B: {
        cnt->b(cnt->src_org, /*CONVERT_TO_SHORTPTR(*/ cnt->dest_opt/*)*/, cnt->stride, cnt->tx_type, cnt->bd);
        return 0;
    }
    case STAGE_CHECK: {

       // return -1;
        if (memcmp(cnt->src_org, cnt->src_cpy, cnt->src_size)) {
            printf("Invalid SRC buffers!!! [ERROR]\n");
            return -1;
        }
        else if (!verbose) {
            printf("Correct SRC buffers. [OK]\n");
        }

        if (memcmp(cnt->dest_c, cnt->dest_opt, cnt->dest_size * sizeof(uint16_t))) {
              printf("Invalid DST buffers!!! [ERROR]\n");

            for (int j = 0; j < cnt->dest_size; j += cnt->stride) {

                for (int i = 0; i < TEST_BITS; ++i) {
                    if (cnt->dest_c[i + j] != cnt->dest_opt[i + j]) {
                        printf("FIRST->\n ");
                        break;
                    }
                    
                }
                printf("\n");

                for (int i = 0; i < TEST_BITS; ++i) {
                    printf("0x%04x ", cnt->dest_c[i + j]);
                }
                printf("\n");
                for (int i = 0; i < TEST_BITS; ++i) {
                    printf("0x%04x ", cnt->dest_opt[i + j]);
                }
                printf("\nS:");
                for (int i = TEST_BITS; i < cnt->stride && i < cnt->dest_size; ++i) {
                    printf("0x%04x ", cnt->dest_c[i + j]);
                }
                printf("\nS:");
                for (int i = TEST_BITS; i < cnt->stride&& i < cnt->dest_size; ++i) {
                    printf("0x%04x ", cnt->dest_opt[i + j]);
                }


                printf("\n");
                printf("\n");
            }
            return -1;
        }
        else if (!verbose) {
            printf("Correct DST buffers. [OK]\n");
        }
        return 0;

        break;
    }
    case STAGE_DESTROY: {
        free(cnt->dest_c);
        free(cnt->dest_opt);
        free(cnt->src_org);
        free(cnt->src_cpy);
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
//
//void av1_iidentity16_c(const int32_t *input, int32_t *output, int8_t cos_bit,
//	const int8_t *stage_range);
//
//void iidentity16_avx2_c(const __m256i *input)
//{
//	int8_t range = 250;
//
//	for (int i = 0; i < 32; i += 2) {
//		av1_iidentity16_c(&(input[i].m256i_i32[0]), &(input[i].m256i_i32[0]), 0, &range);
//	}
//
//	///*__m256i s4;
//	//for (int i = 0; i < 8; ++i) {
//	//    s4.m256i_i32[i] = 4;
//	//}
//
//	//output[0] = _mm256_mullo_epi32(input[0], s4);
//	//output[1] = _mm256_mullo_epi32(input[1], s4);
//	//output[2] = _mm256_mullo_epi32(input[2], s4);
//	//output[3] = _mm256_mullo_epi32(input[3], s4);
//	//*/
//
//	//for (int i =0; i<128;++i) {
//	//    output[i] = _mm256_slli_epi32(input[i], 2);
//	///*    output[i*4] = _mm256_slli_epi32(input[i * 4 + 0], 2);
//	//    output[i * 4+1] = _mm256_slli_epi32(input[i * 4 + 1], 2);
//	//    output[i * 4 + 2] = _mm256_slli_epi32(input[i * 4 + 2], 2);
//	//    output[i * 4 + 3] = _mm256_slli_epi32(input[i * 4 + 3], 2);*/
//	//}
//
//	//// ((__m128i*)output)[0] = _mm_sllv_epi32(((__m128i*)input)[0], 2);
//
//}




static void print_i32(int32_t*x, int nums)
{
	for (int i = 0; i < nums; i += 4) {
		printf("%2i: %2i %2i %2i %2i\n", i, x[i], x[i + 1], x[i + 2], x[i + 3]);
	}
}

void iidentity16_and_round_shift_avx2(const __m256i *input, int shift);

static const int NewSqrt2Bits = 12;
// 2^12 * sqrt(2)
static const int32_t NewSqrt2 = 5793;

//static INLINE void iidentity16_and_round_shift_avx2(__m256i *input, int shift)
//{
//	// Input values takes 18 bits, can be multiplied with NewSqrt2 in 32 bits space.
//	// Multiplied by half value NewSqrt2, instead (2*NewSqrt2), and round_shift() by one bit less (NewSqrt2Bits-1).
//	// round_shift(NewSqrt2Bits-1) and next round_shift(shift) in one pass.
//	const __m256i scalar = _mm256_set1_epi32(NewSqrt2);
//	__m256i rnding = _mm256_set1_epi32((1 << (NewSqrt2Bits - 2)) + (1 << (shift + NewSqrt2Bits - 2)));
//
//	for (int i = 0; i < 32; i++) {
//		input[i] = _mm256_mullo_epi32(input[i], scalar);
//		input[i] = _mm256_add_epi32(input[i], rnding);
//		input[i] = _mm256_srai_epi32(input[i], NewSqrt2Bits - 1 + shift);
//	}
//}
//
//int TestCaseIIdentity16(void** context, enum TEST_STAGE stage, int test_id, int verbose)
//{
//	struct context {
//		__m256i outA[32];
//		__m256i outB[32];
//		int rand;
//		int stableA;
//		int stableB;
//	} *cnt;
//	verbose = 1;
//
//	if (stage == STAGE_GET_ID_MAX) {
//		return 1;  //Once test id
//	}
//
//	if (stage == STAGE_CREATE) {
//		printf("Create test: %s case %i/%i\n", __FUNCTION__, test_id + 1, (TestCaseIIdentity32)(NULL, STAGE_GET_ID_MAX, 0, 0));
//		*context = malloc(sizeof(struct context));
//		cnt = (struct context*)*context;
//		cnt->rand = 0;
//		return 0;
//	}
//
//	assert(*context);
//	cnt = (struct context*)*context;
//
//	switch (stage) {
//	case STAGE_RAND_VALUES: {
//		int32_t mask = (1 << 18) - 1;
//		for (int i = 0; i < 32*8; ++i) {
//			((int32_t*)cnt->outA)[i] = (i + 1 << 30 - 15 + cnt->rand) & mask;
//			((int32_t*)cnt->outB)[i] = ((int32_t*)cnt->outA)[i];
//		}
//		cnt->rand++;
//		cnt->stableA = 1;
//		cnt->stableB = 1;
//		return 0;
//	}
//	case STAGE_EXECUTE_A: {
//		//if (!cnt->stableA) {
//		//	int32_t mask = (1 << 18) - 2;
//		//	for (int i = 0; i < 32 * 8; ++i) {
//		//		//((int32_t*)cnt->outA)[i] = clamp_value(((int32_t*)cnt->outA)[i], 18);
//		//	//	((int32_t*)cnt->outA)[i] &= mask;
//		//	}
//		//}
//		//cnt->stableA = 0;
//		iidentity16_avx2_c(((int32_t*)cnt->outA));
//		av1_round_shift_array_c(((int32_t*)cnt->outA), 32 * 8, 2);
//
//
//		return 0;
//	}
//	case STAGE_EXECUTE_B: {
//		//if (!cnt->stableB) {
//		//	int32_t mask = (1 << 18) - 2;
//		//	for (int i = 0; i < 32 * 8; ++i) {
//		//		/*((int32_t*)cnt->outB)[i] = clamp_value(((int32_t*)cnt->outB)[i], 18);*/
//		//	//	((int32_t*)cnt->outB)[i] &= mask;
//		//	}
//		//}
//		//cnt->stableB = 0;
//
//		iidentity16_and_round_shift_avx2(((int32_t*)cnt->outB), 2);
//		return 0;
//	}
//	case STAGE_CHECK: {
//		if (!verbose) { 
//			print_i32((int32_t*)cnt->outA, 32);
//			print_i32((int32_t*)cnt->outB, 32);
//		}
//
//
//		if (memcmp((int32_t*)cnt->outA, (int32_t*)cnt->outB, sizeof(cnt->outA))) {
//			printf("Invalid outA buffers!!! [ERROR]\n");
//
//			print_i32(cnt->outA,100);
//			print_i32(cnt->outB, 100);
//
//       
//			return -1;
//		}
//		
//		return 0;
//
//		break;
//	}
//	case STAGE_DESTROY: {
//		free(*context);
//		*context = NULL;
//		return 0;
//		break;
//	}
//	default:
//		assert(0);
//	}
//
//	return -1;
//}



