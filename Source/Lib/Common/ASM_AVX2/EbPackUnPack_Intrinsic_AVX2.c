/*
* Copyright(c) 2019 Intel Corporation
* SPDX - License - Identifier: BSD - 2 - Clause - Patent
*/

#include <emmintrin.h>
#include <immintrin.h>
#include <stdint.h>

void eb_enc_un_pack8_bit_data_avx2_intrin(uint16_t *in_16bit_buffer, uint32_t in_stride,
                                          uint8_t *out_8bit_buffer, uint32_t out_stride,
                                          uint32_t width, uint32_t height) {
    uint32_t x, y;

    __m256i ymm_00ff, in_pixel0, in_pixel1;
    __m256i in_pixel0_shft_r_2_u8;
    ymm_00ff = _mm256_set1_epi16(0x00FF);

    if (width == 8) {
        __m128i xmm_00ff, in_pixel0, in_pixel1, in_pixel1_shftR_2_u8;
        __m128i in_pixel0_shft_r_2_u8, in_pixel0_shft_r_2, in_pixel1_shft_r_2;
        xmm_00ff = _mm_set1_epi16(0x00FF);
        for (y = 0; y < height; y += 2) {
            in_pixel0 = _mm_loadu_si128((__m128i *)in_16bit_buffer);
            in_pixel1 = _mm_loadu_si128((__m128i *)(in_16bit_buffer + in_stride));

            in_pixel0_shft_r_2 = _mm_and_si128(_mm_srli_epi16(in_pixel0, 2), xmm_00ff);
            in_pixel1_shft_r_2 = _mm_and_si128(_mm_srli_epi16(in_pixel1, 2), xmm_00ff);

            in_pixel0_shft_r_2_u8 = _mm_packus_epi16(in_pixel0_shft_r_2, in_pixel0_shft_r_2);
            in_pixel1_shftR_2_u8 = _mm_packus_epi16(in_pixel1_shft_r_2, in_pixel1_shft_r_2);

            _mm_storel_epi64((__m128i *)out_8bit_buffer, in_pixel0_shft_r_2_u8);
            _mm_storel_epi64((__m128i *)(out_8bit_buffer + out_stride), in_pixel1_shftR_2_u8);

            out_8bit_buffer += 2 * out_stride;
            in_16bit_buffer += 2 * in_stride;
        }
    } else if (width == 16) {
        for (y = 0; y < height; y += 2) {
            in_pixel0 = _mm256_loadu_si256((__m256i *)in_16bit_buffer);
            in_pixel1 = _mm256_loadu_si256((__m256i *)(in_16bit_buffer + in_stride));

            in_pixel0_shft_r_2_u8 =
                _mm256_packus_epi16(_mm256_and_si256(_mm256_srli_epi16(in_pixel0, 2), ymm_00ff),
                                    _mm256_and_si256(_mm256_srli_epi16(in_pixel1, 2), ymm_00ff));

            *(uint64_t *)out_8bit_buffer       = _mm256_extract_epi64(in_pixel0_shft_r_2_u8, 0);
            *(uint64_t *)(out_8bit_buffer + 8) = _mm256_extract_epi64(in_pixel0_shft_r_2_u8, 2);
            *(uint64_t *)(out_8bit_buffer + out_stride) =
                _mm256_extract_epi64(in_pixel0_shft_r_2_u8, 1);
            *(uint64_t *)(out_8bit_buffer + out_stride + 8) =
                _mm256_extract_epi64(in_pixel0_shft_r_2_u8, 3);

            out_8bit_buffer += 2 * out_stride;
            in_16bit_buffer += 2 * in_stride;
        }
    } else if (width == 32) {
        __m256i in_pixel2, in_pixel3;
        __m256i out8_0_u8, out8_1_u8;

        for (y = 0; y < height; y += 2) {
            in_pixel0 = _mm256_loadu_si256((__m256i *)in_16bit_buffer);
            in_pixel1 = _mm256_loadu_si256((__m256i *)(in_16bit_buffer + 16));
            in_pixel2 = _mm256_loadu_si256((__m256i *)(in_16bit_buffer + in_stride));
            in_pixel3 = _mm256_loadu_si256((__m256i *)(in_16bit_buffer + in_stride + 16));

            out8_0_u8 =
                _mm256_packus_epi16(_mm256_and_si256(_mm256_srli_epi16(in_pixel0, 2), ymm_00ff),
                                    _mm256_and_si256(_mm256_srli_epi16(in_pixel1, 2), ymm_00ff));
            out8_1_u8 =
                _mm256_packus_epi16(_mm256_and_si256(_mm256_srli_epi16(in_pixel2, 2), ymm_00ff),
                                    _mm256_and_si256(_mm256_srli_epi16(in_pixel3, 2), ymm_00ff));

            *(uint64_t *)out_8bit_buffer        = _mm256_extract_epi64(out8_0_u8, 0);
            *(uint64_t *)(out_8bit_buffer + 8)  = _mm256_extract_epi64(out8_0_u8, 2);
            *(uint64_t *)(out_8bit_buffer + 16) = _mm256_extract_epi64(out8_0_u8, 1);
            *(uint64_t *)(out_8bit_buffer + 24) = _mm256_extract_epi64(out8_0_u8, 3);
            out_8bit_buffer += out_stride;

            *(uint64_t *)out_8bit_buffer        = _mm256_extract_epi64(out8_1_u8, 0);
            *(uint64_t *)(out_8bit_buffer + 8)  = _mm256_extract_epi64(out8_1_u8, 2);
            *(uint64_t *)(out_8bit_buffer + 16) = _mm256_extract_epi64(out8_1_u8, 1);
            *(uint64_t *)(out_8bit_buffer + 24) = _mm256_extract_epi64(out8_1_u8, 3);
            out_8bit_buffer += out_stride;
            in_16bit_buffer += 2 * in_stride;
        }
    } else if (width == 64) {
        __m256i in_pixel2, in_pixel3;
        __m256i out8_0_u8, out8_1_u8;

        for (y = 0; y < height; ++y) {
            in_pixel0 = _mm256_loadu_si256((__m256i *)in_16bit_buffer);
            in_pixel1 = _mm256_loadu_si256((__m256i *)(in_16bit_buffer + 16));
            in_pixel2 = _mm256_loadu_si256((__m256i *)(in_16bit_buffer + 32));
            in_pixel3 = _mm256_loadu_si256((__m256i *)(in_16bit_buffer + 48));

            out8_0_u8 =
                _mm256_packus_epi16(_mm256_and_si256(_mm256_srli_epi16(in_pixel0, 2), ymm_00ff),
                                    _mm256_and_si256(_mm256_srli_epi16(in_pixel1, 2), ymm_00ff));
            out8_1_u8 =
                _mm256_packus_epi16(_mm256_and_si256(_mm256_srli_epi16(in_pixel2, 2), ymm_00ff),
                                    _mm256_and_si256(_mm256_srli_epi16(in_pixel3, 2), ymm_00ff));

            *(uint64_t *)out_8bit_buffer        = _mm256_extract_epi64(out8_0_u8, 0);
            *(uint64_t *)(out_8bit_buffer + 8)  = _mm256_extract_epi64(out8_0_u8, 2);
            *(uint64_t *)(out_8bit_buffer + 16) = _mm256_extract_epi64(out8_0_u8, 1);
            *(uint64_t *)(out_8bit_buffer + 24) = _mm256_extract_epi64(out8_0_u8, 3);
            *(uint64_t *)(out_8bit_buffer + 32) = _mm256_extract_epi64(out8_1_u8, 0);
            *(uint64_t *)(out_8bit_buffer + 40) = _mm256_extract_epi64(out8_1_u8, 2);
            *(uint64_t *)(out_8bit_buffer + 48) = _mm256_extract_epi64(out8_1_u8, 1);
            *(uint64_t *)(out_8bit_buffer + 56) = _mm256_extract_epi64(out8_1_u8, 3);

            out_8bit_buffer += out_stride;
            in_16bit_buffer += in_stride;
        }
    } else {
        uint32_t in_stride_diff  = (2 * in_stride) - width;
        uint32_t out_stride_diff = (2 * out_stride) - width;

        uint32_t in_stride_diff64  = in_stride - width;
        uint32_t out_stride_diff64 = out_stride - width;

        if (!(width & 63)) {
            __m256i in_pixel2, in_pixel3;
            __m256i out8_0_u8, out8_1_u8;

            for (x = 0; x < height; x += 1) {
                for (y = 0; y < width; y += 64) {
                    in_pixel0 = _mm256_loadu_si256((__m256i *)in_16bit_buffer);
                    in_pixel1 = _mm256_loadu_si256((__m256i *)(in_16bit_buffer + 16));
                    in_pixel2 = _mm256_loadu_si256((__m256i *)(in_16bit_buffer + 32));
                    in_pixel3 = _mm256_loadu_si256((__m256i *)(in_16bit_buffer + 48));

                    out8_0_u8 = _mm256_packus_epi16(
                        _mm256_and_si256(_mm256_srli_epi16(in_pixel0, 2), ymm_00ff),
                        _mm256_and_si256(_mm256_srli_epi16(in_pixel1, 2), ymm_00ff));
                    out8_1_u8 = _mm256_packus_epi16(
                        _mm256_and_si256(_mm256_srli_epi16(in_pixel2, 2), ymm_00ff),
                        _mm256_and_si256(_mm256_srli_epi16(in_pixel3, 2), ymm_00ff));

                    *(uint64_t *)out_8bit_buffer        = _mm256_extract_epi64(out8_0_u8, 0);
                    *(uint64_t *)(out_8bit_buffer + 8)  = _mm256_extract_epi64(out8_0_u8, 2);
                    *(uint64_t *)(out_8bit_buffer + 16) = _mm256_extract_epi64(out8_0_u8, 1);
                    *(uint64_t *)(out_8bit_buffer + 24) = _mm256_extract_epi64(out8_0_u8, 3);
                    *(uint64_t *)(out_8bit_buffer + 32) = _mm256_extract_epi64(out8_1_u8, 0);
                    *(uint64_t *)(out_8bit_buffer + 40) = _mm256_extract_epi64(out8_1_u8, 2);
                    *(uint64_t *)(out_8bit_buffer + 48) = _mm256_extract_epi64(out8_1_u8, 1);
                    *(uint64_t *)(out_8bit_buffer + 56) = _mm256_extract_epi64(out8_1_u8, 3);

                    out_8bit_buffer += 64;
                    in_16bit_buffer += 64;
                }
                in_16bit_buffer += in_stride_diff64;
                out_8bit_buffer += out_stride_diff64;
            }
        } else if (!(width & 31)) {
            __m256i in_pixel2, in_pixel3;
            __m256i out8_0_u8, out8_1_u8;

            for (x = 0; x < height; x += 2) {
                for (y = 0; y < width; y += 32) {
                    in_pixel0 = _mm256_loadu_si256((__m256i *)in_16bit_buffer);
                    in_pixel1 = _mm256_loadu_si256((__m256i *)(in_16bit_buffer + 16));
                    in_pixel2 = _mm256_loadu_si256((__m256i *)(in_16bit_buffer + in_stride));
                    in_pixel3 = _mm256_loadu_si256((__m256i *)(in_16bit_buffer + in_stride + 16));

                    out8_0_u8 = _mm256_packus_epi16(
                        _mm256_and_si256(_mm256_srli_epi16(in_pixel0, 2), ymm_00ff),
                        _mm256_and_si256(_mm256_srli_epi16(in_pixel1, 2), ymm_00ff));
                    out8_1_u8 = _mm256_packus_epi16(
                        _mm256_and_si256(_mm256_srli_epi16(in_pixel2, 2), ymm_00ff),
                        _mm256_and_si256(_mm256_srli_epi16(in_pixel3, 2), ymm_00ff));

                    *(uint64_t *)out_8bit_buffer        = _mm256_extract_epi64(out8_0_u8, 0);
                    *(uint64_t *)(out_8bit_buffer + 8)  = _mm256_extract_epi64(out8_0_u8, 2);
                    *(uint64_t *)(out_8bit_buffer + 16) = _mm256_extract_epi64(out8_0_u8, 1);
                    *(uint64_t *)(out_8bit_buffer + 24) = _mm256_extract_epi64(out8_0_u8, 3);

                    *(uint64_t *)(out_8bit_buffer + out_stride) =
                        _mm256_extract_epi64(out8_1_u8, 0);
                    *(uint64_t *)(out_8bit_buffer + out_stride + 8) =
                        _mm256_extract_epi64(out8_1_u8, 2);
                    *(uint64_t *)(out_8bit_buffer + out_stride + 16) =
                        _mm256_extract_epi64(out8_1_u8, 1);
                    *(uint64_t *)(out_8bit_buffer + out_stride + 24) =
                        _mm256_extract_epi64(out8_1_u8, 3);

                    out_8bit_buffer += 32;
                    in_16bit_buffer += 32;
                }
                in_16bit_buffer += in_stride_diff;
                out_8bit_buffer += out_stride_diff;
            }
        } else if (!(width & 15)) {
            for (x = 0; x < height; x += 2) {
                for (y = 0; y < width; y += 16) {
                    in_pixel0 = _mm256_loadu_si256((__m256i *)in_16bit_buffer);
                    in_pixel1 = _mm256_loadu_si256((__m256i *)(in_16bit_buffer + in_stride));

                    in_pixel0_shft_r_2_u8 = _mm256_packus_epi16(
                        _mm256_and_si256(_mm256_srli_epi16(in_pixel0, 2), ymm_00ff),
                        _mm256_and_si256(_mm256_srli_epi16(in_pixel1, 2), ymm_00ff));

                    *(uint64_t *)out_8bit_buffer = _mm256_extract_epi64(in_pixel0_shft_r_2_u8, 0);
                    *(uint64_t *)(out_8bit_buffer + 8) =
                        _mm256_extract_epi64(in_pixel0_shft_r_2_u8, 2);
                    *(uint64_t *)(out_8bit_buffer + out_stride) =
                        _mm256_extract_epi64(in_pixel0_shft_r_2_u8, 1);
                    *(uint64_t *)(out_8bit_buffer + out_stride + 8) =
                        _mm256_extract_epi64(in_pixel0_shft_r_2_u8, 3);

                    out_8bit_buffer += 16;
                    in_16bit_buffer += 16;
                }
                in_16bit_buffer += in_stride_diff;
                out_8bit_buffer += out_stride_diff;
            }
        } else if (!(width & 7)) {
            __m128i xmm_00ff, in_pixel0, in_pixel1, in_pixel1_shftR_2_u8;
            __m128i in_pixel0_shft_r_2_u8, in_pixel0_shft_r_2, in_pixel1_shft_r_2;
            xmm_00ff = _mm_set1_epi16(0x00FF);
            for (x = 0; x < height; x += 2) {
                for (y = 0; y < width; y += 8) {
                    in_pixel0 = _mm_loadu_si128((__m128i *)in_16bit_buffer);
                    in_pixel1 = _mm_loadu_si128((__m128i *)(in_16bit_buffer + in_stride));

                    in_pixel0_shft_r_2 = _mm_and_si128(_mm_srli_epi16(in_pixel0, 2), xmm_00ff);
                    in_pixel1_shft_r_2 = _mm_and_si128(_mm_srli_epi16(in_pixel1, 2), xmm_00ff);

                    in_pixel0_shft_r_2_u8 = _mm_packus_epi16(in_pixel0_shft_r_2, in_pixel0_shft_r_2);
                    in_pixel1_shftR_2_u8 = _mm_packus_epi16(in_pixel1_shft_r_2, in_pixel1_shft_r_2);

                    _mm_storel_epi64((__m128i *)out_8bit_buffer, in_pixel0_shft_r_2_u8);
                    _mm_storel_epi64((__m128i *)(out_8bit_buffer + out_stride),
                                     in_pixel1_shftR_2_u8);

                    out_8bit_buffer += 8;
                    in_16bit_buffer += 8;
                }
                in_16bit_buffer += in_stride_diff;
                out_8bit_buffer += out_stride_diff;
            }
        } else {
            __m128i xmm_00ff, in_pixel0, in_pixel1, in_pixel1_shftR_2_u8;
            __m128i in_pixel0_shft_r_2_u8, in_pixel0_shft_r_2, in_pixel1_shft_r_2;
            xmm_00ff             = _mm_set1_epi16(0x00FF);
            uint32_t width_down4 = width & (~0x3);
            uint16_t in_pixel;
            for (x = 0; x < height; x += 2) {
                for (y = 0; y < width_down4; y += 4) {
                    in_pixel0 = _mm_loadl_epi64((__m128i *)in_16bit_buffer);
                    in_pixel1 = _mm_loadl_epi64((__m128i *)(in_16bit_buffer + in_stride));

                    in_pixel0_shft_r_2 = _mm_and_si128(_mm_srli_epi16(in_pixel0, 2), xmm_00ff);
                    in_pixel1_shft_r_2 = _mm_and_si128(_mm_srli_epi16(in_pixel1, 2), xmm_00ff);

                    in_pixel0_shft_r_2_u8 = _mm_packus_epi16(in_pixel0_shft_r_2, in_pixel0_shft_r_2);
                    in_pixel1_shftR_2_u8 = _mm_packus_epi16(in_pixel1_shft_r_2, in_pixel1_shft_r_2);

                    *(uint32_t *)out_8bit_buffer = _mm_cvtsi128_si32(in_pixel0_shft_r_2_u8);
                    *(uint32_t *)(out_8bit_buffer + out_stride) =
                        _mm_cvtsi128_si32(in_pixel1_shftR_2_u8);

                    out_8bit_buffer += 4;
                    in_16bit_buffer += 4;
                }

                /* Calculate lefts pixels in 2 lines,
                 * when width is not divided by 4.
                 */
                for (; y < width; y++) {
                    in_pixel                        = *in_16bit_buffer;
                    *out_8bit_buffer                = (uint8_t)(in_pixel >> 2);
                    in_pixel                        = *(in_16bit_buffer + in_stride);
                    *(out_8bit_buffer + out_stride) = (uint8_t)(in_pixel >> 2);
                    ++out_8bit_buffer;
                    ++in_16bit_buffer;
                }

                in_16bit_buffer += in_stride_diff;
                out_8bit_buffer += out_stride_diff;
            }
        }
    }
}
