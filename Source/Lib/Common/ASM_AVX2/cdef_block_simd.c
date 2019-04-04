/*
* Copyright (c) 2016, Alliance for Open Media. All rights reserved
*
* This source code is subject to the terms of the BSD 2 Clause License and
* the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
* was not distributed with this source code in the LICENSE file, you can
* obtain it at www.aomedia.org/license/software. If the Alliance for Open
* Media Patent License 1.0 was not distributed with this source code in the
* PATENTS file, you can obtain it at www.aomedia.org/license/patent.
*/

#include "EbDefinitions.h"
#include <immintrin.h>

#include "aom_dsp_rtcd.h"
#include "EbCdef.h"
#include "EbBitstreamUnit.h"

#include "v128_intrinsics_x86.h"
#include "v256_intrinsics_x86.h"

#include "aom_dsp_rtcd.h"

#define SIMD_FUNC(name) name##_avx2



/* partial A is a 16-bit vector of the form:
[x8 x7 x6 x5 x4 x3 x2 x1] and partial B has the form:
[0  y1 y2 y3 y4 y5 y6 y7].
This function computes (x1^2+y1^2)*C1 + (x2^2+y2^2)*C2 + ...
(x7^2+y2^7)*C7 + (x8^2+0^2)*C8 where the C1..C8 constants are in const1
and const2. */
static INLINE v256 fold_mul_and_sum(v256 partial, v256 const_var) {
    partial = _mm256_shuffle_epi8(partial, _mm256_set_epi32(
        0x0f0e0100, 0x03020504, 0x07060908, 0x0b0a0d0c,
        0x0f0e0d0c, 0x0b0a0908, 0x07060504, 0x03020100));
    partial = _mm256_permutevar8x32_epi32(partial,
        _mm256_set_epi32(7, 3, 6, 2, 5, 1, 4, 0));
    partial = _mm256_shuffle_epi8(partial, _mm256_set_epi32(
        0x0f0e0b0a, 0x0d0c0908, 0x07060302, 0x05040100,
        0x0f0e0b0a, 0x0d0c0908, 0x07060302, 0x05040100));
    partial = _mm256_madd_epi16(partial, partial);
    partial = _mm256_mullo_epi32(partial, const_var);
    return partial;
}

static INLINE v128 hsum4(v128 x0, v128 x1, v128 x2, v128 x3) {
    v128 t0, t1, t2, t3;
    t0 = v128_ziplo_32(x1, x0);
    t1 = v128_ziplo_32(x3, x2);
    t2 = v128_ziphi_32(x1, x0);
    t3 = v128_ziphi_32(x3, x2);
    x0 = v128_ziplo_64(t1, t0);
    x1 = v128_ziphi_64(t1, t0);
    x2 = v128_ziplo_64(t3, t2);
    x3 = v128_ziphi_64(t3, t2);
    return v128_add_32(v128_add_32(x0, x1), v128_add_32(x2, x3));
}

/* Computes cost for directions 0, 5, 6 and 7. We can call this function again
to compute the remaining directions. */
static INLINE void compute_directions(v128 lines[8], int32_t tmp_cost1[4]) {

    v128 partial6;
    v128 tmp;

    v256 partial4;
    v256 partial5;
    v256 partial7;
    v256 tmp_avx2;

    /* Partial sums for lines 0 and 1. */
    partial4 = _mm256_insertf128_si256(_mm256_castsi128_si256(
        v128_shl_n_byte(lines[0], 14)), v128_shr_n_byte(lines[0], 2), 0x1);
    tmp_avx2 = _mm256_insertf128_si256(_mm256_castsi128_si256(
        v128_shl_n_byte(lines[1], 12)), v128_shr_n_byte(lines[1], 4), 0x1);
    partial4 = _mm256_add_epi16(partial4, tmp_avx2);

    tmp = v128_add_16(lines[0], lines[1]);

    partial5 = _mm256_insertf128_si256(_mm256_castsi128_si256(
        v128_shl_n_byte(tmp, 10)), v128_shr_n_byte(tmp, 6), 0x1);
    partial7 = _mm256_insertf128_si256(_mm256_castsi128_si256(
        v128_shl_n_byte(tmp, 4)), v128_shr_n_byte(tmp, 12), 0x1);

    partial6 = tmp;

    /* Partial sums for lines 2 and 3. */
    tmp_avx2 = _mm256_insertf128_si256(_mm256_castsi128_si256(
        v128_shl_n_byte(lines[2], 10)), v128_shr_n_byte(lines[2], 6), 0x1);
    partial4 = _mm256_add_epi16(partial4, tmp_avx2);
    tmp_avx2 = _mm256_insertf128_si256(_mm256_castsi128_si256(
        v128_shl_n_byte(lines[3], 8)), v128_shr_n_byte(lines[3], 8), 0x1);
    partial4 = _mm256_add_epi16(partial4, tmp_avx2);

    tmp = v128_add_16(lines[2], lines[3]);

    tmp_avx2 = _mm256_insertf128_si256(_mm256_castsi128_si256(
        v128_shl_n_byte(tmp, 8)), v128_shr_n_byte(tmp, 8), 0x1);
    partial5 = _mm256_add_epi16(partial5, tmp_avx2);
    tmp_avx2 = _mm256_insertf128_si256(_mm256_castsi128_si256(
        v128_shl_n_byte(tmp, 6)), v128_shr_n_byte(tmp, 10), 0x1);
    partial7 = _mm256_add_epi16(partial7, tmp_avx2);

    partial6 = v128_add_16(partial6, tmp);

    /* Partial sums for lines 4 and 5. */
    tmp_avx2 = _mm256_insertf128_si256(_mm256_castsi128_si256(
        v128_shl_n_byte(lines[4], 6)), v128_shr_n_byte(lines[4], 10), 0x1);
    partial4 = _mm256_add_epi16(partial4, tmp_avx2);
    tmp_avx2 = _mm256_insertf128_si256(_mm256_castsi128_si256(
        v128_shl_n_byte(lines[5], 4)), v128_shr_n_byte(lines[5], 12), 0x1);
    partial4 = _mm256_add_epi16(partial4, tmp_avx2);

    tmp = v128_add_16(lines[4], lines[5]);

    tmp_avx2 = _mm256_insertf128_si256(_mm256_castsi128_si256(
        v128_shl_n_byte(tmp, 6)), v128_shr_n_byte(tmp, 10), 0x1);
    partial5 = _mm256_add_epi16(partial5, tmp_avx2);
    tmp_avx2 = _mm256_insertf128_si256(_mm256_castsi128_si256(
        v128_shl_n_byte(tmp, 8)), v128_shr_n_byte(tmp, 8), 0x1);
    partial7 = _mm256_add_epi16(partial7, tmp_avx2);

    partial6 = v128_add_16(partial6, tmp);

    /* Partial sums for lines 6 and 7. */
    tmp_avx2 = _mm256_insertf128_si256(_mm256_castsi128_si256(
        v128_shl_n_byte(lines[6], 2)), v128_shr_n_byte(lines[6], 14), 0x1);
    partial4 = _mm256_add_epi16(partial4, tmp_avx2);
    tmp_avx2 = _mm256_insertf128_si256(_mm256_setzero_si256(), lines[7], 0x0);
    partial4 = _mm256_add_epi16(partial4, tmp_avx2);

    tmp = v128_add_16(lines[6], lines[7]);

    tmp_avx2 = _mm256_insertf128_si256(_mm256_castsi128_si256(
        v128_shl_n_byte(tmp, 4)), v128_shr_n_byte(tmp, 12), 0x1);
    partial5 = _mm256_add_epi16(partial5, tmp_avx2);
    tmp_avx2 = _mm256_insertf128_si256(_mm256_castsi128_si256(
        v128_shl_n_byte(tmp, 10)), v128_shr_n_byte(tmp, 6), 0x1);
    partial7 = _mm256_add_epi16(partial7, tmp_avx2);

    partial6 = v128_add_16(partial6, tmp);

    /* Compute costs in terms of partial sums. */
    partial4 = fold_mul_and_sum(partial4, _mm256_set_epi32(
        105, 120, 140, 168, 210, 280, 420, 840));
    partial7 = fold_mul_and_sum(partial7, _mm256_set_epi32(
        105, 105, 105, 140, 210, 420, 0, 0));
    partial5 = fold_mul_and_sum(partial5, _mm256_set_epi32(
        105, 105, 105, 140, 210, 420, 0, 0));

    partial6 = v128_madd_s16(partial6, partial6);
    partial6 = v128_mullo_s32(partial6, v128_dup_32(105));

    v128 a, b, c;
    a = _mm_add_epi32(_mm256_castsi256_si128(partial4),
        _mm256_extracti128_si256(partial4, 1));
    b = _mm_add_epi32(_mm256_castsi256_si128(partial5),
        _mm256_extracti128_si256(partial5, 1));
    c = _mm_add_epi32(_mm256_castsi256_si128(partial7),
        _mm256_extracti128_si256(partial7, 1));

    v128_store_unaligned(tmp_cost1, hsum4(a, b, partial6, c));
}

/* transpose and reverse the order of the lines -- equivalent to a 90-degree
counter-clockwise rotation of the pixels. */
static INLINE void array_reverse_transpose_8x8(v128 *in, v128 *res) {
    const v128 tr0_0 = v128_ziplo_16(in[1], in[0]);
    const v128 tr0_1 = v128_ziplo_16(in[3], in[2]);
    const v128 tr0_2 = v128_ziphi_16(in[1], in[0]);
    const v128 tr0_3 = v128_ziphi_16(in[3], in[2]);
    const v128 tr0_4 = v128_ziplo_16(in[5], in[4]);
    const v128 tr0_5 = v128_ziplo_16(in[7], in[6]);
    const v128 tr0_6 = v128_ziphi_16(in[5], in[4]);
    const v128 tr0_7 = v128_ziphi_16(in[7], in[6]);

    const v128 tr1_0 = v128_ziplo_32(tr0_1, tr0_0);
    const v128 tr1_1 = v128_ziplo_32(tr0_5, tr0_4);
    const v128 tr1_2 = v128_ziphi_32(tr0_1, tr0_0);
    const v128 tr1_3 = v128_ziphi_32(tr0_5, tr0_4);
    const v128 tr1_4 = v128_ziplo_32(tr0_3, tr0_2);
    const v128 tr1_5 = v128_ziplo_32(tr0_7, tr0_6);
    const v128 tr1_6 = v128_ziphi_32(tr0_3, tr0_2);
    const v128 tr1_7 = v128_ziphi_32(tr0_7, tr0_6);

    res[7] = v128_ziplo_64(tr1_1, tr1_0);
    res[6] = v128_ziphi_64(tr1_1, tr1_0);
    res[5] = v128_ziplo_64(tr1_3, tr1_2);
    res[4] = v128_ziphi_64(tr1_3, tr1_2);
    res[3] = v128_ziplo_64(tr1_5, tr1_4);
    res[2] = v128_ziphi_64(tr1_5, tr1_4);
    res[1] = v128_ziplo_64(tr1_7, tr1_6);
    res[0] = v128_ziphi_64(tr1_7, tr1_6);
}

int32_t SIMD_FUNC(cdef_find_dir)(const uint16_t *img, int32_t stride, int32_t *var,
    int32_t coeff_shift) {
    int32_t i;
    int32_t cost[8];
    int32_t best_cost = 0;
    int32_t best_dir = 0;
    v128 lines[8];
    v128 const_128 = v128_dup_16(128);
    for (i = 0; i < 8; i++) {
        lines[i] = v128_load_unaligned(&img[i * stride]);
        lines[i] =
            v128_sub_16(v128_shr_s16(lines[i], coeff_shift), const_128);
    }

    /* Compute "mostly vertical" directions. */
    compute_directions(lines, cost + 4);

    array_reverse_transpose_8x8(lines, lines);

    /* Compute "mostly horizontal" directions. */
    compute_directions(lines, cost);

    for (i = 0; i < 8; i++) {
        if (cost[i] > best_cost) {
            best_cost = cost[i];
            best_dir = i;
        }
    }

    /* Difference between the optimal variance and the variance along the
    orthogonal direction. Again, the sum(x^2) terms cancel out. */
    *var = best_cost - cost[(best_dir + 4) & 7];
    /* We'd normally divide by 840, but dividing by 1024 is close enough
    for what we're going to do with this. */
    *var >>= 10;
    return best_dir;
}

// sign(a-b) * min(abs(a-b), max(0, threshold - (abs(a-b) >> adjdamp)))
SIMD_INLINE v256 constrain16(v256 a, v256 b, v256 threshold,
    uint32_t adjdamp) {
    v256 diff = v256_sub_16(a, b);
    const v256 sign = v256_shr_n_s16(diff, 15);
    diff = v256_abs_s16(diff);
    const v256 s =
        v256_ssub_u16(threshold, v256_shr_u16(diff, adjdamp));
    return v256_xor(v256_add_16(sign, v256_min_s16(diff, s)), sign);
}

// sign(a - b) * min(abs(a - b), max(0, strength - (abs(a - b) >> adjdamp)))
SIMD_INLINE v128 constrain(v256 a, v256 b, uint32_t strength,
    uint32_t adjdamp) {
    const v256 diff16 = v256_sub_16(a, b);
    v128 diff = v128_pack_s16_s8(v256_high_v128(diff16), v256_low_v128(diff16));
    const v128 sign = v128_cmplt_s8(diff, v128_zero());
    diff = v128_abs_s8(diff);
    return v128_xor(
        v128_add_8(sign,
            v128_min_u8(diff, v128_ssub_u8(v128_dup_8(strength),
                v128_shr_u8(diff, adjdamp)))),
        sign);
}

void SIMD_FUNC(cdef_filter_block_4x4_8)(uint8_t *dst, int32_t dstride,
    const uint16_t *in, int32_t pri_strength,
    int32_t sec_strength, int32_t dir,
    int32_t pri_damping, int32_t sec_damping,
    /* AOM_UNUSED*/ int32_t max_unused,
    int32_t coeff_shift) {
    (void)max_unused;
    v128 p0, p1, p2, p3;
    v256 sum, row, tap, res;
    v256 max, min, large = v256_dup_16(CDEF_VERY_LARGE);
    int32_t po1 = cdef_directions[dir][0];
    int32_t po2 = cdef_directions[dir][1];
    int32_t s1o1 = cdef_directions[(dir + 2) & 7][0];
    int32_t s1o2 = cdef_directions[(dir + 2) & 7][1];
    int32_t s2o1 = cdef_directions[(dir + 6) & 7][0];
    int32_t s2o2 = cdef_directions[(dir + 6) & 7][1];

    const int32_t *pri_taps = cdef_pri_taps[(pri_strength >> coeff_shift) & 1];
    const int32_t *sec_taps = cdef_sec_taps[(pri_strength >> coeff_shift) & 1];

    if (pri_strength)
        pri_damping = AOMMAX(0, pri_damping - get_msb(pri_strength));
    if (sec_strength)
        sec_damping = AOMMAX(0, sec_damping - get_msb(sec_strength));

    sum = v256_zero();
    row = _mm256_set_epi64x(*(uint64_t*)(in),
        *(uint64_t*)(in + CDEF_BSTRIDE),
        *(uint64_t*)(in + 2 * CDEF_BSTRIDE),
        *(uint64_t*)(in + 3 * CDEF_BSTRIDE));
    max = min = row;

    if (pri_strength) {
        // Primary near taps
        tap = _mm256_set_epi64x(*(uint64_t*)(in + po1),
            *(uint64_t*)(in + CDEF_BSTRIDE + po1),
            *(uint64_t*)(in + 2 * CDEF_BSTRIDE + po1),
            *(uint64_t*)(in + 3 * CDEF_BSTRIDE + po1));
        max = v256_max_s16(max, v256_andn(tap, v256_cmpeq_16(tap, large)));
        min = v256_min_s16(min, tap);
        p0 = constrain(tap, row, pri_strength, pri_damping);
        tap = _mm256_set_epi64x(*(uint64_t*)(in - po1),
            *(uint64_t*)(in + CDEF_BSTRIDE - po1),
            *(uint64_t*)(in + 2 * CDEF_BSTRIDE - po1),
            *(uint64_t*)(in + 3 * CDEF_BSTRIDE - po1));
        max = v256_max_s16(max, v256_andn(tap, v256_cmpeq_16(tap, large)));
        min = v256_min_s16(min, tap);
        p1 = constrain(tap, row, pri_strength, pri_damping);

        // sum += pri_taps[0] * (p0 + p1)
        sum = v256_add_16(sum, v256_madd_us8(v256_dup_8(pri_taps[0]),
            v256_from_v128(v128_ziphi_8(p0, p1),
                v128_ziplo_8(p0, p1))));

        // Primary far taps
        tap = _mm256_set_epi64x(*(uint64_t*)(in + po2),
            *(uint64_t*)(in + CDEF_BSTRIDE + po2),
            *(uint64_t*)(in + 2 * CDEF_BSTRIDE + po2),
            *(uint64_t*)(in + 3 * CDEF_BSTRIDE + po2));
        max = v256_max_s16(max, v256_andn(tap, v256_cmpeq_16(tap, large)));
        min = v256_min_s16(min, tap);
        p0 = constrain(tap, row, pri_strength, pri_damping);
        tap = _mm256_set_epi64x(*(uint64_t*)(in - po2),
            *(uint64_t*)(in + CDEF_BSTRIDE - po2),
            *(uint64_t*)(in + 2 * CDEF_BSTRIDE - po2),
            *(uint64_t*)(in + 3 * CDEF_BSTRIDE - po2));
        max = v256_max_s16(max, v256_andn(tap, v256_cmpeq_16(tap, large)));
        min = v256_min_s16(min, tap);
        p1 = constrain(tap, row, pri_strength, pri_damping);

        // sum += pri_taps[1] * (p0 + p1)
        sum = v256_add_16(sum, v256_madd_us8(v256_dup_8(pri_taps[1]),
            v256_from_v128(v128_ziphi_8(p0, p1),
                v128_ziplo_8(p0, p1))));
    }

    if (sec_strength) {
        // Secondary near taps
        tap = _mm256_set_epi64x(*(uint64_t*)(in + s1o1),
            *(uint64_t*)(in + CDEF_BSTRIDE + s1o1),
            *(uint64_t*)(in + 2 * CDEF_BSTRIDE + s1o1),
            *(uint64_t*)(in + 3 * CDEF_BSTRIDE + s1o1));
        max = v256_max_s16(max, v256_andn(tap, v256_cmpeq_16(tap, large)));
        min = v256_min_s16(min, tap);
        p0 = constrain(tap, row, sec_strength, sec_damping);
        tap = _mm256_set_epi64x(*(uint64_t*)(in - s1o1),
            *(uint64_t*)(in + CDEF_BSTRIDE - s1o1),
            *(uint64_t*)(in + 2 * CDEF_BSTRIDE - s1o1),
            *(uint64_t*)(in + 3 * CDEF_BSTRIDE - s1o1));
        max = v256_max_s16(max, v256_andn(tap, v256_cmpeq_16(tap, large)));
        min = v256_min_s16(min, tap);
        p1 = constrain(tap, row, sec_strength, sec_damping);
        tap = _mm256_set_epi64x(*(uint64_t*)(in + s2o1),
            *(uint64_t*)(in + CDEF_BSTRIDE + s2o1),
            *(uint64_t*)(in + 2 * CDEF_BSTRIDE + s2o1),
            *(uint64_t*)(in + 3 * CDEF_BSTRIDE + s2o1));
        max = v256_max_s16(max, v256_andn(tap, v256_cmpeq_16(tap, large)));
        min = v256_min_s16(min, tap);
        p2 = constrain(tap, row, sec_strength, sec_damping);
        tap = _mm256_set_epi64x(*(uint64_t*)(in - s2o1),
            *(uint64_t*)(in + CDEF_BSTRIDE - s2o1),
            *(uint64_t*)(in + 2 * CDEF_BSTRIDE - s2o1),
            *(uint64_t*)(in + 3 * CDEF_BSTRIDE - s2o1));
        max = v256_max_s16(max, v256_andn(tap, v256_cmpeq_16(tap, large)));
        min = v256_min_s16(min, tap);
        p3 = constrain(tap, row, sec_strength, sec_damping);

        // sum += sec_taps[0] * (p0 + p1 + p2 + p3)
        p0 = v128_add_8(p0, p1);
        p2 = v128_add_8(p2, p3);
        sum = v256_add_16(sum, v256_madd_us8(v256_dup_8(sec_taps[0]),
            v256_from_v128(v128_ziphi_8(p0, p2),
                v128_ziplo_8(p0, p2))));

        // Secondary far taps
        tap = _mm256_set_epi64x(*(uint64_t*)(in + s1o2),
            *(uint64_t*)(in + CDEF_BSTRIDE + s1o2),
            *(uint64_t*)(in + 2 * CDEF_BSTRIDE + s1o2),
            *(uint64_t*)(in + 3 * CDEF_BSTRIDE + s1o2));
        max = v256_max_s16(max, v256_andn(tap, v256_cmpeq_16(tap, large)));
        min = v256_min_s16(min, tap);
        p0 = constrain(tap, row, sec_strength, sec_damping);
        tap = _mm256_set_epi64x(*(uint64_t*)(in - s1o2),
            *(uint64_t*)(in + CDEF_BSTRIDE - s1o2),
            *(uint64_t*)(in + 2 * CDEF_BSTRIDE - s1o2),
            *(uint64_t*)(in + 3 * CDEF_BSTRIDE - s1o2));
        max = v256_max_s16(max, v256_andn(tap, v256_cmpeq_16(tap, large)));
        min = v256_min_s16(min, tap);
        p1 = constrain(tap, row, sec_strength, sec_damping);
        tap = _mm256_set_epi64x(*(uint64_t*)(in + s2o2),
            *(uint64_t*)(in + CDEF_BSTRIDE + s2o2),
            *(uint64_t*)(in + 2 * CDEF_BSTRIDE + s2o2),
            *(uint64_t*)(in + 3 * CDEF_BSTRIDE + s2o2));
        max = v256_max_s16(max, v256_andn(tap, v256_cmpeq_16(tap, large)));
        min = v256_min_s16(min, tap);
        p2 = constrain(tap, row, sec_strength, sec_damping);
        tap = _mm256_set_epi64x(*(uint64_t*)(in - s2o2),
            *(uint64_t*)(in + CDEF_BSTRIDE - s2o2),
            *(uint64_t*)(in + 2 * CDEF_BSTRIDE - s2o2),
            *(uint64_t*)(in + 3 * CDEF_BSTRIDE - s2o2));
        max = v256_max_s16(max, v256_andn(tap, v256_cmpeq_16(tap, large)));
        min = v256_min_s16(min, tap);
        p3 = constrain(tap, row, sec_strength, sec_damping);

        // sum += sec_taps[1] * (p0 + p1 + p2 + p3)
        p0 = v128_add_8(p0, p1);
        p2 = v128_add_8(p2, p3);

        sum = v256_add_16(sum, v256_madd_us8(v256_dup_8(sec_taps[1]),
            v256_from_v128(v128_ziphi_8(p0, p2),
                v128_ziplo_8(p0, p2))));
    }

    // res = row + ((sum - (sum < 0) + 8) >> 4)
    sum = v256_add_16(sum, v256_cmplt_s16(sum, v256_zero()));
    res = v256_add_16(sum, v256_dup_16(8));
    res = v256_shr_n_s16(res, 4);
    res = v256_add_16(row, res);
    res = v256_min_s16(v256_max_s16(res, min), max);
    res = v256_pack_s16_u8(res, res);

    p0 = v256_low_v128(res);
    u32_store_aligned(&dst[0 * dstride], v64_high_u32(v128_high_v64(p0)));
    u32_store_aligned(&dst[1 * dstride], v64_low_u32(v128_high_v64(p0)));
    u32_store_aligned(&dst[2 * dstride], v64_high_u32(v128_low_v64(p0)));
    u32_store_aligned(&dst[3 * dstride], v64_low_u32(v128_low_v64(p0)));
}

void SIMD_FUNC(cdef_filter_block_8x8_8)(uint8_t *dst, int32_t dstride,
    const uint16_t *in, int32_t pri_strength,
    int32_t sec_strength, int32_t dir,
    int32_t pri_damping, int32_t sec_damping,
    /*AOM_UNUSED*/ int32_t max_unused,
    int32_t coeff_shift) {
    (void)max_unused;
    int32_t i;
    v128 p0, p1, p2, p3;
    v256 sum, row, res, tap;
    v256 max, min, large = v256_dup_16(CDEF_VERY_LARGE);
    int32_t po1 = cdef_directions[dir][0];
    int32_t po2 = cdef_directions[dir][1];
    int32_t s1o1 = cdef_directions[(dir + 2) & 7][0];
    int32_t s1o2 = cdef_directions[(dir + 2) & 7][1];
    int32_t s2o1 = cdef_directions[(dir + 6) & 7][0];
    int32_t s2o2 = cdef_directions[(dir + 6) & 7][1];

    const int32_t *pri_taps = cdef_pri_taps[(pri_strength >> coeff_shift) & 1];
    const int32_t *sec_taps = cdef_sec_taps[(pri_strength >> coeff_shift) & 1];
    v256 pri_taps_0 = v256_dup_8(pri_taps[0]);
    v256 pri_taps_1 = v256_dup_8(pri_taps[1]);
    v256 sec_taps_0 = v256_dup_8(sec_taps[0]);
    v256 sec_taps_1 = v256_dup_8(sec_taps[1]);
    v256 duplicate_8 = v256_dup_16(8);

    if (pri_strength)
        pri_damping = AOMMAX(0, pri_damping - get_msb(pri_strength));
    if (sec_strength)
        sec_damping = AOMMAX(0, sec_damping - get_msb(sec_strength));
    for (i = 0; i < 8; i += 2) {
        sum = v256_zero();
        row = _mm256_insertf128_si256(_mm256_castsi128_si256(
            _mm_loadu_si128((__m128i *)(in + (i + 1) * CDEF_BSTRIDE))),
            _mm_loadu_si128((__m128i *)(in + i * CDEF_BSTRIDE)), 0x1);

        max = min = row;
        // Primary near taps
        tap = _mm256_insertf128_si256(_mm256_castsi128_si256(
            _mm_loadu_si128((__m128i *)(in + (i + 1) * CDEF_BSTRIDE + po1))),
            _mm_loadu_si128((__m128i *)(in + i * CDEF_BSTRIDE + po1)), 0x1);
        max = v256_max_s16(max, v256_andn(tap, v256_cmpeq_16(tap, large)));
        min = v256_min_s16(min, tap);
        p0 = constrain(tap, row, pri_strength, pri_damping);
        tap = _mm256_insertf128_si256(_mm256_castsi128_si256(
            _mm_loadu_si128((__m128i *)(in + (i + 1) * CDEF_BSTRIDE - po1))),
            _mm_loadu_si128((__m128i *)(in + i * CDEF_BSTRIDE - po1)), 0x1);
        max = v256_max_s16(max, v256_andn(tap, v256_cmpeq_16(tap, large)));
        min = v256_min_s16(min, tap);
        p1 = constrain(tap, row, pri_strength, pri_damping);

        // sum += pri_taps[0] * (p0 + p1)
        sum = v256_add_16(sum, v256_madd_us8(pri_taps_0,
            v256_from_v128(v128_ziphi_8(p0, p1),
                v128_ziplo_8(p0, p1))));

        // Primary far taps
        tap = _mm256_insertf128_si256(_mm256_castsi128_si256(
            _mm_loadu_si128((__m128i *)(in + (i + 1) * CDEF_BSTRIDE + po2))),
            _mm_loadu_si128((__m128i *)(in + i * CDEF_BSTRIDE + po2)), 0x1);
        max = v256_max_s16(max, v256_andn(tap, v256_cmpeq_16(tap, large)));
        min = v256_min_s16(min, tap);
        p0 = constrain(tap, row, pri_strength, pri_damping);
        tap = _mm256_insertf128_si256(_mm256_castsi128_si256(
            _mm_loadu_si128((__m128i *)(in + (i + 1) * CDEF_BSTRIDE - po2))),
            _mm_loadu_si128((__m128i *)(in + i * CDEF_BSTRIDE - po2)), 0x1);
        max = v256_max_s16(max, v256_andn(tap, v256_cmpeq_16(tap, large)));
        min = v256_min_s16(min, tap);
        p1 = constrain(tap, row, pri_strength, pri_damping);

        // sum += pri_taps[1] * (p0 + p1)
        sum = v256_add_16(sum, v256_madd_us8(pri_taps_1,
            v256_from_v128(v128_ziphi_8(p0, p1),
                v128_ziplo_8(p0, p1))));

        // Secondary near taps
        tap = _mm256_insertf128_si256(_mm256_castsi128_si256(
            _mm_loadu_si128((__m128i *)(in + (i + 1) * CDEF_BSTRIDE + s1o1))),
            _mm_loadu_si128((__m128i *)(in + i * CDEF_BSTRIDE + s1o1)), 0x1);
        max = v256_max_s16(max, v256_andn(tap, v256_cmpeq_16(tap, large)));
        min = v256_min_s16(min, tap);
        p0 = constrain(tap, row, sec_strength, sec_damping);
        tap = _mm256_insertf128_si256(_mm256_castsi128_si256(
            _mm_loadu_si128((__m128i *)(in + (i + 1) * CDEF_BSTRIDE - s1o1))),
            _mm_loadu_si128((__m128i *)(in + i * CDEF_BSTRIDE - s1o1)), 0x1);
        max = v256_max_s16(max, v256_andn(tap, v256_cmpeq_16(tap, large)));
        min = v256_min_s16(min, tap);
        p1 = constrain(tap, row, sec_strength, sec_damping);
        tap = _mm256_insertf128_si256(_mm256_castsi128_si256(
            _mm_loadu_si128((__m128i *)(in + (i + 1) * CDEF_BSTRIDE + s2o1))),
            _mm_loadu_si128((__m128i *)(in + i * CDEF_BSTRIDE + s2o1)), 0x1);
        max = v256_max_s16(max, v256_andn(tap, v256_cmpeq_16(tap, large)));
        min = v256_min_s16(min, tap);
        p2 = constrain(tap, row, sec_strength, sec_damping);
        tap = _mm256_insertf128_si256(_mm256_castsi128_si256(
            _mm_loadu_si128((__m128i *)(in + (i + 1) * CDEF_BSTRIDE - s2o1))),
            _mm_loadu_si128((__m128i *)(in + i * CDEF_BSTRIDE - s2o1)), 0x1);
        max = v256_max_s16(max, v256_andn(tap, v256_cmpeq_16(tap, large)));
        min = v256_min_s16(min, tap);
        p3 = constrain(tap, row, sec_strength, sec_damping);

        // sum += sec_taps[0] * (p0 + p1 + p2 + p3)
        p0 = v128_add_8(p0, p1);
        p2 = v128_add_8(p2, p3);
        sum = v256_add_16(sum, v256_madd_us8(sec_taps_0,
            v256_from_v128(v128_ziphi_8(p0, p2),
                v128_ziplo_8(p0, p2))));

        // Secondary far taps
        tap = _mm256_insertf128_si256(_mm256_castsi128_si256(
            _mm_loadu_si128((__m128i *)(in + (i + 1) * CDEF_BSTRIDE + s1o2))),
            _mm_loadu_si128((__m128i *)(in + i * CDEF_BSTRIDE + s1o2)), 0x1);
        max = v256_max_s16(max, v256_andn(tap, v256_cmpeq_16(tap, large)));
        min = v256_min_s16(min, tap);
        p0 = constrain(tap, row, sec_strength, sec_damping);
        tap = _mm256_insertf128_si256(_mm256_castsi128_si256(
            _mm_loadu_si128((__m128i *)(in + (i + 1) * CDEF_BSTRIDE - s1o2))),
            _mm_loadu_si128((__m128i *)(in + i * CDEF_BSTRIDE - s1o2)), 0x1);
        max = v256_max_s16(max, v256_andn(tap, v256_cmpeq_16(tap, large)));
        min = v256_min_s16(min, tap);
        p1 = constrain(tap, row, sec_strength, sec_damping);
        tap = _mm256_insertf128_si256(_mm256_castsi128_si256(
            _mm_loadu_si128((__m128i *)(in + (i + 1) * CDEF_BSTRIDE + s2o2))),
            _mm_loadu_si128((__m128i *)(in + i * CDEF_BSTRIDE + s2o2)), 0x1);
        max = v256_max_s16(max, v256_andn(tap, v256_cmpeq_16(tap, large)));
        min = v256_min_s16(min, tap);
        p2 = constrain(tap, row, sec_strength, sec_damping);
        tap = _mm256_insertf128_si256(_mm256_castsi128_si256(
            _mm_loadu_si128((__m128i *)(in + (i + 1) * CDEF_BSTRIDE - s2o2))),
            _mm_loadu_si128((__m128i *)(in + i * CDEF_BSTRIDE - s2o2)), 0x1);
        max = v256_max_s16(max, v256_andn(tap, v256_cmpeq_16(tap, large)));
        min = v256_min_s16(min, tap);
        p3 = constrain(tap, row, sec_strength, sec_damping);

        // sum += sec_taps[1] * (p0 + p1 + p2 + p3)
        p0 = v128_add_8(p0, p1);
        p2 = v128_add_8(p2, p3);
        sum = v256_add_16(sum, v256_madd_us8(sec_taps_1,
            v256_from_v128(v128_ziphi_8(p0, p2),
                v128_ziplo_8(p0, p2))));

        // res = row + ((sum - (sum < 0) + 8) >> 4)
        sum = v256_add_16(sum, v256_cmplt_s16(sum, v256_zero()));
        res = v256_add_16(sum, duplicate_8);
        res = v256_shr_n_s16(res, 4);
        res = v256_add_16(row, res);
        res = v256_min_s16(v256_max_s16(res, min), max);
        res = v256_pack_s16_u8(res, res);

        *(uint64_t*)(dst + i * dstride) = _mm256_extract_epi64(res, 1);
        *(uint64_t*)(dst + (i + 1) * dstride) = _mm256_extract_epi64(res, 0);
    }
}

void SIMD_FUNC(cdef_filter_block_4x4_16)(uint16_t *dst, int32_t dstride,
    const uint16_t *in, int32_t pri_strength,
    int32_t sec_strength, int32_t dir,
    int32_t pri_damping, int32_t sec_damping,
    /*AOM_UNUSED*/ int32_t max_unused,
    int32_t coeff_shift) {
    (void)max_unused;
    v256 p0, p1, p2, p3, sum, row, res;
    v256 max, min, large = v256_dup_16(CDEF_VERY_LARGE);
    int32_t po1 = cdef_directions[dir][0];
    int32_t po2 = cdef_directions[dir][1];
    int32_t s1o1 = cdef_directions[(dir + 2) & 7][0];
    int32_t s1o2 = cdef_directions[(dir + 2) & 7][1];
    int32_t s2o1 = cdef_directions[(dir + 6) & 7][0];
    int32_t s2o2 = cdef_directions[(dir + 6) & 7][1];

    const int32_t *pri_taps = cdef_pri_taps[(pri_strength >> coeff_shift) & 1];
    const int32_t *sec_taps = cdef_sec_taps[(pri_strength >> coeff_shift) & 1];

    v256 pri_strength_256 = v256_dup_16(pri_strength);
    v256 sec_strength_256 = v256_dup_16(sec_strength);

    if (pri_strength)
        pri_damping = AOMMAX(0, pri_damping - get_msb(pri_strength));
    if (sec_strength)
        sec_damping = AOMMAX(0, sec_damping - get_msb(sec_strength));

    sum = v256_zero();
    row = _mm256_set_epi64x(*(uint64_t*)(in),
        *(uint64_t*)(in + 1 * CDEF_BSTRIDE),
        *(uint64_t*)(in + 2 * CDEF_BSTRIDE),
        *(uint64_t*)(in + 3 * CDEF_BSTRIDE));
    min = max = row;

    // Primary near taps
    p0 = _mm256_set_epi64x(*(uint64_t*)(in + po1),
        *(uint64_t*)(in + 1 * CDEF_BSTRIDE + po1),
        *(uint64_t*)(in + 2 * CDEF_BSTRIDE + po1),
        *(uint64_t*)(in + 3 * CDEF_BSTRIDE + po1));
    p1 = _mm256_set_epi64x(*(uint64_t*)(in - po1),
        *(uint64_t*)(in + 1 * CDEF_BSTRIDE - po1),
        *(uint64_t*)(in + 2 * CDEF_BSTRIDE - po1),
        *(uint64_t*)(in + 3 * CDEF_BSTRIDE - po1));

    max =
        v256_max_s16(v256_max_s16(max, v256_andn(p0, v256_cmpeq_16(p0, large))),
        v256_andn(p1, v256_cmpeq_16(p1, large)));
    min = v256_min_s16(v256_min_s16(min, p0), p1);
    p0 = constrain16(p0, row, pri_strength_256, pri_damping);
    p1 = constrain16(p1, row, pri_strength_256, pri_damping);

    // sum += pri_taps[0] * (p0 + p1)
    sum = v256_add_16(
        sum, v256_mullo_s16(v256_dup_16(pri_taps[0]), v256_add_16(p0, p1)));

    // Primary far taps
    p0 = _mm256_set_epi64x(*(uint64_t*)(in + po2),
        *(uint64_t*)(in + 1 * CDEF_BSTRIDE + po2),
        *(uint64_t*)(in + 2 * CDEF_BSTRIDE + po2),
        *(uint64_t*)(in + 3 * CDEF_BSTRIDE + po2));
    p1 = _mm256_set_epi64x(*(uint64_t*)(in - po2),
        *(uint64_t*)(in + 1 * CDEF_BSTRIDE - po2),
        *(uint64_t*)(in + 2 * CDEF_BSTRIDE - po2),
        *(uint64_t*)(in + 3 * CDEF_BSTRIDE - po2));
    max =
        v256_max_s16(v256_max_s16(max, v256_andn(p0, v256_cmpeq_16(p0, large))),
            v256_andn(p1, v256_cmpeq_16(p1, large)));
    min = v256_min_s16(v256_min_s16(min, p0), p1);
    p0 = constrain16(p0, row, pri_strength_256, pri_damping);
    p1 = constrain16(p1, row, pri_strength_256, pri_damping);

    // sum += pri_taps[1] * (p0 + p1)
    sum = v256_add_16(
        sum, v256_mullo_s16(v256_dup_16(pri_taps[1]), v256_add_16(p0, p1)));

    // Secondary near taps
    p0 = _mm256_set_epi64x(*(uint64_t*)(in + s1o1),
        *(uint64_t*)(in + 1 * CDEF_BSTRIDE + s1o1),
        *(uint64_t*)(in + 2 * CDEF_BSTRIDE + s1o1),
        *(uint64_t*)(in + 3 * CDEF_BSTRIDE + s1o1));
    p1 = _mm256_set_epi64x(*(uint64_t*)(in - s1o1),
        *(uint64_t*)(in + 1 * CDEF_BSTRIDE - s1o1),
        *(uint64_t*)(in + 2 * CDEF_BSTRIDE - s1o1),
        *(uint64_t*)(in + 3 * CDEF_BSTRIDE - s1o1));
    p2 = _mm256_set_epi64x(*(uint64_t*)(in + s2o1),
        *(uint64_t*)(in + 1 * CDEF_BSTRIDE + s2o1),
        *(uint64_t*)(in + 2 * CDEF_BSTRIDE + s2o1),
        *(uint64_t*)(in + 3 * CDEF_BSTRIDE + s2o1));
    p3 = _mm256_set_epi64x(*(uint64_t*)(in - s2o1),
        *(uint64_t*)(in + 1 * CDEF_BSTRIDE - s2o1),
        *(uint64_t*)(in + 2 * CDEF_BSTRIDE - s2o1),
        *(uint64_t*)(in + 3 * CDEF_BSTRIDE - s2o1));
    max =
        v256_max_s16(v256_max_s16(max, v256_andn(p0, v256_cmpeq_16(p0, large))),
            v256_andn(p1, v256_cmpeq_16(p1, large)));
    max =
        v256_max_s16(v256_max_s16(max, v256_andn(p2, v256_cmpeq_16(p2, large))),
            v256_andn(p3, v256_cmpeq_16(p3, large)));
    min = v256_min_s16(
        v256_min_s16(v256_min_s16(v256_min_s16(min, p0), p1), p2), p3);
    p0 = constrain16(p0, row, sec_strength_256, sec_damping);
    p1 = constrain16(p1, row, sec_strength_256, sec_damping);
    p2 = constrain16(p2, row, sec_strength_256, sec_damping);
    p3 = constrain16(p3, row, sec_strength_256, sec_damping);

    // sum += sec_taps[0] * (p0 + p1 + p2 + p3)
    sum = v256_add_16(sum, v256_mullo_s16(v256_dup_16(sec_taps[0]),
        v256_add_16(v256_add_16(p0, p1),
            v256_add_16(p2, p3))));

    // Secondary far taps
    p0 = _mm256_set_epi64x(*(uint64_t*)(in + s1o2),
        *(uint64_t*)(in + 1 * CDEF_BSTRIDE + s1o2),
        *(uint64_t*)(in + 2 * CDEF_BSTRIDE + s1o2),
        *(uint64_t*)(in + 3 * CDEF_BSTRIDE + s1o2));
    p1 = _mm256_set_epi64x(*(uint64_t*)(in - s1o2),
        *(uint64_t*)(in + 1 * CDEF_BSTRIDE - s1o2),
        *(uint64_t*)(in + 2 * CDEF_BSTRIDE - s1o2),
        *(uint64_t*)(in + 3 * CDEF_BSTRIDE - s1o2));
    p2 = _mm256_set_epi64x(*(uint64_t*)(in + s2o2),
        *(uint64_t*)(in + 1 * CDEF_BSTRIDE + s2o2),
        *(uint64_t*)(in + 2 * CDEF_BSTRIDE + s2o2),
        *(uint64_t*)(in + 3 * CDEF_BSTRIDE + s2o2));
    p3 = _mm256_set_epi64x(*(uint64_t*)(in - s2o2),
        *(uint64_t*)(in + 1 * CDEF_BSTRIDE - s2o2),
        *(uint64_t*)(in + 2 * CDEF_BSTRIDE - s2o2),
        *(uint64_t*)(in + 3 * CDEF_BSTRIDE - s2o2));
    max =
        v256_max_s16(v256_max_s16(max, v256_andn(p0, v256_cmpeq_16(p0, large))),
            v256_andn(p1, v256_cmpeq_16(p1, large)));
    max =
        v256_max_s16(v256_max_s16(max, v256_andn(p2, v256_cmpeq_16(p2, large))),
            v256_andn(p3, v256_cmpeq_16(p3, large)));
    min = v256_min_s16(
        v256_min_s16(v256_min_s16(v256_min_s16(min, p0), p1), p2), p3);
    p0 = constrain16(p0, row, sec_strength_256, sec_damping);
    p1 = constrain16(p1, row, sec_strength_256, sec_damping);
    p2 = constrain16(p2, row, sec_strength_256, sec_damping);
    p3 = constrain16(p3, row, sec_strength_256, sec_damping);

    // sum += sec_taps[1] * (p0 + p1 + p2 + p3)
    sum = v256_add_16(sum, v256_mullo_s16(v256_dup_16(sec_taps[1]),
        v256_add_16(v256_add_16(p0, p1),
            v256_add_16(p2, p3))));

    // res = row + ((sum - (sum < 0) + 8) >> 4)
    sum = v256_add_16(sum, v256_cmplt_s16(sum, v256_zero()));
    res = v256_add_16(sum, v256_dup_16(8));
    res = v256_shr_n_s16(res, 4);
    res = v256_add_16(row, res);
    res = v256_min_s16(v256_max_s16(res, min), max);

    *(uint64_t*)(dst) = _mm256_extract_epi64(res, 3);
    *(uint64_t*)(dst + 1 * dstride) = _mm256_extract_epi64(res, 2);
    *(uint64_t*)(dst + 2 * dstride) = _mm256_extract_epi64(res, 1);
    *(uint64_t*)(dst + 3 * dstride) = _mm256_extract_epi64(res, 0);
}

void SIMD_FUNC(cdef_filter_block_8x8_16)(uint16_t *dst, int32_t dstride,
    const uint16_t *in, int32_t pri_strength,
    int32_t sec_strength, int32_t dir,
    int32_t pri_damping, int32_t sec_damping,
    /*AOM_UNUSED*/ int32_t max_unused,
    int32_t coeff_shift) {
    (void)max_unused;
    int32_t i;
    v256 sum, p0, p1, p2, p3, row, res;
    v256 max, min, large = v256_dup_16(CDEF_VERY_LARGE);
    int32_t po1 = cdef_directions[dir][0];
    int32_t po2 = cdef_directions[dir][1];
    int32_t s1o1 = cdef_directions[(dir + 2) & 7][0];
    int32_t s1o2 = cdef_directions[(dir + 2) & 7][1];
    int32_t s2o1 = cdef_directions[(dir + 6) & 7][0];
    int32_t s2o2 = cdef_directions[(dir + 6) & 7][1];
    //SSE CHKN
    const int32_t *pri_taps = cdef_pri_taps[(pri_strength >> coeff_shift) & 1];
    const int32_t *sec_taps = cdef_sec_taps[(pri_strength >> coeff_shift) & 1];

    v256 pri_taps_0 = v256_dup_16(pri_taps[0]);
    v256 pri_taps_1 = v256_dup_16(pri_taps[1]);
    v256 sec_taps_0 = v256_dup_16(sec_taps[0]);
    v256 sec_taps_1 = v256_dup_16(sec_taps[1]);
    v256 duplicate_8 = v256_dup_16(8);
    v256 pri_strength_256 = v256_dup_16(pri_strength);
    v256 sec_strength_256 = v256_dup_16(sec_strength);

    if (pri_strength)
        pri_damping = AOMMAX(0, pri_damping - get_msb(pri_strength));
    if (sec_strength)
        sec_damping = AOMMAX(0, sec_damping - get_msb(sec_strength));

    for (i = 0; i < 8; i += 2) {
        sum = v256_zero();
        row = _mm256_insertf128_si256(_mm256_castsi128_si256(
            _mm_loadu_si128((__m128i *)(in + (i + 1) * CDEF_BSTRIDE))),
            _mm_loadu_si128((__m128i *)(in + i * CDEF_BSTRIDE)), 0x1);

        min = max = row;
        // Primary near taps
        p0 = _mm256_insertf128_si256(_mm256_castsi128_si256(
            _mm_loadu_si128((__m128i *)(in + (i + 1) * CDEF_BSTRIDE + po1))),
            _mm_loadu_si128((__m128i *)(in + i * CDEF_BSTRIDE + po1)), 0x1);
        p1 = _mm256_insertf128_si256(_mm256_castsi128_si256(
            _mm_loadu_si128((__m128i *)(in + (i + 1) * CDEF_BSTRIDE - po1))),
            _mm_loadu_si128((__m128i *)(in + i * CDEF_BSTRIDE - po1)), 0x1);
        max =
            v256_max_s16(v256_max_s16(max, v256_andn(p0, v256_cmpeq_16(p0, large))),
                v256_andn(p1, v256_cmpeq_16(p1, large)));
        min = v256_min_s16(v256_min_s16(min, p0), p1);
        p0 = constrain16(p0, row, pri_strength_256, pri_damping);
        p1 = constrain16(p1, row, pri_strength_256, pri_damping);

        // sum += pri_taps[0] * (p0 + p1)
        sum = v256_add_16(
            sum, v256_mullo_s16(pri_taps_0, v256_add_16(p0, p1)));

        // Primary far taps
        p0 = _mm256_insertf128_si256(_mm256_castsi128_si256(
            _mm_loadu_si128((__m128i *)(in + (i + 1) * CDEF_BSTRIDE + po2))),
            _mm_loadu_si128((__m128i *)(in + i * CDEF_BSTRIDE + po2)), 0x1);
        p1 = _mm256_insertf128_si256(_mm256_castsi128_si256(
            _mm_loadu_si128((__m128i *)(in + (i + 1) * CDEF_BSTRIDE - po2))),
            _mm_loadu_si128((__m128i *)(in + i * CDEF_BSTRIDE - po2)), 0x1);
        max =
            v256_max_s16(v256_max_s16(max, v256_andn(p0, v256_cmpeq_16(p0, large))),
                v256_andn(p1, v256_cmpeq_16(p1, large)));
        min = v256_min_s16(v256_min_s16(min, p0), p1);
        p0 = constrain16(p0, row, pri_strength_256, pri_damping);
        p1 = constrain16(p1, row, pri_strength_256, pri_damping);

        // sum += pri_taps[1] * (p0 + p1)
        sum = v256_add_16(
            sum, v256_mullo_s16(pri_taps_1, v256_add_16(p0, p1)));

        // Secondary near taps
        p0 = _mm256_insertf128_si256(_mm256_castsi128_si256(
            _mm_loadu_si128((__m128i *)(in + (i + 1) * CDEF_BSTRIDE + s1o1))),
            _mm_loadu_si128((__m128i *)(in + i * CDEF_BSTRIDE + s1o1)), 0x1);
        p1 = _mm256_insertf128_si256(_mm256_castsi128_si256(
            _mm_loadu_si128((__m128i *)(in + (i + 1) * CDEF_BSTRIDE - s1o1))),
            _mm_loadu_si128((__m128i *)(in + i * CDEF_BSTRIDE - s1o1)), 0x1);
        p2 = _mm256_insertf128_si256(_mm256_castsi128_si256(
            _mm_loadu_si128((__m128i *)(in + (i + 1) * CDEF_BSTRIDE + s2o1))),
            _mm_loadu_si128((__m128i *)(in + i * CDEF_BSTRIDE + s2o1)), 0x1);
        p3 = _mm256_insertf128_si256(_mm256_castsi128_si256(
            _mm_loadu_si128((__m128i *)(in + (i + 1) * CDEF_BSTRIDE - s2o1))),
            _mm_loadu_si128((__m128i *)(in + i * CDEF_BSTRIDE - s2o1)), 0x1);
        max =
            v256_max_s16(v256_max_s16(max, v256_andn(p0, v256_cmpeq_16(p0, large))),
                v256_andn(p1, v256_cmpeq_16(p1, large)));
        max =
            v256_max_s16(v256_max_s16(max, v256_andn(p2, v256_cmpeq_16(p2, large))),
                v256_andn(p3, v256_cmpeq_16(p3, large)));
        min = v256_min_s16(
            v256_min_s16(v256_min_s16(v256_min_s16(min, p0), p1), p2), p3);
        p0 = constrain16(p0, row, sec_strength_256, sec_damping);
        p1 = constrain16(p1, row, sec_strength_256, sec_damping);
        p2 = constrain16(p2, row, sec_strength_256, sec_damping);
        p3 = constrain16(p3, row, sec_strength_256, sec_damping);

        // sum += sec_taps[0] * (p0 + p1 + p2 + p3)
        sum = v256_add_16(sum, v256_mullo_s16(sec_taps_0,
            v256_add_16(v256_add_16(p0, p1),
                v256_add_16(p2, p3))));

        // Secondary far taps
        p0 = _mm256_insertf128_si256(_mm256_castsi128_si256(
            _mm_loadu_si128((__m128i *)(in + (i + 1) * CDEF_BSTRIDE + s1o2))),
            _mm_loadu_si128((__m128i *)(in + i * CDEF_BSTRIDE + s1o2)), 0x1);
        p1 = _mm256_insertf128_si256(_mm256_castsi128_si256(
            _mm_loadu_si128((__m128i *)(in + (i + 1) * CDEF_BSTRIDE - s1o2))),
            _mm_loadu_si128((__m128i *)(in + i * CDEF_BSTRIDE - s1o2)), 0x1);
        p2 = _mm256_insertf128_si256(_mm256_castsi128_si256(
            _mm_loadu_si128((__m128i *)(in + (i + 1)* CDEF_BSTRIDE + s2o2))),
            _mm_loadu_si128((__m128i *)(in + i * CDEF_BSTRIDE + s2o2)), 0x1);
        p3 = _mm256_insertf128_si256(_mm256_castsi128_si256(
            _mm_loadu_si128((__m128i *)(in + (i + 1) * CDEF_BSTRIDE - s2o2))),
            _mm_loadu_si128((__m128i *)(in + i * CDEF_BSTRIDE - s2o2)), 0x1);
        max =
            v256_max_s16(v256_max_s16(max, v256_andn(p0, v256_cmpeq_16(p0, large))),
                v256_andn(p1, v256_cmpeq_16(p1, large)));
        max =
            v256_max_s16(v256_max_s16(max, v256_andn(p2, v256_cmpeq_16(p2, large))),
                v256_andn(p3, v256_cmpeq_16(p3, large)));
        min = v256_min_s16(
            v256_min_s16(v256_min_s16(v256_min_s16(min, p0), p1), p2), p3);
        p0 = constrain16(p0, row, sec_strength_256, sec_damping);
        p1 = constrain16(p1, row, sec_strength_256, sec_damping);
        p2 = constrain16(p2, row, sec_strength_256, sec_damping);
        p3 = constrain16(p3, row, sec_strength_256, sec_damping);

        // sum += sec_taps[1] * (p0 + p1 + p2 + p3)
        sum = v256_add_16(sum, v256_mullo_s16(sec_taps_1,
            v256_add_16(v256_add_16(p0, p1),
            v256_add_16(p2, p3))));

        // res = row + ((sum - (sum < 0) + 8) >> 4)
        sum = v256_add_16(sum, v256_cmplt_s16(sum, v256_zero()));
        res = v256_add_16(sum, duplicate_8);
        res = v256_shr_n_s16(res, 4);
        res = v256_add_16(row, res);
        res = v256_min_s16(v256_max_s16(res, min), max);
        v128_store_unaligned(&dst[i * dstride], v256_high_v128(res));
        v128_store_unaligned(&dst[(i + 1) * dstride], _mm256_castsi256_si128(res));
    }
}

void SIMD_FUNC(cdef_filter_block)(uint8_t *dst8, uint16_t *dst16, int32_t dstride,
    const uint16_t *in, int32_t pri_strength,
    int32_t sec_strength, int32_t dir, int32_t pri_damping,
    int32_t sec_damping, int32_t bsize, int32_t max,
    int32_t coeff_shift) {
    if (dst8) {
        if (bsize == BLOCK_8X8) {
            SIMD_FUNC(cdef_filter_block_8x8_8)
                (dst8, dstride, in, pri_strength, sec_strength, dir, pri_damping,
                    sec_damping, max, coeff_shift);
        }
        else if (bsize == BLOCK_4X8) {
            SIMD_FUNC(cdef_filter_block_4x4_8)
                (dst8, dstride, in, pri_strength, sec_strength, dir, pri_damping,
                    sec_damping, max, coeff_shift);
            SIMD_FUNC(cdef_filter_block_4x4_8)
                (dst8 + 4 * dstride, dstride, in + 4 * CDEF_BSTRIDE, pri_strength,
                    sec_strength, dir, pri_damping, sec_damping, max, coeff_shift);
        }
        else if (bsize == BLOCK_8X4) {
            SIMD_FUNC(cdef_filter_block_4x4_8)
                (dst8, dstride, in, pri_strength, sec_strength, dir, pri_damping,
                    sec_damping, max, coeff_shift);
            SIMD_FUNC(cdef_filter_block_4x4_8)
                (dst8 + 4, dstride, in + 4, pri_strength, sec_strength, dir, pri_damping,
                    sec_damping, max, coeff_shift);
        }
        else {
            SIMD_FUNC(cdef_filter_block_4x4_8)
                (dst8, dstride, in, pri_strength, sec_strength, dir, pri_damping,
                    sec_damping, max, coeff_shift);
        }
    }
    else {
        if (bsize == BLOCK_8X8) {
            SIMD_FUNC(cdef_filter_block_8x8_16)
                (dst16, dstride, in, pri_strength, sec_strength, dir, pri_damping,
                    sec_damping, max, coeff_shift);
        }
        else if (bsize == BLOCK_4X8) {
            SIMD_FUNC(cdef_filter_block_4x4_16)
                (dst16, dstride, in, pri_strength, sec_strength, dir, pri_damping,
                    sec_damping, max, coeff_shift);
            SIMD_FUNC(cdef_filter_block_4x4_16)
                (dst16 + 4 * dstride, dstride, in + 4 * CDEF_BSTRIDE, pri_strength,
                    sec_strength, dir, pri_damping, sec_damping, max, coeff_shift);
        }
        else if (bsize == BLOCK_8X4) {
            SIMD_FUNC(cdef_filter_block_4x4_16)
                (dst16, dstride, in, pri_strength, sec_strength, dir, pri_damping,
                    sec_damping, max, coeff_shift);
            SIMD_FUNC(cdef_filter_block_4x4_16)
                (dst16 + 4, dstride, in + 4, pri_strength, sec_strength, dir, pri_damping,
                    sec_damping, max, coeff_shift);
        }
        else {
            assert(bsize == BLOCK_4X4);
            SIMD_FUNC(cdef_filter_block_4x4_16)
                (dst16, dstride, in, pri_strength, sec_strength, dir, pri_damping,
                    sec_damping, max, coeff_shift);
        }
    }
}

void SIMD_FUNC(copy_rect8_8bit_to_16bit)(uint16_t *dst, int32_t dstride,
    const uint8_t *src, int32_t sstride, int32_t v,
    int32_t h) {
    int32_t i, j;
    for (i = 0; i < v; i++) {
        for (j = 0; j < (h & ~0x7); j += 8) {
            v64 row = v64_load_unaligned(&src[i * sstride + j]);
            v128_store_unaligned(&dst[i * dstride + j], v128_unpack_u8_s16(row));
        }
        for (; j < h; j++) {
            dst[i * dstride + j] = src[i * sstride + j];
        }
    }
}

void SIMD_FUNC(copy_rect8_16bit_to_16bit)(uint16_t *dst, int32_t dstride,
    const uint16_t *src, int32_t sstride,
    int32_t v, int32_t h) {
    int32_t i, j;
    for (i = 0; i < v; i++) {
        for (j = 0; j < (h & ~0x7); j += 8) {
            v128 row = v128_load_unaligned(&src[i * sstride + j]);
            v128_store_unaligned(&dst[i * dstride + j], row);
        }
        for (; j < h; j++) {
            dst[i * dstride + j] = src[i * sstride + j];
        }
    }
}

