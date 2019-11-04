/*
 * Copyright(c) 2019 Netflix, Inc.
 * SPDX - License - Identifier: BSD - 2 - Clause - Patent
 */

/******************************************************************************
 * @file TestEnv.c
 *
 * @brief Environment setup for unit test
 *
 * @author Cidana-Edmond
 *
 ******************************************************************************/

#include "aom_dsp_rtcd.h"

/** setup_test_env is a util for unit test setup environment without create a
 * encoder */
void setup_test_env() {
    CPU_FLAGS cpu_flags = get_cpu_flags();

#ifdef NON_AVX512_SUPPORT
    /* Disable bits equal, or upper that CPU_FLAGS_AVX512F */
    cpu_flags &= (CPU_FLAGS_AVX512F - 1);
#endif

    setup_rtcd_internal(cpu_flags);
}
