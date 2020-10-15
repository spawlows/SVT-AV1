/*
* Copyright(c) 2020 Intel Corporation
*
* This source code is subject to the terms of the BSD 2 Clause License and
* the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
* was not distributed with this source code in the LICENSE file, you can
* obtain it at https://www.aomedia.org/license/software-license. If the Alliance for Open
* Media Patent License 1.0 was not distributed with this source code in the
* PATENTS file, you can obtain it at https://www.aomedia.org/license/patent-license.
*/


/*
* This file contains only debug macros that are used during the development
* and are supposed to be cleaned up every tag cycle
* all macros must have the following format:
* - enabling a feature should be prefixed by ENABLE_
* - disableing a feature should be prefixed by DISABLE_
* - tuning a feature should be prefixed by TUNE_
* - adding a new feature should be prefixed by FEATURE_
* - bug fixes should be prefixed by FIX_
* - all macros must have a coherent comment explaining what the MACRO is doing
* - #if 0 / #if 1 are not to be used
*/


#ifndef EbDebugMacros_h
#define EbDebugMacros_h

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

// undefining this macro would allow the AVX512 optimization to be enabled by default
#ifndef NON_AVX512_SUPPORT
#define NON_AVX512_SUPPORT
#endif

#define FIX_RC_BUG 1 // Fix the one pass QP assignment using frames_to_be_encoded
#define FIX_VBR_BUG 1 // Fix 1st pass bug (bug from rebasing the branch)
#define FIX_10BIT     1 // fix 1st pass for 10bit input
#define FIX_RC_TOKEN     1 // fix RC token check to include double dash

#define FIX_PAD_CHROMA_AFTER_MCTF     1 // Padding chroma after altref
#define FEATURE_NEW_DELAY             1 // Change delay some sorts of I in PicDecision
#define FEATURE_INL_ME                1 //Enable in-loop ME
#if FEATURE_INL_ME
#define TUNE_IME_REUSE_TPL_RESULT     1 // Reuse TPL results for iLoopME
#define TUNE_INL_TPL_ENHANCEMENT      1 // Refinement for TPL
#define TUNE_INL_ME_RECON_INPUT       1 // Perform ME GM, TPL on input/recon: 1 on input, 0 on recon
#if !TUNE_IME_REUSE_TPL_RESULT
#define TUNE_SIGNAL_TPL_ME_OQ         1 // A separate signal_xxx_oq for TPL ME
#endif
#endif

#define FEATURE_IN_LOOP_TPL 1 // Moving TPL to in loop
#if FEATURE_IN_LOOP_TPL
#define ENABLE_TPL_ZERO_LAD     1 // Enable TPL in loop to work with zero LAD
#define TUNE_TPL                1   // Tuning TPL algorithm for QP assignment
#define ENABLE_TPL_TRAILING     1 //enable trailing pictures for TPL
#define FIX_GM_BUG              1 // FIX GM r2r difference
#endif



//FOR DEBUGGING - Do not remove
#define NO_ENCDEC         0 // bypass encDec to test cmpliance of MD. complained achieved when skip_flag is OFF. Port sample code from VCI-SW_AV1_Candidate1 branch

#ifdef __cplusplus
}
#endif // __cplusplus

#endif // EbDebugMacros_h
