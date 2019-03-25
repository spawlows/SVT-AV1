#ifndef _TEST_CODE_H_
#define _TEST_CODE_H_
 
enum TEST_STAGE{
    STAGE_GET_ID_MAX = 0,
    STAGE_CREATE,
    STAGE_RAND_VALUES,
    STAGE_EXECUTE_A,
    STAGE_EXECUTE_B,
    STAGE_CHECK,
    STAGE_DESTROY
};

typedef int(*fnTestCase)(void** context, enum TEST_STAGE stage, int test_id, int verbose);

int TestCase_mse_4x4_16bit(void** context, enum TEST_STAGE stage, int test_id, int verbose);
int TestCase_get_proj_subspace(void** context, enum TEST_STAGE stage, int test_id, int verbose);
int TestCaseTranspose_8nx8n(void** context, enum TEST_STAGE stage, int test_id, int verbose);
int TestCaseCompute4xMSadSub_SSE2_INTRIN(void** context, enum TEST_STAGE stage, int test_id, int verbose);
int TestCase_fwd_txfm2d_16x4(void** context, enum TEST_STAGE stage, int test_id, int verbose);
int TestCase_fwd_txfm2d_4x16(void** context, enum TEST_STAGE stage, int test_id, int verbose);
int TestCase_fwd_txfm2d_4x8(void** context, enum TEST_STAGE stage, int test_id, int verbose);
int TestCase_fwd_txfm2d_8x4(void** context, enum TEST_STAGE stage, int test_id, int verbose);
int TestCase_sad_calculation_32x32_64x64(void** context, enum TEST_STAGE stage, int test_id, int verbose);
int TestCase_sad_calculation_nsq(void** context, enum TEST_STAGE stage, int test_id, int verbose);
int TestCase_sad_calculation_8x8_16x16(void** context, enum TEST_STAGE stage, int test_id, int verbose);
int TestCase_highbd_paeth_predictor(void** context, enum TEST_STAGE stage, int test_id, int verbose);
int TestCase_fill_rect(void** context, enum TEST_STAGE stage, int test_id, int verbose);

int TestCase_fwd_txfm2d_64x64(void** context, enum TEST_STAGE stage, int test_id, int verbose);

int TestCaseIIdentity16(void** context, enum TEST_STAGE stage, int test_id, int verbose);
int TestCaseInv4x4(void** context, enum TEST_STAGE stage, int test_id, int verbose);
int TestCaseInv8x8(void** context, enum TEST_STAGE stage, int test_id, int verbose);
int TestCaseInv16x16(void** context, enum TEST_STAGE stage, int test_id, int verbose);
int TestCaseInv32x32(void** context, enum TEST_STAGE stage, int test_id, int verbose);
int TestCaseIIdentity32(void** context, enum TEST_STAGE stage, int test_id, int verbose);


#endif /*_TEST_CODE_H_*/