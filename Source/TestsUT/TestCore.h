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

int TestCase_fill_rect(void** context, enum TEST_STAGE stage, int test_id, int verbose);


#endif /*_TEST_CODE_H_*/