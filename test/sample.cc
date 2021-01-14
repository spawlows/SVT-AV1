#include <stdlib.h>
#include "gtest/gtest.h"
//#include "random.h"
//#include "EbTime.h"
//#include "EbUtility.h"

namespace {

//
//TEST(Suit1, TestCase1) {
//  //  printf("XXX");
//}
//TEST(Suit1, TestCase2) {
//  //  printf("XXX");
//    //EXPECT_EQ(0,1) << "ABC";
//  //  ASSERT_EQ(0,1);
//       EXPECT_EQ(1,1);
//}
//
//TEST(Suit2, TestCase1) {
//  //  printf("XXX");
//}
//TEST(Suit2, TestCase2) {
//  //  printf("XXX");
//    //EXPECT_EQ(0,1) << "ABC";
//  //  ASSERT_EQ(0,1);
//       EXPECT_EQ(1,1);
//}
//
//
//
//
//#include "gtest/gtest.h"
//
//int factorial(unsigned int a) {
//    int fac = 1;
//    for (int i = 2; i <= a; ++i) fac *= i;
//    return fac;
//}
//
//TEST(FactorialTest, CheckOne) {
//    ASSERT_EQ(1, factorial(0));
//    ASSERT_EQ(1, factorial(1));
//}
//
//TEST(FactorialTest, CheckComplex) {
//    ASSERT_EQ(2, factorial(2));
//    ASSERT_EQ(6, factorial(3));
//    ASSERT_EQ(8*7*6*5*4*3*2, factorial(8));
//}

	
#include "gtest/gtest.h"

//
//TEST(FactorialTest, CheckComplex) {
//    char *buff = NULL;
//    char * buff_ref = NULL;
//    int buff_size = 1;
//    ASSERT_NE(NULL, buff) << "Buffer buff is NULL, expect allocation ";
//    ASSERT_NE(NULL, buff_ref) << "Buffer buff_ref is NULL!";
//    EXPECT_EQ(0, memcmp(buff, buff_ref, buff_size));
//    int COLOR_CHANNELS = 4;
//    int MAX_STRIDE = 5;
//    int **accum_ref_ptr = NULL, **accum_tst_ptr = NULL;
//
//    for (int color_channel = 0; color_channel < COLOR_CHANNELS; color_channel++) {
//        ASSERT_NE(NULL, accum_ref_ptr[color_channel])
//            << "Buffer accum_ref_ptr is NULL for channel " << color_channel;
//        ASSERT_NE(NULL, accum_tst_ptr[color_channel])
//            << "Buffer accum_tst_ptr is NULL for channel " << color_channel;
//        EXPECT_EQ(0, memcmp(accum_ref_ptr[color_channel],
//                         accum_tst_ptr[color_channel],
//                         MAX_STRIDE * MAX_STRIDE * sizeof(uint32_t)))
//            << "Buffers different for channel " << color_channel;;
//    }
//}


///* SAMPLE TEST MACRO */
//TEST(SampleTest, TestCase1) {
//}
//TEST(SampleTest, TestCase2) {
//}
//TEST(SampleTest2, TestCase1) {
//}
//TEST(SampleTest2, TestCase2) {
//}
//ASSERT_NEAR
//

//class ClassFixture : public ::testing::Test {
//  protected:
//    static void SetUpTestCase() {...}    // Prepare data for all tests
//    static void TearDownTestCase() {...} //Release data for all tests
//    virtual void SetUp() {...}           //Prepare data for each test
//    virtual void TearDown(){...}         //Release data for each test
//    void run_test() {...}
//};

/* SAMPLE TEST_F MACRO */
//class ClassFixture : public ::testing::Test {
//  protected:
//    static char* mem;
//
//    static void SetUpTestCase() {
//        printf("Create\n");
//        mem = (char*)malloc(10);
//    }
//    static void TearDownTestCase() {
//        printf("Destroy\n");
//        free(mem);
//    }
//    virtual void SetUp() {
//        printf("SetUp\n");
//        memset(mem, 0, 10);
//    }
//    virtual void TearDown() {
//        printf("SetDown\n");
//    }
//    void run_test() {
//        ASSERT_TRUE(mem[0] > 3 &&  mem[0]  < 8);
//        printf("Test value %i\n",  ((char*)mem)[0]);
//    }
//};
//char* ClassFixture::mem = NULL;
//
//TEST_F(ClassFixture, TestCaseFixture1) {
//    mem[0] = 5;
//    run_test();
//}
//
//TEST_F(ClassFixture, TestCaseFixture2) {
//    mem[0] = 7;
//    run_test();
//}

//TEST_F(test_fixture, test_name) {
//   ... test body ....
//}

//
///* SAMPLE INSTANTIATE_TEST_CASE_P and TEST_P MACRO ONE PARAM */
//
//class GeneratorFixtureOne : public ::testing::TestWithParam<int>  {
//  protected:
//    int a;
//
//    virtual void SetUp() {
//        a = GetParam();
//    }
//
//    void validate_test() {
//        printf("TestValid %i\n", a);
//    }
//
//    void run_test() {
//        printf("TestRun %i\n", a);
//    }
//};
//
//TEST_P(GeneratorFixtureOne, Validate) {
//    validate_test();
//}
//
//TEST_P(GeneratorFixtureOne, TestCaseFixture) {
//    run_test();
//}
//
//INSTANTIATE_TEST_CASE_P(GenTest, GeneratorFixtureOne,
//    ::testing::Range(0,5)); 

//
////::testing::Range(0,6,2),       /*0,2,4*/
////::testing::Range(0,5),         /*0,1,2,3,4*/
////::testing::Values(11, 22, 33), /*11, 22, 33*/
////::testing::ValuesIn(names)     /*"Name1", "Name2", "Name3"*/

//
///* SAMPLE INSTANTIATE_TEST_CASE_P and TEST_P MACRO MULTIPLE PARAMS */
////
//typedef ::testing::tuple<char*, int, int> fixture_param_t; /* minimum 2 params */
//
//class GeneratorFixture : public ::testing::TestWithParam<fixture_param_t>  {
//  protected:
//    char *name;
//    int w;
//    int h;
//    virtual void SetUp() {
//        name = std::get<0>(GetParam());
//        w = std::get<1>(GetParam());
//        h = std::get<2>(GetParam());
//    }
//
//    void run_test() {
//        ASSERT_EQ(0, (w % 2) || (h % 2));
//        printf("Test %s Size: %ix%i \n", name, w, h);
//    }
//};
//
//TEST_P(GeneratorFixture, TestCaseFixture) {
//    run_test();
//}
//
//char* names[] = {"Name1", "Name2", "Name3"};
//INSTANTIATE_TEST_CASE_P(GenTest, GeneratorFixture,
//       ::testing::Combine(
//           ::testing::ValuesIn(names),
//           ::testing::Range(4,18, 4), /*4,8,12,16*/
//           ::testing::Values(4, 8, 16)
//           ));


TEST(TestCase1, TestName1) {
}
TEST(TestCase1, TestName2) {
}
TEST(TestCase2, TestName1) {
}
TEST(TestCase2, TestName2) {
}


////////////////
//


}  // namespace
