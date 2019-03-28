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
#include <assert.h>
#include "TestCore.h"

#if defined (_MSC_VER)
typedef LARGE_INTEGER Timer;
static Timer Frequency;
#define GetTime(aa) QueryPerformanceCounter(aa);
#else

#include <stdio.h> // for printf()
#include <sys/time.h> // for clock_gettime()
#include <unistd.h> // for usleep()


typedef struct timeval Timer;
#define GetTime(aa)  gettimeofday(aa, NULL);
#endif

fnTestCase cases[] = {
    TestCase_fill_rect
    //TestCase_highbd_paeth_predictor
    //TestCase_sad_calculation_8x8_16x16,
    //TestCase_sad_calculation_32x32_64x64,
    //TestCase_sad_calculation_nsq
    //TestCase_sad_calculation_nsq
    //TestCaseCompute4xMSadSub_SSE2_INTRIN,
    //TestCaseTranspose_8nx8n,
    //TestCase_fwd_txfm2d_8x4,
    //TestCase_fwd_txfm2d_4x8,
    //TestCase_fwd_txfm2d_16x4,
    //TestCase_fwd_txfm2d_64x64
//    TestCase_get_proj_subspace
    //TestCase_get_proj_subspace
	//TestCaseInv4x4,
	//TestCaseInv8x8,
												////TestCaseIIdentity16,
	//TestCaseInv16x16,
	//TestCaseInv32x32,
										////TestCaseIIdentity32
};



float GetTimerStopMS(Timer start)
{
#if defined (_MSC_VER)
    Timer timer;
    QueryPerformanceCounter(&timer);

    timer.QuadPart = timer.QuadPart - start.QuadPart;
    timer.QuadPart *= 1000;
    //timer.QuadPart /= Frequency.QuadPart;

    return (float)timer.QuadPart / (float)Frequency.QuadPart;
#else
  struct timeval end;
  gettimeofday(&end, NULL);
  long secs_used = (end.tv_sec - start.tv_sec); //avoid overflow by subtracting first
  return ((float)(((secs_used * 1000000) + end.tv_usec) - (start.tv_usec)))/1000.0f;
#endif
}


int RunPerformaceBody(fnTestCase fn, void *cnt, int id)
{
    int ret = -1;

	//Adaptacja testu do czasu wykonywania
    int repeatExecution = 1000000;
	float testTime = 4000.0f;

	////Bez ograniczenia czasu:
	//repeatExecution = 10000000;
	//testTime = 0.1f;
	////repeatExecution = 10000000;

   // repeatExecution = 2;

    ret = fn(&cnt, STAGE_RAND_VALUES, id, 0);
    if (ret) {
        printf("ERROR: %s:%i ret: %i\n", __FUNCTION__, __LINE__, ret);
        return ret;
    }
    float aa = 0.0f;
    int RepetGlobal = 0;
    Timer ts;
    GetTime(&ts);
	

	//while (aa < testTime) {
		for (int j = 0; j < repeatExecution; ++j) {
			fn(&cnt, STAGE_EXECUTE_A, id, 0);
		}
		aa = GetTimerStopMS(ts);
		//RepetGlobal++;
	//}
    printf("time calc: %f\n", aa);
    float bb = 0.0f;
  GetTime(&ts);
	
	//for (int w=0;w< RepetGlobal;++w) {
		for (int j = 0; j < repeatExecution; ++j) {
			fn(&cnt, STAGE_EXECUTE_B, id, 0);
		}
		bb = GetTimerStopMS(ts);
//	}

    //printf("time calc: %f\n", bb);
	float cc = -1;
    if (bb > 0.0001f) {
		cc = (aa) / (bb);
        //printf("Function2 run faster: %.2f\n", aa / bb);
    }
	printf("Time: a %6.2f:%i b %6.2f      run faster: %5.2f\n", aa, RepetGlobal*repeatExecution, bb, cc);

    ret = fn(&cnt, STAGE_CHECK, id, 1); //verbose
    return ret;
}
 


int RunTestCase(fnTestCase fn, int runUT, int runPerf)
{
    int id_max = fn(NULL, STAGE_GET_ID_MAX, 0, 0);
    int ret = -1;
    for (int i = 0; i < id_max; ++i) {
        void* cnt = NULL;
        ret = fn(&cnt, STAGE_CREATE, i, runUT);
        if (ret)
            return ret;

        if (runUT) {
            //TEST UT
            for (int j = 0; j < 500; ++j) {
                ret = fn(&cnt, STAGE_RAND_VALUES, i, runUT);
                if (ret) {
                    printf("ERROR: %s:%i ret: %i\n", __FUNCTION__, __LINE__, ret);
                    return ret;
                }



                //TODO: Run 2 razy bo output czasami siê zmienia!!!
				for (int g=0; g< j%2 +1; ++g) {
					ret = fn(&cnt, STAGE_EXECUTE_A, i, runUT);
					if (ret) {
						printf("ERROR: %s:%i ret: %i\n", __FUNCTION__, __LINE__, ret);
						return ret;
					}
				}

				for (int g = 0; g < j % 2 + 1; ++g) {
					ret = fn(&cnt, STAGE_EXECUTE_B, i, runUT);
					if (ret) {
						printf("ERROR: %s:%i ret: %i\n", __FUNCTION__, __LINE__, ret);
						return ret;
					}
				}


                ret = fn(&cnt, STAGE_CHECK, i, runUT);
                if (ret) {
                    printf("ERROR: %s:%i ret: %i\n", __FUNCTION__, __LINE__, ret);
                    return ret;
                }
            }
        }
        else {
            assert(runPerf);
            //TEST PERFORMANCE
            ret = RunPerformaceBody(fn, cnt, i);
            if (ret) {
                printf("ERROR: %s:%i ret: %i\n", __FUNCTION__, __LINE__, ret);
                return ret;
            }
#if defined (_MSC_VER)
			Sleep(1700); //To cool down CPU ;)
#endif
        }

        ret = fn(&cnt, STAGE_DESTROY, i, runUT);
        if (ret) {
            printf("ERROR: %s:%i ret: %i\n", __FUNCTION__, __LINE__, ret);
            return ret;
        }
    }

    return ret; 
}


int TestAll(int runUT, int runPerf)
{
    int ret = -1;

    for (int i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
        ret = RunTestCase(cases[i], runUT, runPerf);
        if (ret)
            return ret;
    }

    return ret;
}


/***************************************
 * Encoder App Main
 ***************************************/
int main(int argc, char* argv[])
{

#if defined (_MSC_VER)
    time_t t;
    srand((unsigned)time(&t));
    QueryPerformanceFrequency(&Frequency);
#endif

    int runUT = 0;
    int runPerf = 0;
    int pause = 0;
    int ret = -1;

    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-tests") || !strcmp(argv[1], "-test")) {
            printf("Run UT!\n");
            runUT = 1;
        } else if (!strcmp(argv[i], "-perf")) {
            printf("Run Performance!\n");
            runPerf = 1;
        } if (!strcmp(argv[i], "-pause")) {
            printf("Pause at end!\n");
            pause = 1;
        }
    }

    if (runUT) {
        ret = TestAll(1, 0);
        printf("Test UT return: %i\n", ret);
        if (ret)
            goto end;
    }

    if (runPerf) {
        ret = TestAll(0, 1);
        printf("Test Perf return: %i\n", ret);
        if (ret)
            goto end;
    }

end:
    printf("Test return: %i\n", ret);
    if (pause)
        system("pause");

    return ret;
}