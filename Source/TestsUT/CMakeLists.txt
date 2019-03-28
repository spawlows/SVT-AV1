# INTEL CONFIDENTIAL
# Copyright © 2018 Intel Corporation.
#
# This software and the related documents are Intel copyrighted materials, 
# and your use of them is governed by the express license under which they were provided to you.
# Unless the License provides otherwise, you may not use, modify, copy, publish, distribute, disclose or transmit 
# this software or the related documents without Intel's prior written permission.
# This software and the related documents are provided as is, with no express or implied warranties, 
# other than those that are expressly stated in the License.

#add_subdirectory (C:/WORK/AV1/SVT-AV1-github/Source/Lib)

# APP Directory CMakeLists.txt
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Bin/${CMAKE_BUILD_TYPE}/)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Bin/${CMAKE_BUILD_TYPE}/)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Bin/${CMAKE_BUILD_TYPE}/)

if(UNIX)
    # Intel Linux
    if("${CMAKE_C_COMPILER_ID}" STREQUAL "Intel")
        SET(CMAKE_C_FLAGS "-fPIC -static-intel -w")
    else()
        SET(CMAKE_C_FLAGS "-march=core-avx2")
    endif()
else()
    # Intel Windows (*Note - The Warning level /W0 should be made to /W4 at some point)
    if("${CMAKE_C_COMPILER_ID}" STREQUAL "Intel")
        SET(CMAKE_C_FLAGS "/W0 /Qwd10148 /Qwd10010 /Qwd10157")
    else()
        SET(CMAKE_C_FLAGS "/arch:AVX2 /MP")    
    endif()
endif()




# Include Subdirectories
#include_directories (${PROJECT_SOURCE_DIR}/Source/API/)
#include_directories (${PROJECT_SOURCE_DIR}/Source/API/OpenMAX/IL/)
#include_directories (${PROJECT_SOURCE_DIR}/Bin/${CMAKE_BUILD_TYPE}/)

#UT
#include_directories (${PROJECT_SOURCE_DIR}/Source/API/)
#include_directories (${PROJECT_SOURCE_DIR}/Source/API/OpenMAX/IL/)
#include_directories (${PROJECT_SOURCE_DIR}/Source/Lib/SRM/)
#include_directories (${PROJECT_SOURCE_DIR}/Source/Lib/System/)
#include_directories (${PROJECT_SOURCE_DIR}/Source/Lib/Codec/)
#include_directories (${PROJECT_SOURCE_DIR}/Source/Lib/Utility/)

##include_directories (${PROJECT_SOURCE_DIR}/Source/Lib/C_DEFAULT/)
#include_directories (${PROJECT_SOURCE_DIR}/Source/Lib/ASM_SSE2/)
#include_directories (${PROJECT_SOURCE_DIR}/Source/Lib/ASM_SSSE3/)
#include_directories (${PROJECT_SOURCE_DIR}/Source/Lib/ASM_SSE4_1/)
#include_directories (${PROJECT_SOURCE_DIR}/Source/Lib/ASM_AVX2/)

#link_directories (${PROJECT_SOURCE_DIR}/Source/Lib/ASM_SSE2/)
#link_directories (${PROJECT_SOURCE_DIR}/Source/Lib/C_DEFAULT/)
#link_directories (${PROJECT_SOURCE_DIR}/Source/Lib/ASM_SSSE3/)
#link_directories (${PROJECT_SOURCE_DIR}/Source/Lib/ASM_SSE4_1/)

#link_directories (${PROJECT_SOURCE_DIR}/Source/Lib/ASM_SSE2/)
#link_directories (${PROJECT_SOURCE_DIR}/Source/Lib/C_DEFAULT/)
#link_directories (${PROJECT_SOURCE_DIR}/Source/Lib/ASM_SSSE3/)
#link_directories (${PROJECT_SOURCE_DIR}/Source/Lib/ASM_SSE4_1/)
#link_directories (${PROJECT_SOURCE_DIR}/Source/Lib/ASM_AVX2/)

link_directories (${PROJECT_SOURCE_DIR}/Bin/${CMAKE_BUILD_TYPE}/)


file(GLOB SOURCES_FILES
    "*.c"
)

#file(GLOB SOURCES_CODEC
#    "../Lib/Codec/*.c"
#)

# App Source Files
add_executable (SvtAv1Tests 
    ${SOURCES_FILES}
    ${SOURCES_CODEC}
)

if(UNIX)
target_link_libraries(SvtAv1Tests LINK_PUBLIC
	m
	pthread
)
endif(UNIX)


#target_link_libraries(SvtAv1Tests LINK_PUBLIC
##    SvtAv1EncLib#
#	C_DEFAULT
#	ASM_SSE2
#	ASM_SSSE3
#	ASM_SSE4_1
#	ASM_AVX2
 #  )

   #run -tests -perf -pause
   