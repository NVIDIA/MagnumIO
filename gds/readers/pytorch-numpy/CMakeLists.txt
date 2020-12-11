# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 3.5)

#project definition
project(tdlg LANGUAGES CXX)

#include the external project stuff
include(ExternalProject)

#global settings
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake ${CMAKE_MODULE_PATH})
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -Wall -Werror=reorder")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

#determine type of build
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
endif()

#now, add the right flags
if (CMAKE_BUILD_TYPE STREQUAL "Debug" )
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_DEBUG}")
else()
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELEASE}")
endif()

#############################################
# Build dependencies: depends on the method
#############################################
add_library(numpy_reader_src-object OBJECT
               cpp/numpy_reader.cpp
            )
			
#target_compile_definitions( numpy_reader_src-object PUBLIC "-DPREC_${PRECISION}" "${OMP_DEFINES}" "${METHOD_DEFINES}" "${CUDA_DEFINES}")
target_include_directories( numpy_reader_src-object PUBLIC "${PYTORCH_DIR}" "${CMAKE_SOURCE_DIR}/cpp" )
add_library(numpy_reader_src-static STATIC $<TARGET_OBJECTS:numpy_reader_src-object>)
set_target_properties(numpy_reader_src-static PROPERTIES OUTPUT_NAME numpy_reader_src)

#target
add_executable( numpy_reader_run src/main.cpp )
target_link_libraries( numpy_reader_run numpy_reader_src-static )
#target_compile_definitions( numpy_reader_run PUBLIC "-DPREC_${PRECISION}" "${OMP_DEFINES}" "${METHOD_DEFINES}" "${CUDA_DEFINES}")
target_include_directories( numpy_reader_run PUBLIC "${CMAKE_SOURCE_DIR}/cpp" )