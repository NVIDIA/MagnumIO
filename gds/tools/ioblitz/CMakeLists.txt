cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

# set(CMAKE_BUILD_TYPE Release)
#set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_BUILD_TYPE RelWithDebInfo)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_COLOR_MAKEFILE ON)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

project(ioblitz LANGUAGES C CXX CUDA VERSION 0.3.0)
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/modules ${CMAKE_MODULE_PATH})

find_package(CUDA REQUIRED)
find_package(MPI REQUIRED)
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
include_directories("/usr/local/gds/lib")
include_directories(${MPI_INCLUDE_PATH})

# CUDA flags
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# treat warnings as errors
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wshadow -pedantic -Werror")

# do not treat warnings as errors
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wshadow")

set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

set(CUDA_USE_STATIC_CUDA_RUNTIME ON)
set(CUDA_VERBOSE_BUILD ON)

##############################################################################
# BUILD libioblitz.so

set (LIB_IOBLITZ_SRCS
  src/ioblitz.h src/ioblitz.c
)

include_directories(./src)

SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/lib/")

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

link_directories("/lib64")
link_directories("/usr/lib64")
link_directories("/usr/local/cuda/lib64")
link_directories("/sw/summit/cuda/11.0.3/lib64")

add_library(ioblitz SHARED ${LIB_IOBLITZ_SRCS})
target_link_libraries(ioblitz Threads::Threads)
target_link_libraries(ioblitz dl)
target_link_libraries(ioblitz ${CUDA_LIBRARIES})
target_link_libraries(ioblitz cuda)
target_link_libraries(ioblitz nvToolsExt)
target_link_libraries(ioblitz "-Wl,--no-undefined")

##############################################################################
