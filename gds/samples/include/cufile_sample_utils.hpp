/*
 * Copyright 2020-2025 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#ifndef __CUFILE_SAMPLE_UTILS_HPP_
#define __CUFILE_SAMPLE_UTILS_HPP_

#include <cassert>
#include <cstring>
#include <random>
#include <chrono>
#include <type_traits>
#include <climits>
#include <cstdlib>
#include <cerrno>

#include <cuda.h>
#include <cuda_runtime.h>

#define check_cudadrivercall(fn) \
	do { \
		CUresult res = (fn); \
		if (res != CUDA_SUCCESS) { \
			const char *name = "unknown"; \
			const char *desc = "no description"; \
			cuGetErrorName(res, &name); \
			cuGetErrorString(res, &desc); \
			std::cerr << "CUDA Driver API error at " << __FILE__ << ":" << __LINE__ << std::endl \
					  << "  Expression: " << #fn << std::endl \
					  << "  Error Name: " << name << std::endl \
					  << "  Error Desc: " << desc << std::endl \
					  << "Exiting." << std::endl; \
			std::exit(EXIT_FAILURE); \
		} \
	} while (0)

#define check_cudaruntimecall_thread(fn, promise) \
	do { \
		cudaError_t res = (fn); \
		if (res != cudaSuccess) { \
			std::cerr << "CUDA Runtime API error at " << __FILE__ << ":" << __LINE__ << std::endl \
					  << "  Expression: " << #fn << std::endl \
					  << "  Error Name: " << cudaGetErrorName(res) << std::endl \
					  << "  Error Desc: " << cudaGetErrorString(res) << std::endl \
					  << "Exiting." << std::endl; \
			promise.set_value(-1); \
			return; \
		} \
	} while (0)

#define check_cudaruntimecall(fn) \
	do { \
		cudaError_t res = (fn); \
		if (res != cudaSuccess) { \
			std::cerr << "CUDA Runtime API error at " << __FILE__ << ":" << __LINE__ << std::endl \
					  << "  Expression: " << #fn << std::endl \
					  << "  Error Name: " << cudaGetErrorName(res) << std::endl \
					  << "  Error Desc: " << cudaGetErrorString(res) << std::endl \
					  << "Exiting." << std::endl; \
			std::exit(EXIT_FAILURE); \
		} \
	} while (0)

struct Prng {
	long rmax_;
	std::mt19937 rand_;
	std::uniform_int_distribution<long> dist_;
	Prng(long rmax) :
		rmax_(rmax),
		rand_(std::chrono::high_resolution_clock::now().time_since_epoch().count()),
		dist_(std::uniform_int_distribution<long>(0, rmax_))
		{}
	long next_random_offset(void) {
		return dist_(rand_);
	}
};

template<typename T>
inline constexpr auto KiB(T x) -> typename std::enable_if<std::is_integral<T>::value, T>::type {
	return x * 1024LL;
}

template<typename T>
inline constexpr auto TOKiB(T x) -> typename std::enable_if<std::is_integral<T>::value, T>::type {
	return x / (1024LL);
}

template<typename T>
inline constexpr auto MiB(T x) -> typename std::enable_if<std::is_integral<T>::value, T>::type {
	return x * 1024LL * 1024LL;
}

template<typename T>
inline constexpr auto TOMiB(T x) -> typename std::enable_if<std::is_integral<T>::value, T>::type {
	return x / (1024LL * 1024LL);
}

template<typename T>
inline constexpr auto GiB(T x) -> typename std::enable_if<std::is_integral<T>::value, T>::type {
	return x * 1024LL * 1024LL * 1024LL;
}

template<typename T>
inline constexpr auto TOGiB(T x) -> typename std::enable_if<std::is_integral<T>::value, T>::type {
	return x / (1024LL * 1024LL * 1024LL);
}

//
// CUDA driver error description
//
inline const char *GetCuErrorString(CUresult curesult) {
	const char *descp = nullptr;
	if (cuGetErrorName(curesult, &descp) != CUDA_SUCCESS)
		descp = "unknown cuda error";
	return descp;
}

//
// CUDA runtime error description
//
inline const char *GetCudaErrorString(cudaError_t cudaerr) {
	return cudaGetErrorName(cudaerr);
}


//
// cuFile APIs return both cuFile specific error codes as well as POSIX error codes
// for ease, the below template can be used for getting the error description depending
// on its type.

// POSIX
template<class T, typename std::enable_if<std::is_integral<T>::value, std::nullptr_t>::type = nullptr>
inline std::string cuFileGetErrorString(T status) {
	status = std::abs(status);
	return IS_CUFILE_ERR(status) ?
		std::string(CUFILE_ERRSTR(status)) : std::string(std::strerror(status));
}

// CUfileError_t
template<class T, typename std::enable_if<!std::is_integral<T>::value, std::nullptr_t>::type = nullptr>
inline std::string cuFileGetErrorString(T status) {
	std::string errStr = cuFileGetErrorString(static_cast<int>(status.err));
	if (IS_CUDA_ERR(status))
		errStr.append(".").append(GetCuErrorString(status.cu_err));
	return errStr;
}

// Round up to the nearest multiple of y
inline size_t round_up(size_t x, size_t y) {
	return ((x + y - 1) / y) * y;
}

// Generate a random float between 0 and 1
inline float randFloat() {
	return static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
}

////////////////////////////////////////////////////////////////////////////
//! Get the size of a file
//! @param[in] fd  File descriptor
//! @return File size
////////////////////////////////////////////////////////////////////////////
size_t GetFileSize(int fd);

////////////////////////////////////////////////////////////////////////////
//! Parse an integer from a string
//! @param[in] str  String to parse
//! @return Parsed integer
////////////////////////////////////////////////////////////////////////////
int parseInt(const char* str);

////////////////////////////////////////////////////////////////////////////
//! Create a test file with the given size
//! @param[in] filename  Name of the file to create
//! @param[in] size  Size of the file to create
//! @return True if the file is created successfully, exit otherwise
////////////////////////////////////////////////////////////////////////////
bool createTestFile(const std::string& filename, size_t size);

#endif
