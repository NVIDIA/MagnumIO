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

#include <iostream>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <algorithm>

#include "../include/cufile_sample_utils.hpp"

size_t GetFileSize(int fd) {
	struct stat st;
	int ret = fstat(fd, &st);
	return (ret == 0) ? st.st_size : 0;
}

int parseInt(const char* str) {
	char* endptr = nullptr;
	errno = 0;  // Clear errno before call

	long val = strtol(str, &endptr, 10);

	if (errno == ERANGE || val > INT_MAX || val < INT_MIN) {
		std::cerr << "Value out of range: " << std::strerror(errno) << std::endl;
		std::exit(EXIT_FAILURE);
	}

	if (endptr == str || *endptr != '\0') {
		std::cerr << "Invalid numeric input: " << str << std::endl;
		std::exit(EXIT_FAILURE);
	}

	return static_cast<int>(val);
}

bool createTestFile(const std::string& filename, size_t size) {
	std::unique_ptr<char[]> hostPtr = nullptr;
	Prng prng(255);
	ssize_t ret = -1;
	int fd = -1;

	fd = open(filename.c_str(), O_CREAT | O_WRONLY | O_TRUNC, 0644);
	if (fd < 0) {
		std::cerr << "Test file open error: " << filename << " "
			<< std::strerror(errno) << std::endl;
		std::exit(EXIT_FAILURE);
	}

	hostPtr = std::make_unique<char[]>(size);
	if (!hostPtr) {
		std::cerr << "Buffer allocation failure: "
			<< std::strerror(errno) << std::endl;
		std::exit(EXIT_FAILURE);
	}

	std::generate(hostPtr.get(), hostPtr.get() + size, [&]() {
		return static_cast<char>(prng.next_random_offset());
	});

	ret = write(fd, hostPtr.get(), size);
	if (ret < 0) {
		std::cerr << "Write error: " << filename << " "
			<< std::strerror(errno) << std::endl;
		std::exit(EXIT_FAILURE);
	}

	fsync(fd);
	close(fd);
	return true;
}
