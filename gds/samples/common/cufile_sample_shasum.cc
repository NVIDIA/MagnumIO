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
#include <fstream>
#include <array>
#include <algorithm>

#include <builtin_types.h>
#include <cuda.h>
#include <dlfcn.h>

#include "../include/cufile_sample_shasum.hpp"

bool ssl_lib_v3 = false;
void *SHA256_lib_handle = nullptr;
SHA256_Init_func_v1 SHA256_Init_v1_p = nullptr;
SHA256_Update_func_v1 SHA256_Update_v1_p = nullptr;
SHA256_Final_func_v1 SHA256_Final_v1_p = nullptr;
SHA256_Init_func_v3 SHA256_Init_v3_p = nullptr;
SHA256_Update_func_v3 SHA256_Update_v3_p = nullptr;
SHA256_Final_func_v3 SHA256_Final_v3_p = nullptr;
SHA256_Mdctx_create_func SHA256_Mdctx_create_p = nullptr;
SHA256_Mdctx_destroy_func SHA256_Mdctx_destroy_p = nullptr;
SHA256_get_digestbyname_func SHA256_get_digestbyname_p = nullptr;

bool FailedToLoadSHA256Symbols() {
	std::cerr << "Unable to load SHA256 symbols" << std::endl;
	dlclose(SHA256_lib_handle);
	SHA256_lib_handle = nullptr;
	SHA256_Init_v1_p = nullptr;
	SHA256_Init_v3_p = nullptr;
	SHA256_Update_v1_p = nullptr;
	SHA256_Update_v3_p = nullptr;
	SHA256_Final_v1_p = nullptr;
	SHA256_Final_v3_p = nullptr;
	SHA256_Mdctx_create_p = nullptr;
	SHA256_Mdctx_destroy_p = nullptr;
	SHA256_get_digestbyname_p = nullptr;
	return false;
}

void UnLoadSHA256Symbols() {
	if (SHA256_lib_handle) {
		dlclose(SHA256_lib_handle);
		SHA256_lib_handle = nullptr;
		SHA256_Init_v1_p = nullptr;
		SHA256_Init_v3_p = nullptr;
		SHA256_Update_v1_p = nullptr;
		SHA256_Update_v3_p = nullptr;
		SHA256_Final_v1_p = nullptr;
		SHA256_Final_v3_p = nullptr;
		SHA256_Mdctx_create_p = nullptr;
		SHA256_Mdctx_destroy_p = nullptr;
		SHA256_get_digestbyname_p = nullptr;
	}
}

bool LoadSHA256Symbols() {
	std::vector<std::string> lib_names = {
		GDSTOOLS_CRYPTO_LIB_C, 
		GDSTOOLS_CRYPTO_LIB_A, 
		GDSTOOLS_CRYPTO_LIB_B
	};

	for (const auto& lib_name : lib_names) {
		SHA256_lib_handle = dlopen(lib_name.c_str(), RTLD_GLOBAL| RTLD_NOW);
		if (SHA256_lib_handle != nullptr) {
			ssl_lib_v3 = lib_name == GDSTOOLS_CRYPTO_LIB_C;
			break;
		}
	}

	if (SHA256_lib_handle == nullptr) {
		std::cerr << "Please install" << GDSTOOLS_CRYPTO_LIB_A << " or " 
			<< GDSTOOLS_CRYPTO_LIB_B << " or " << GDSTOOLS_CRYPTO_LIB_C 
			<< "depending on your platform " << std::endl;
		return false;
	}	

	if (!ssl_lib_v3) {
		SHA256_Init_v1_p = (SHA256_Init_func_v1) dlsym(SHA256_lib_handle, "SHA256_Init");
		if (SHA256_Init_v1_p == nullptr) {
			return FailedToLoadSHA256Symbols();
		}

		SHA256_Update_v1_p = (SHA256_Update_func_v1) dlsym(SHA256_lib_handle, "SHA256_Update");
		if (SHA256_Update_v1_p == nullptr) {
			return FailedToLoadSHA256Symbols();
		}

		SHA256_Final_v1_p = (SHA256_Final_func_v1) dlsym(SHA256_lib_handle, "SHA256_Final");
		if (SHA256_Final_v1_p == nullptr) {
			return FailedToLoadSHA256Symbols();
		}
	} else {
		SHA256_Init_v3_p = (SHA256_Init_func_v3) dlsym(SHA256_lib_handle, "EVP_DigestInit_ex");
		if (SHA256_Init_v3_p == nullptr) {
			std::cerr << "Unable to load EVP_DigestInit_ex symbols" << std::endl;
			return FailedToLoadSHA256Symbols();
		}

		SHA256_Update_v3_p = (SHA256_Update_func_v3) dlsym(SHA256_lib_handle, "EVP_DigestUpdate");
		if (SHA256_Update_v3_p == nullptr) {
			std::cerr << "Unable to load EVP_DigestUpdate symbols" << std::endl;
			return FailedToLoadSHA256Symbols();
		}

		SHA256_Final_v3_p = (SHA256_Final_func_v3) dlsym(SHA256_lib_handle, "EVP_DigestFinal_ex");
		if (SHA256_Final_v3_p == nullptr) {
			std::cerr << "Unable to load EVP_DigestFinal_ex symbols" << std::endl;
			return FailedToLoadSHA256Symbols();
		}

		SHA256_Mdctx_create_p = (SHA256_Mdctx_create_func) dlsym(SHA256_lib_handle, "EVP_MD_CTX_new");
		if (SHA256_Mdctx_create_p == nullptr) {
			std::cerr << "Unable to load EVP_MD_CTX_new symbols" << std::endl;
			return FailedToLoadSHA256Symbols();
		}

		SHA256_Mdctx_destroy_p = (SHA256_Mdctx_destroy_func) dlsym(SHA256_lib_handle, "EVP_MD_CTX_free");
		if (SHA256_Mdctx_destroy_p == nullptr) {
			std::cerr << "Unable to load EVP_MD_CTX_free symbols" << std::endl;
			return FailedToLoadSHA256Symbols();
		}

		SHA256_get_digestbyname_p = (SHA256_get_digestbyname_func) dlsym(SHA256_lib_handle, "EVP_get_digestbyname");
		if (SHA256_get_digestbyname_p == nullptr) {
			std::cerr << "Unable to load EVP_get_digestbyname symbols" << std::endl;
			return FailedToLoadSHA256Symbols();
		}
	}

	return true;
}

const EVP_MD *EVP_get_digestbyname_sample(const char *name) {
	if (SHA256_get_digestbyname_p) {
		return SHA256_get_digestbyname_p(name);
	}
	return nullptr;
}

EVP_MD_CTX *SHA256_Mdctx_create_sample() {
	if (SHA256_Mdctx_create_p) {
		return SHA256_Mdctx_create_p();
	}
	return nullptr;
}

void SHA256_Mdctx_destroy_sample(void *ctx) {
	if (SHA256_Mdctx_destroy_p) {
		SHA256_Mdctx_destroy_p(static_cast<EVP_MD_CTX*>(ctx));
	}
}

int SHA256_Init_sample(void *ctx) {
	if (!ssl_lib_v3) {
		if (SHA256_Init_v1_p) {
			return SHA256_Init_v1_p(static_cast<SHA256_CTX*>(ctx));
		}
	} else {
		if (SHA256_Init_v3_p) {
			return SHA256_Init_v3_p(static_cast<EVP_MD_CTX*>(ctx), EVP_get_digestbyname_sample("sha256"), nullptr);
		} 
	}
	return 0;
}

int SHA256_Update_sample(void *ctx, const void *data, size_t len) {
	if (!ssl_lib_v3) {
		if (SHA256_Update_v1_p) {
			return SHA256_Update_v1_p(static_cast<SHA256_CTX*>(ctx), data, len);
		}
	} else {
		if (SHA256_Update_v3_p) {
			return SHA256_Update_v3_p(static_cast<EVP_MD_CTX*>(ctx), data, len);
		}
	}
	return 0;
}

int SHA256_Final_sample(unsigned char *md, void *ctx) {
	if (!ssl_lib_v3) {
		if (SHA256_Final_v1_p) {
			return SHA256_Final_v1_p(md, static_cast<SHA256_CTX*>(ctx));
		}
	} else {
		if (SHA256_Final_v3_p) {
			unsigned int n;
			return SHA256_Final_v3_p(static_cast<EVP_MD_CTX*>(ctx), md, &n);
		}
	}
	return 0;
}

int SHASUM256(const std::string& fpath, std::array<unsigned char, SHA256_DIGEST_LENGTH>& md, size_t bytes, size_t offset) {
	std::array<char, SHA256_CHUNK_SIZE> buf;
	EVP_MD_CTX *mdCtx = nullptr;
	SHA256_CTX shaCtx;
	size_t filesize;

	std::ifstream fp(fpath, std::ifstream::in | std::ifstream::binary);
	if (!fp.is_open()) {
		std::cerr << "file open failed" << std::endl;
		return -1;
	}
	
	// Get total file size
	fp.seekg(0, fp.end);
	filesize = fp.tellg();

	if (filesize == 0) {
		fp.close();
		std::cerr << "file is empty" << std::endl;
		return -1;
	}

	if (offset > filesize) {
		fp.close();
		std::cerr << "offset (" << offset << ") exceeds file size (" << filesize << ")" << std::endl;
		return -1;
	}

	// Move to offset position
	fp.seekg(offset, fp.beg);

	size_t remaining = filesize - offset;
	if (bytes == 0 || bytes > remaining) {
		bytes = remaining;
	}

	if (bytes == 0) {
		fp.close();
		std::cerr << "no bytes to read after applying offset" << std::endl;
		return -1;
	}

	if (LoadSHA256Symbols() == false) {
		std::cerr << "libcrypto not loaded" << std::endl;
		return -1;
	}

	if (ssl_lib_v3) {
		if ((mdCtx = SHA256_Mdctx_create_sample()) == nullptr) {
			std::cerr << "MD context creation failed" << std::endl;
			return -1;
		}
		SHA256_Init_sample(static_cast<void*>(mdCtx));
	} else {
		SHA256_Init_sample(static_cast<void*>(&shaCtx));
	}

	while (bytes && !fp.eof()) {
		size_t to_read = std::min(bytes, static_cast<size_t>(SHA256_CHUNK_SIZE));
		fp.read(buf.data(), to_read);
		size_t read_bytes = fp.gcount();
		if (read_bytes == 0) break;

		if (!ssl_lib_v3) {
			SHA256_Update_sample(static_cast<void*>(&shaCtx), buf.data(), read_bytes);
		} else {
			SHA256_Update_sample(static_cast<void*>(mdCtx), buf.data(), read_bytes);
		}
		bytes -= read_bytes;
	}

	fp.close();

	if (!ssl_lib_v3) {
		SHA256_Final_sample(md.data(), static_cast<void*>(&shaCtx));
	} else {
		SHA256_Final_sample(md.data(), static_cast<void*>(mdCtx));
		SHA256_Mdctx_destroy_sample(static_cast<void*>(mdCtx));
	}

	UnLoadSHA256Symbols();
	return 0;
}

int SHASUM256_MEM(OSMemoryType memType, char *ptr, size_t memSize, std::array<unsigned char, SHA256_DIGEST_LENGTH>& md, size_t ptrOff, size_t bytes) {
	EVP_MD_CTX *mdCtx = nullptr;
	SHA256_CTX shaCtx;

	std::array<char, SHA256_CHUNK_SIZE> buf;
	char *offsetPtr = ptr + ptrOff;

	if (memSize <= ptrOff) {
		std::cerr << "invalid parameters" << std::endl;
		return -1;
	}

	if (bytes > (memSize - ptrOff)) {
		std::cerr << bytes << ":" << (memSize - ptrOff) << std::endl;
		std::cerr << "bytes more than size" << std::endl;
		return -1;
	}

	if (!bytes)
		bytes = memSize - ptrOff;

	if (LoadSHA256Symbols() == false) {
		std::cerr << "libcrypto not loaded" << std::endl;
		return -1;
	}

	if (ssl_lib_v3) {
		if ((mdCtx = SHA256_Mdctx_create_sample()) == nullptr) {
			std::cerr << "MD context creation failed" << std::endl;
			return -1;
		}
		SHA256_Init_sample(static_cast<void*>(mdCtx));
	} else {
		SHA256_Init_sample(static_cast<void*>(&shaCtx));
	}

	while (bytes) {
		size_t size = std::min(bytes, static_cast<size_t>(SHA256_CHUNK_SIZE));
		if (memType == OSMemoryType::DEVICE) {
			cudaMemcpy(buf.data(), offsetPtr, size, cudaMemcpyDeviceToHost);
		} else {
			std::memcpy(buf.data(), offsetPtr, size);
		}
		if (!ssl_lib_v3) {
			SHA256_Update_sample(static_cast<void*>(&shaCtx), buf.data(), size);
		} else {
			SHA256_Update_sample(static_cast<void*>(mdCtx), buf.data(), size);
		}
		bytes -= size;
		offsetPtr += size;
	}

	if (!ssl_lib_v3) {
		SHA256_Final_sample(md.data(), static_cast<void*>(&shaCtx));
	} else {
		SHA256_Final_sample(md.data(), static_cast<void*>(mdCtx));
		SHA256_Mdctx_destroy_sample(static_cast<void*>(mdCtx));
	}

	UnLoadSHA256Symbols();
	return 0;
}

void DumpSHASUM(std::array<unsigned char, SHA256_DIGEST_LENGTH>& md) {
	for (int i = 0; i < SHA256_DIGEST_LENGTH ; i++)
		std::cout << std::hex << static_cast<int>(md[i]);
	std::cout << std::dec << std::endl;
}