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

#ifndef __CUFILE_SAMPLE_SHASUM_HPP_
#define __CUFILE_SAMPLE_SHASUM_HPP_

#include <array>

#include <openssl/sha.h>
#include <openssl/evp.h>

#include "cufile_sample_utils.hpp"

#define GDSTOOLS_CRYPTO_LIB_A "libcrypto.so.1.1"
#define GDSTOOLS_CRYPTO_LIB_B "libcrypto.so.10"
#define GDSTOOLS_CRYPTO_LIB_C "libssl.so.3"
 
constexpr size_t SHA256_CHUNK_SIZE = KiB(64);

enum OSMemoryType {
	HOST,
	DEVICE
};

// Function pointers for libcrypto's symbols when using dlopen
using SHA256_Init_func_v1 = int(*)(SHA256_CTX*);
using SHA256_Update_func_v1 = int(*)(SHA256_CTX*, const void*, size_t);
using SHA256_Final_func_v1 = int(*)(unsigned char*, SHA256_CTX*);
using SHA256_Init_func_v3 = int(*)(EVP_MD_CTX*, const EVP_MD*, ENGINE*);
using SHA256_Update_func_v3 = int(*)(EVP_MD_CTX*, const void*, size_t);
using SHA256_Final_func_v3 = int(*)(EVP_MD_CTX*, unsigned char*, unsigned int*);
using SHA256_Mdctx_create_func = EVP_MD_CTX*(*)();
using SHA256_Mdctx_destroy_func = void(*)(EVP_MD_CTX*);
using SHA256_get_digestbyname_func = EVP_MD*(*)(const char*);

// Variable for SSL library version
extern bool ssl_lib_v3;

// Handle for the SHA256 library
extern void *SHA256_lib_handle;

// Variables for SHA256 symbols
extern SHA256_Init_func_v1 SHA256_Init_v1_p;
extern SHA256_Update_func_v1 SHA256_Update_v1_p;
extern SHA256_Final_func_v1 SHA256_Final_v1_p;
extern SHA256_Init_func_v3 SHA256_Init_v3_p;
extern SHA256_Update_func_v3 SHA256_Update_v3_p;
extern SHA256_Final_func_v3 SHA256_Final_v3_p;
extern SHA256_Mdctx_create_func SHA256_Mdctx_create_p;
extern SHA256_Mdctx_destroy_func SHA256_Mdctx_destroy_p;
extern SHA256_get_digestbyname_func SHA256_get_digestbyname_p;

// Helpers for maintaining SHA256 symbols
bool FailedToLoadSHA256Symbols();
void UnLoadSHA256Symbols();
bool LoadSHA256Symbols();

// Helpers for creating a SHA256 sample
const EVP_MD *EVP_get_digestbyname_sample(const char *name);
EVP_MD_CTX *SHA256_Mdctx_create_sample();
void SHA256_Mdctx_destroy_sample(void *ctx);
int SHA256_Init_sample(void *ctx);
int SHA256_Update_sample(void *ctx, const void *data, size_t len);
int SHA256_Final_sample(unsigned char *md, void *ctx);

////////////////////////////////////////////////////////////////////////////
//! SHASUM256 routine : computes digest of nbytes of a file
//! @param[in] fpath  File path
//! @param[in] md  SHA256 digest
//! @param[in] bytes  Number of bytes to compute digest for
//! @param[in] offset  Offset into the file
//! @return 0 on success, -1 on error
////////////////////////////////////////////////////////////////////////////
int SHASUM256(const std::string& fpath, std::array<unsigned char, SHA256_DIGEST_LENGTH>& md, size_t bytes = 0, size_t offset = 0);

////////////////////////////////////////////////////////////////////////////
//! SHASUM256_MEM routine : computes digest of nbytes from a (device or host) memory region
//! @param[in] memType  Memory type
//! @param[in] hostPtr  Host pointer
//! @param[in] memSize  Memory size
//! @param[in] md  SHA256 digest
//! @param[in] ptrOff  Pointer offset
//! @param[in] bytes  Number of bytes to compute digest for
//! @return 0 on success, -1 on error
////////////////////////////////////////////////////////////////////////////
int SHASUM256_MEM(OSMemoryType memType, char *hostPtr, size_t memSize, std::array<unsigned char, SHA256_DIGEST_LENGTH>& md, size_t ptrOff, size_t bytes = 0);

////////////////////////////////////////////////////////////////////////////
//! DumpSHASUM function : dumps the SHA256 digest to the console
//! @param[in] md  SHA256 digest
////////////////////////////////////////////////////////////////////////////
void DumpSHASUM(std::array<unsigned char, SHA256_DIGEST_LENGTH>& md);
 
#endif
