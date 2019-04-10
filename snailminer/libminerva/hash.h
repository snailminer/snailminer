#ifndef  _HASH_H_
#define  _HASH_H_

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

#define DATA_LENGTH     2048
#define PMT_SIZE        4
#define TBL_SIZE        16
#define HEAD_SIZE       32
#define DGST_SIZE       32
#define TARG_SIZE       16


void fchainhash(uint64_t dataset[], uint8_t mining_hash[DGST_SIZE], uint64_t nonce, uint8_t digs[DGST_SIZE]);
//void compute(uint64_t nonce_start);

#endif
