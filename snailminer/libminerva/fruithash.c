#include "hash.h"
#include "sha3.h"

int kXor[256];

//int genLookupTable(uint64_t *plookup, uint32 *ptable)
//{
//        uint64_t *ptbl = plookup;
//        int lkt_wz = (DATA_LENGTH) / 64;
//        int lkt_sz = DATA_LENGTH*lkt_wz;
//
//        int idx = 0;
//        for (int k = 0; k < 16; k++)
//        {
//                uint64_t *plkt = plookup+k*lkt_sz;
//                uint32 *ptbl = ptable + k*DATA_LENGTH*PMT_SIZE;
//                for (int x = 0; x < DATA_LENGTH; x++)
//                {
//                        if (x == 0 && k == 13)
//                                x = x;
//                        for (int y = 0; y < PMT_SIZE; y++)
//                        {
//                                int val = *ptable;
//                                if (val == 0xFFF)
//                                {
//                                        ptable++;
//                                        continue;
//                                }
//
//                                int v_i = val / 64;
//                                int v_r = val % 64;
//                                plkt[v_i] |= ((uint64_t)1 << v_r);
//                                ptable++;
//                        }
//                        plkt += lkt_wz;
//                }
//                //printf("\n");
//        }
//
//        return 0;
//}

int xor64(uint64_t val) {
        int r  = 0;

        for (int k = 0; k < 64; k++) {
                r ^= (int)(val & 0x1);
                val = val >> 1;
        }
        return r;
}

int muliple(uint64_t input[32], uint64_t *prow)
{
        int r = 0;
        for (int k = 0; k < 32; k++)
        {
                if (input[k] != 0 && prow[k] != 0)
                        r ^= xor64(input[k] & prow[k]);
        }

        return r;
}

int MatMuliple(uint64_t input[32], uint64_t output[32], uint64_t pmat[])
{
    uint64_t *prow = pmat;

    for (int k = 0; k < 2048; k++)
    {
        int k_i = k / 64;
        int k_r = k % 64;
        unsigned int temp;
        temp = muliple(input, prow);

        output[k_i] |= ((uint64_t)temp << k_r);
        prow += 32;
    }

    return 0;
}

int shift2048(uint64_t in[32], int sf)
{
    int sf_i = sf / 64;
    int sf_r = sf % 64;
    uint64_t mask = ((uint64_t)1 << sf_r) - 1;
    int bits = (64 - sf_r);
    uint64_t res;

    if (sf_i == 1)
    {
        uint64_t val = in[0];
        for (int k = 0; k < 31; k++)
        {
            in[k] = in[k + 1];
        }
        in[31] = val;
    }
    res = (in[0] & mask) << bits;
    for (int k = 0; k < 31; k++)
    {
        uint64_t val = (in[k + 1] & mask) << bits;
        in[k] = (in[k] >> sf_r) + val;
    }
    in[31] = (in[31] >> sf_r) + res;
    return 0;
}

int scramble(uint64_t *permute_in, uint64_t dataset[])
{
    uint64_t *ptbl;
    uint64_t permute_out[32] = { 0 };

    for (int k = 0; k < 64; k++)
    {
        int sf, bs;
        sf = permute_in[0] & 0x7f;
        bs = permute_in[31] >> 60;
        ptbl = dataset + bs * 2048 * 32;
        MatMuliple(permute_in, permute_out, ptbl);

        shift2048(permute_out, sf);
        for (int k = 0; k < 32; k++)
        {
            permute_in[k] = permute_out[k];
            permute_out[k] = 0;
        }
    }

    return 0;
}

int byteReverse(uint8_t sha512_out[64])
{
    uint8_t temp;

    for (int k = 0; k < 32; k++)
    {
        temp = sha512_out[k];
        sha512_out[k] = sha512_out[63 - k];
        sha512_out[63 - k] = temp;
    }

    return 0;
}

//int convertLE(uint8_t header[HEAD_SIZE])
//{
//    int wz = HEAD_SIZE / 4;
//
//    for (int k = 0; k < wz; k++)
//    {
//        uint8_t temp[4];
//        temp[0] = header[k * 4 + 3];
//        temp[1] = header[k * 4 + 2];
//        temp[2] = header[k * 4 + 1];
//        temp[3] = header[k * 4 + 0];
//        header[k * 4 + 0] = temp[0];
//        header[k * 4 + 1] = temp[1];
//        header[k * 4 + 2] = temp[2];
//        header[k * 4 + 3] = temp[3];
//    }
//    return 0;
//}

//int convertWD(uint8_t header[HEAD_SIZE])
//{
//    uint8_t temp[HEAD_SIZE];
//    int wz = HEAD_SIZE / 4;
//    for (int k = 0; k < wz; k++)
//    {
//        int i = 7 - k;
//        temp[k * 4] = header[i * 4];
//        temp[k * 4 + 1] = header[i * 4 + 1];
//        temp[k * 4 + 2] = header[i * 4 + 2];
//        temp[k * 4 + 3] = header[i * 4 + 3];
//    }
//    for (int k = 0; k < HEAD_SIZE; k++)
//    {
//        header[k] = temp[k];
//    }
//    return 0;
//}

//int compare(byte dgst[DGST_SIZE], byte target1[TARG_SIZE], byte target2[TARG_SIZE])
//{
//        for (int k = TARG_SIZE - 1; k >= 0; k--)
//        {
//                int dif = (int)dgst[k] - (int)target1[k];
//                if (dif > 0)
//                        return 0;
//                if (dif < 0)
//                        return 1;
//        }
//        for (int k = TARG_SIZE - 1; k >= 0; k--)
//        {
//                int dif = (int)dgst[k + 16] - (int)target2[k];
//                if (dif > 0)
//                        return 0;
//                if (dif < 0)
//                        return 1;
//        }
//        return 0;
//}


//void compute(uint64_t nonce_start)
//{
//        byte digs[DGST_SIZE];
//        const uint64_t offset = gridDim.x * blockDim.x;
//        nonce_start += threadIdx.x + blockIdx.x * blockDim.x;
//
//        while (nonce_start < gFoundIdx)
//        {
//                fchainhash(nonce_start, digs);
//
//                if (compare(digs, kTarget1, kTarget2) == 1)
//                {
//                        atomicMin((unsigned long long int*)&gFoundIdx, unsigned long long int(nonce_start));
//                        break;
//                }
//                // Get result here
//                printf("Current nonce : %llu\n", nonce_start);
//                nonce_start += offset;
//        }
//}

void fchainhash(uint64_t dataset[], uint8_t mining_hash[DGST_SIZE], uint64_t nonce, uint8_t digs[DGST_SIZE])
{
        uint8_t seed[64] = { 0 };
        uint8_t output[DGST_SIZE] = { 0 };

        uint32_t val0 = (uint32_t)(nonce & 0xFFFFFFFF);
        uint32_t val1 = (uint32_t)(nonce >> 32);
        for (int k = 3; k >= 0; k--)
        {
                seed[k] = val0 & 0xFF;
                val0 >>= 8;
        }

        for (int k = 7; k >= 4; k--)
        {
                seed[k] = val1 & 0xFF;
                val1 >>= 8;
        }

        for (int k = 0; k < HEAD_SIZE; k++)
        {
                seed[k+8] = mining_hash[k];
        }

        uint8_t sha512_out[64];
        sha3(seed, 64, sha512_out, 64);
        byteReverse(sha512_out);
        uint64_t permute_in[32] = { 0 };
        for (int k = 0; k < 8; k++)
        {
                for (int x = 0; x < 8; x++)
                {
                        int sft = x * 8;
                        uint64_t val = ((uint64_t)sha512_out[k*8+x] << sft);
                        permute_in[k] += val;
                }
        }

        for (int k = 1; k < 4; k++)
        {
                for (int x = 0; x < 8; x++)
                        permute_in[k * 8 + x] = permute_in[x];
        }

        scramble(permute_in, dataset);

        uint8_t dat_in[256];
        for (int k = 0; k < 32; k++)
        {
                uint64_t val = permute_in[k];
                for (int x = 0; x < 8; x++)
                {
                        dat_in[k * 8 + x] = val & 0xFF;
                        val = val >> 8;
                }
        }

        for (int k = 0; k < 64; k++)
        {
                uint8_t temp;
                temp = dat_in[k * 4];
                dat_in[k * 4] = dat_in[k * 4 + 3];
                dat_in[k * 4 + 3] = temp;
                temp = dat_in[k * 4 + 1];
                dat_in[k * 4 + 1] = dat_in[k * 4 + 2];
                dat_in[k * 4 + 2] = temp;
        }

        //unsigned char output[64];
        sha3(dat_in, 256, output, 32);
        // reverse byte
        for (int k = 0; k < DGST_SIZE; k++)
        {
                digs[k] = output[DGST_SIZE - k - 1];
        }
}
