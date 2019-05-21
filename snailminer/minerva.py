import sys
if sys.version_info < (3, 6):
    from sha3 import sha3_512, sha3_256
else:
    from hashlib import sha3_512, sha3_256

from .dataset import TABLE_ORG
import numpy as np

DATALENGTH = 2048
PMTSIZE = 4
TBLSIZE = 16
HEADSIZE = 32
DGSTSIZE = 32
UPDATABLOCKLENGTH = 12000
STARTUPDATENUM = 10240
OFF_SKIP_LEN = 32768
OFF_CYCLE_LEN = 8192
SKIP_CYCLE_LEN = 2048
OFF_STATR = 12


class DataArray(object):

    def __init__(self, data, start=None):
        self.data = data
        self.capacity = len(self.data)
        if start is not None:
            self.start = start
        else:
            self.start = 0

    def slice(self, start=None, stop=None):
        return self.__class__(self.data, self.start+start)

    def __getitem__(self, i):
        return self.data[self.start+i]


def xor64(val):
    r = 0
    for _ in range(64):
        r ^= (val & 0x1)
        val = val >> 1

    return r


def multiple(input, prow):
    r = 0
    for k in range(32):
        if input[k] != 0 and prow[k] != 0:
            r ^= xor64(input[k] & prow[k])

    return r


def mat_multiple(input, output, pmat):
    prow = pmat
    point = 0

    for k in range(2048):

        kI = k // 64
        kR = k % 64
#       tab = prow[point:]
        tab = prow.slice(point)
        temp = multiple(input, tab)

        output[kI] |= (temp << kR)
        point += 32

    return output


def shift2048(data, sf):
    sfI = sf // 64
    sfR = sf % 64
    mask = (1 << sfR) - 1
    bits = (64 - sfR)
    res = 0
    if sfI == 1:
        val = data[0]
        for k in range(31):
            data[k] = data[k+1]
        data[31] = val

    res = (data[0] & mask) << bits

    for k in range(31):
        val = (data[k+1] & mask) << bits
        data[k] = (data[k] >> sfR) + val

    data[31] = (data[31] >> sfR) + res

    return data


def scramble(permute_in, dataset):
    permute_out = [0]*32

    for k in range(64):
        sf = permute_in[0] & 0x7f
        bs = permute_in[31] >> 60

#       ptbl = dataset[bs*2048*32:]
        ptbl = dataset.slice(bs*2048*32)

        mat_multiple(permute_in, permute_out, ptbl)
        shift2048(permute_out, sf)

        for k in range(32):
            permute_in[k] = permute_out[k]
            permute_out[k] = 0

    return permute_in


def fruit_hash(dataset, mining_hash, nonce):
    seed = [0] * 64
    output = [0] * DGSTSIZE

    val0 = nonce & 0xFFFFFFFF
    val1 = nonce >> 32

    for k in reversed(range(4)):
        seed[k] = val0 & 0xFF
        val0 >>= 8

    for k in reversed(range(4, 8)):
        seed[k] = val1 & 0xFF
        val1 >>= 8

    dgst = [0] * DGSTSIZE

    for k in range(HEADSIZE):
        seed[k+8] = mining_hash[k]

    sha512_out = sha3_512(bytes(seed)).digest()
    sha512_out = bytes(reversed(sha512_out))

    permute_in = [0] * 32

    for k in range(8):
        for x in range(8):
            sft = x * 8
            val = sha512_out[k*8+x] << sft
            permute_in[k] += val

    for k in range(1, 4):
        for x in range(8):
            permute_in[k*8+x] = permute_in[x]

    scramble(permute_in, dataset)

    dat_in = [0]*256

    for k in range(32):
        val = permute_in[k]
        for x in range(8):
            dat_in[k*8+x] = val & 0xFF
            val = val >> 8

    for k in range(64):
        temp = dat_in[k*4]
        dat_in[k*4] = dat_in[k*4+3]
        dat_in[k*4+3] = temp
        temp = dat_in[k*4+1]
        dat_in[k*4+1] = dat_in[k*4+2]
        dat_in[k*4+2] = temp

    output = sha3_256(bytes(dat_in)).digest()

    return output


def table_init():
    tab = [0] * (TBLSIZE * DATALENGTH * PMTSIZE)
    for k in range(TBLSIZE):
        for x in range(DATALENGTH * PMTSIZE):
            tab[k * DATALENGTH * PMTSIZE + x] = TABLE_ORG[k][x]

    lookup = [0] * (TBLSIZE*DATALENGTH*PMTSIZE*32)
    lktWz = DATALENGTH // 64
    lktSz = DATALENGTH * lktWz

    for k in range(TBLSIZE):
        plkt = k * lktSz

        for x in range(DATALENGTH):
            c = 0
            for y in range(PMTSIZE):
                val = tab[k*DATALENGTH*PMTSIZE+x*PMTSIZE+y]
                if val == 0xFFF:
                    continue
                vI = val // 64
                vR = val % 64
                lookup[plkt+vI] |= 1 << vR
                c = c + 1
            if c == 0:
                vI = x // 64
                vR = x % 64
                lookup[plkt+vI] |= 1 << vR
            plkt += lktWz

    return lookup

