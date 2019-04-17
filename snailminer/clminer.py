from miner import Miner
from queue import Empty
from struct import pack, unpack, error
from threading import Lock
from time import sleep, time
import sys
from timeit import default_timer as timer
from hashlib import sha3_512, sha3_256

import pyopencl as cl
import numpy
import numpy as np
import minerva


def initialize():
    platforms = cl.get_platforms()
    assert len(platforms) > 0, 'No OpenCL platforms support'

    platforms = cl.get_platforms()
    devices = platforms[0].get_devices(cl.device_type.GPU)

    if devices:
        print('OpenCL devices:')
        for i in range(len(devices)):
            print('[%d]\t%s' % (i, devices[i].name))

    return devices


class OpenCLMiner(Miner):

    def __init__(self):
        super(OpenCLMiner, self).__init__(None, {})
        self.defines = ''
        self.device = initialize()[0]
        self.device_name = self.device.name.strip('\r\n \x00\t')
        print('device name: %s' % self.device_name)

    def load_kernel(self):
        self.context = cl.Context([self.device], None, None)
        with open('truehash.cl') as kernel_file:
            kernel = kernel_file.read()

        self.program = cl.Program(self.context, kernel).build(self.defines)
        self.kernel = self.program.search

        self.worksize = self.kernel.get_work_group_info(cl.kernel_work_group_info.WORK_GROUP_SIZE, self.device)
        print('worksize %s' % self.worksize)


    def mining_thread(self):
        m.load_kernel()

        dataset = minerva.table_init()
        queue = cl.CommandQueue(self.context)

#       import pdb
#       pdb.set_trace()

        # epoch dataset
        dataset = numpy.array(dataset, dtype=numpy.uint64)
        dataset_buf = cl.Buffer(self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=dataset)
        # mining header hash of 32 bytes
#       header = numpy.zeros(32, numpy.uint8)
        header = numpy.fromstring(sha3_256(b'').digest(), dtype=numpy.uint8)
        header_buf = cl.Buffer(self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=header)
        # 16 bytes boundary for block diffculty
        target = numpy.zeros(16, numpy.uint8)
        target_buf = cl.Buffer(self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=target)
        # output nonce
        output = numpy.zeros(2, numpy.uint64)
        output_buf = cl.Buffer(self.context, cl.mem_flags.WRITE_ONLY, output.nbytes)

        # digest
        digest = numpy.zeros(32, numpy.uint8)
        digest_buf = cl.Buffer(self.context, cl.mem_flags.WRITE_ONLY, digest.nbytes)

        while True:
            print(timer())
#           self.kernel.set_arg(0, dataset_buf)
#           self.kernel.set_arg(1, header_buf)
#           self.kernel.set_arg(2, target_buf)
#           self.kernel.set_arg(3, numpy.uint64(3))
#           self.kernel.set_arg(4, output_buf)
#           cl.enqueue_nd_range_kernel(queue, self.kernel, (1,), (1,))
            self.kernel(queue, (1,), None, dataset_buf, header_buf, target_buf, numpy.uint64(0), output_buf, digest_buf)
            cl.enqueue_copy(queue, output, output_buf)
            cl.enqueue_copy(queue, digest, digest_buf)
            print(timer())
            print("nonce: %s, digest:%s" % (output[0], digest))

            self.kernel(queue, (1,), None, dataset_buf, header_buf, target_buf, numpy.uint64(0), output_buf, digest_buf)
            cl.enqueue_copy(queue, output, output_buf)
            cl.enqueue_copy(queue, digest, digest_buf)
            print(timer())
            print("nonce: %s, digest:%s" % (output[0], digest))
#           print("nonce: %s, digest:%s" % (output[0], digest))

            break



if __name__ == '__main__':

    m = OpenCLMiner()
    m.mining_thread()
