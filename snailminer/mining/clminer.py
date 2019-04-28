import time
import os
import sys
import logging
from hashlib import sha3_512, sha3_256
from queue import Empty
from threading import Lock

import pyopencl as cl
import numpy
import numpy as np
from snailminer import minerva
from snailminer.mining.miner import Miner

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG = logging.getLogger(__name__)


def initialize():
    platforms = cl.get_platforms()
    assert len(platforms) > 0, 'No OpenCL platforms support'

    platforms = cl.get_platforms()
    devices = platforms[0].get_devices(cl.device_type.GPU)

    if devices:
        LOG.debug('OpenCL devices:')
        for i in range(len(devices)):
            LOG.debug('[%d]\t%s' % (i, devices[i].name))

    return devices


class OpenCLMiner(Miner):

    def __init__(self, work_queue=None, result_queue=None):
        super(OpenCLMiner, self).__init__(None, {})
        self.output_size = 8
        self.defines = ''
        self.defines += (' -DOUTPUT_SIZE=' + str(self.output_size))
        self.device = initialize()[0]
        self.device_name = self.device.name.strip('\r\n \x00\t')
        self.work_queue = work_queue
        self.result_queue = result_queue
        self.current = None
        LOG.debug('device name: %s' % self.device_name)

    def load_kernel(self):
        self.context = cl.Context([self.device], None, None)
        with open(os.path.join(BASE_DIR, 'truehash.cl')) as kernel_file:
            kernel = kernel_file.read()

        self.program = cl.Program(self.context, kernel).build(self.defines)
        self.kernel = self.program.search

        self.worksize = self.kernel.get_work_group_info(cl.kernel_work_group_info.WORK_GROUP_SIZE, self.device)
        LOG.info('set worksize=%s' % self.worksize)

    def mining_thread(self):
        self.load_kernel()

        dataset = minerva.table_init()
        queue = cl.CommandQueue(self.context)
        # epoch dataset
        dataset = numpy.array(dataset, dtype=numpy.uint64)
        dataset_buf = cl.Buffer(self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=dataset)

        """
        # example of hash header ''
        # mining header hash of 32 bytes
        #header = numpy.zeros(32, numpy.uint8)
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


        self.kernel.set_arg(0, dataset_buf)
        self.kernel.set_arg(1, header_buf)
        self.kernel.set_arg(2, target_buf)
        self.kernel.set_arg(3, numpy.uint64(3))
        self.kernel.set_arg(4, output_buf)
        cl.enqueue_nd_range_kernel(queue, self.kernel, (1,), (1,))

        """

        while True:

            if self.should_stop:
                LOG.info("stop clminer...")
                break

            if not self.current or not self.work_queue.empty():
                work = self.work_queue.get()
                if work is None:
                    # receive exit msg, abort the mining routine
                    break
                self.current = work.copy()
                LOG.info("fetch work job=%s, fruit=%s, target=%s",
                         work['header'][:10]+'..',
                         work['fruit_target'],
                         work['target'])

                # mining header hash of 32 bytes
                header = numpy.fromstring(bytes.fromhex(work['header'][2:]), dtype=numpy.uint8)
                header_buf = cl.Buffer(self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=header)
                # 16 bytes boundary for block diffculty
                target = numpy.fromstring(bytes.fromhex(work['target'][2:]), dtype=numpy.uint8)
                target_buf = cl.Buffer(self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=target)
                # 16 bytes boundary for fruit diffculty
                fruit_target = numpy.fromstring(bytes.fromhex(work['fruit_target'][2:]), dtype=numpy.uint8)
                fruit_buf = cl.Buffer(self.context,
                                             cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                                             hostbuf=fruit_target)
                # output nonce
                output = numpy.zeros(self.output_size, numpy.uint64)
                output_buf = cl.Buffer(self.context, cl.mem_flags.WRITE_ONLY, output.nbytes)
                # digest
                digest = numpy.zeros(32*self.output_size, numpy.uint8)
                digest_buf = cl.Buffer(self.context, cl.mem_flags.WRITE_ONLY, digest.nbytes)
                # output count
                count = numpy.zeros(1, numpy.uint32)
                count_buf = cl.Buffer(self.context, cl.mem_flags.WRITE_ONLY, count.nbytes)

                start_nonce = work['nonce']

            self.kernel(queue, (self.worksize,), None, dataset_buf, header_buf, target_buf, fruit_buf, numpy.uint64(start_nonce), output_buf, digest_buf, count_buf)

            cl.enqueue_copy(queue, output, output_buf)
            cl.enqueue_copy(queue, digest, digest_buf)
            cl.enqueue_copy(queue, count, count_buf)

            LOG.debug("searching start=%s, count=%s" % (start_nonce, count[0]))
            if count > 0:
                # result just containing work package detail
                for index in range(count[0]):
                    result = self.current.copy()
                    result['digest'] = '0x' + digest[index*32:(index+1)*32].tobytes().hex()
                    result['found_nonce'] = output[index]
                    LOG.info("search found job=%s nonce=%s digest=%s",
                             result['header'][:10]+'..',
                             output[index],
                             result['digest'])
                    self.result_queue.put(result)

                # clear global output buffer
                cl.enqueue_copy(queue, output_buf, numpy.zeros(self.output_size, numpy.uint64))
                cl.enqueue_copy(queue, digest_buf, numpy.zeros(32*self.output_size, numpy.uint8))
                cl.enqueue_copy(queue, count_buf, numpy.zeros(1, numpy.uint32))

                _target = int(self.current['target'], 0)
                if _target > 0 and int.from_bytes(digest[:16], "big") < _target:
                    # Mined block
                    LOG.info("found block digest=%s", result['digest'])
                    self.current = None

            start_nonce += self.worksize

