from queue import Queue

from tornado import (
    gen,
    ioloop,
)

from snailminer.getwork import Getwork
from snailminer.mining.clminer import OpenCLMiner

class Collector(object):

    def __init__(self, endpoint, miners=None, work_queue=None):
        self.endpoint = endpoint
        self.miners = miners
        self.work_queue = work_queue
        self.getwork = Getwork(endpoint=endpoint,
                               collector=self)

    def enqueue_work(self, work):
        print("enqueue work :%s" % work)
        self.work_queue.put(work)

    async def run(self):
        while True:
            await self.getwork.request()
            await gen.sleep(3)

    def stop(self):
        self.work_queue.put(None)


def run():
    work_queue = Queue()
    m = OpenCLMiner(work_queue)
    m.start()

    pool = Collector('http://localhost:8545', work_queue=work_queue)
    io_loop = ioloop.IOLoop.current()
    try:
        io_loop.add_callback(pool.run)
        io_loop.start()
    except (KeyboardInterrupt, SystemExit):
        print("Stopping miner ...")
        pool.stop()
        m.stop()
        io_loop.stop()
        io_loop.close()

    return 0
