from queue import Queue
from threading import Thread
from time import time


class Miner(object):
	def __init__(self, device_index, options):
		self.device_index = device_index
		self.work_queue = Queue()
		self.should_stop = False

	def start(self):
		self.should_stop = False
		Thread(target=self.mining_thread).start()
		self.start_time = time()

	def stop(self, message = None):
		if message: print('\n%s' % message)
		self.should_stop = True
