import json
import random

from tornado import gen
from tornado.httpclient import AsyncHTTPClient as HTTPClient
from tornado.httpclient import HTTPClientError

class Getwork(object):

    def __init__(self, endpoint, collector=None, result_queue=None):
        self.endpoint = endpoint
        self.http_client = HTTPClient(defaults=dict(request_timeout=2))
        self.should_stop = False
#       self.work_queue = work_queue
        self.result_queue = result_queue
        self.collector = collector
        self.current_work = dict()

    async def run(self):
        while True:
            await self.request()
            await gen.sleep(3)

    async def request(self):
        headers = {
            "Content-Type": "application/json",
        }
        body = {"jsonrpc":"2.0","method":"etrue_getWork","params":[],"id":0}
        try:
            response = await self.http_client.fetch(self.endpoint,
                                                    raise_error=False,
                                                    headers=headers,
                                                    method='POST',
                                                    body=json.dumps(body))
            if response.code == 200:
                self.getwork(json.loads(response.body).get('result', []))
            else:
                print('http response error %s' % response.code)
        except Exception as e:
            print('http error %s' % e)

    def getwork(self, result):
        """
        Getwork from rpc endpoint, including header, seed, and target.
        """
        work = {}
        work['header'] = result[0]
        work['fruit_target'] = 2 ** 128 // int(result[2], 0)
        work['target'] = 2 ** 128 // int(result[3], 0)
        work['nonce'] = self.gen_start_nonce()

        if self.current_work.get('header') != work['header']:
            self.current_work = work
            self.enqueue_work(work)

    def enqueue_work(self, work):
        self.collector.enqueue_work(work)

    def gen_start_nonce(self):
        return random.randrange(2**24) << 40
