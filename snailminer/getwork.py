import json
import logging
import random

from tornado import gen
from tornado.httpclient import AsyncHTTPClient as HTTPClient
from tornado.httpclient import HTTPClientError

LOG = logging.getLogger(__name__)


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

    async def commit_result(self):
        if not self.result_queue.empty():
            result = self.result_queue.get()
            headers = {
                "Content-Type": "application/json",
            }
            body = {
                "jsonrpc": "2.0",
                "method": "etrue_submitWork",
                "params": [hex(result['found_nonce']), result['header'], result['digest']],
                "id": 0,
            }
            try:
                response = await self.http_client.fetch(self.endpoint,
                                                        raise_error=False,
                                                        headers=headers,
                                                        method='POST',
                                                        body=json.dumps(body))
                if response.code == 200:
                    resp = json.loads(response.body).get('result')
                    LOG.info('commit solution %s: ret=%s', result['header'], resp)
                else:
                    LOG.error('http response error %s' % response.code)
            except Exception as e:
                LOG.error('http error %s' % e)

    async def request(self):
        headers = {
            "Content-Type": "application/json",
        }
        body = {
            "jsonrpc": "2.0",
            "method": "etrue_getWork",
            "params": [],
            "id": 0,
        }
        try:
            response = await self.http_client.fetch(self.endpoint,
                                                    raise_error=False,
                                                    headers=headers,
                                                    method='POST',
                                                    body=json.dumps(body))
            if response.code == 200:
                resp = json.loads(response.body)
                self.getwork(resp.get('result', []))
            else:
                LOG.error('http response error %s' % response.code)
        except Exception as e:
            LOG.error('http error %s' % e)

    def getwork(self, result):
        """
        Getwork from rpc endpoint, including header, seed, and target.
        """
        if len(result) != 4:
            LOG.debug('getwork result empty')
        work = {}
        work['header'] = result[0]
        work['fruit_target'] = result[2]
        work['target'] = result[3]
        work['nonce'] = self.gen_start_nonce()

        if self.current_work.get('header') != work['header']:
            self.current_work = work
            self.enqueue_work(work)

    def enqueue_work(self, work):
        self.collector.enqueue_work(work)

    def gen_start_nonce(self):
        return random.randrange(2**24) << 40
