import json

from tornado.httpclient import AsyncHTTPClient as HTTPClient
from tornado.httpclient import HTTPClientError

class Getwork(object):

    def __init__(self, endpoint):
        self.endpoint = endpoint
        self.http_client = HTTPClient()

    def process(self):
        pass

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
        print('getwork %s' % result)
