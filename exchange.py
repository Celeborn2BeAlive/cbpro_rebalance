from copra.rest import Client as RestClient
from copra.rest.client import APIRequestError
import json
import asyncio
import logging
from datetime import datetime
import time

COINBASE_MAX_REQUEST_TRIALS_COUNT = 100
COINBASE_TIME_TO_SLEEP = 0.2


class CoinbaseExchange():
    def __init__(self, config: dict):
        self.rest_client = \
            RestClient(asyncio.get_event_loop(), auth=True, key=config['apiKey'],
                       secret=config['apiSecret'], passphrase=config['passPhrase'])
        self.previous_request_time = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.rest_client.close()

    async def client_request(self, function, *args):
        current_time = time.time()
        elapsed_time = current_time - self.previous_request_time
        if elapsed_time < COINBASE_TIME_TO_SLEEP:
            await asyncio.sleep(COINBASE_TIME_TO_SLEEP)
        trials_count = 0
        while trials_count < COINBASE_MAX_REQUEST_TRIALS_COUNT:
            trials_count += 1
            try:
                result = await function(*args)
                self.previous_request_time = current_time
                return result
            except APIRequestError as e:
                # Public rate limit exceeded [429]
                if e.response.status == 429:
                    logging.warning(
                        'Public rate limit exceeded, waiting a bit...')
                    await asyncio.sleep(COINBASE_TIME_TO_SLEEP)
                else:
                    raise e

    async def products(self):
        return sorted(await self.client_request(self.rest_client.products), key=lambda e: e['id'])

    async def ticker(self, product_id):
        return await self.client_request(self.rest_client.ticker, product_id)

    async def limit_order(self, side, product_id, order_price, size):
        try:
            return await self.client_request(self.rest_client.limit_order,
                                             side, product_id, order_price, size,
                                             time_in_force='GTT', cancel_after='hour',
                                             post_only=True)
        except APIRequestError as e:
            if e.response.status == 404:
                # Post only mode [400]: order has been cancelled, we return None
                return None
            raise e  # Unknown error, can do nothing from here

    async def get_order(self, order_id):
        try:
            return await self.client_request(self.rest_client.get_order, order_id)
        except APIRequestError as e:
            if e.response.status == 404:
                # NotFound [404]: order has been cancelled, we return None
                return None
            raise e  # Unknown error, can do nothing from here

    async def accounts(self):
        return sorted(await self.client_request(self.rest_client.accounts), key=lambda e: e['id'])

    async def fees(self):
        return await self.client_request(self.rest_client.fees)

    async def currencies(self):
        return await self.client_request(self.rest_client.currencies)

    async def time(self):
        return datetime.now().strftime('%Y-%m-%dT%H:%M:%S')


class FreezedStateExchange():
    def __init__(self, config: dict):
        self.config = config
        self.next_order_id = 0
        self.pending_orders = {}
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def products(self):
        return self.config['products']

    async def ticker(self, product_id):
        return self.config['tickers'][product_id]

    async def limit_order(self, side, product_id, order_price, size):
        created_time = datetime.now()
        time_fmt = '%Y-%m-%dT%H:%M:%S'
        response = {
            "id": f"{self.next_order_id}",
            "price": f"{order_price}",
            "size": f"{size}",
            "product_id": f"{product_id}",
            "side": f"{side}",
            "stp": "dc",
            "type": "limit",
            "time_in_force": "GTT",
            "expire_time": (created_time + timedelta(hours=1)).strftime(time_fmt),
            "post_only": True,
            "created_at": created_time.strftime(time_fmt),
            "fill_fees": "0",
            "filled_size": "0",
            "executed_value": "0",
            "status": "pending",
            "settled": False
        }
        self.next_order_id += 1
        self.pending_orders[response['id']] = response

        return response

    async def get_order(self, order_id):
        if order_id in self.pending_orders:
            order = self.pending_orders[order_id]
            del self.pending_orders[order_id]
            order['fill_size'] = order['size']
            executed_value = float(order['size']) * float(order['price'])
            order['executed_value'] = str(executed_value)
            maker_fee_rate = float(self.config['fees']['maker_fee_rate'])
            order['fill_fees'] = str(maker_fee_rate * float(executed_value))
            order['status'] = 'done'
            order['settled'] = True
            return order
        return None

    async def accounts(self):
        return self.config['accounts']

    async def fees(self):
        return self.config['fees']

    async def currencies(self):
        return self.config['currencies']

    async def time(self):
        return self.config['time']


def make_exchange(args):
    with open(args.config, 'r') as f:
        config = json.load(f)
    if args.exchange_type == 'live':
        return CoinbaseExchange(config)
    elif args.exchange_type == 'freezed':
        return FreezedStateExchange(config)
    return None
