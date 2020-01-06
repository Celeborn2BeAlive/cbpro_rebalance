import unittest
import json
import pprint
import asyncio
from main import make_exchange, get_freezed_state_exchange_config


class Args:
    def __init__(self, config_file):
        self.exchange_type = 'freezed'
        self.config = config_file


class TestConfig1(unittest.TestCase):
    def setUp(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(None)

        self.config_file = 'test-data/freezed-config-2019-12-15.json'
        self.portfolio_file = 'test-data/portfolio-2019-12-15.json'
        self.exchange = make_exchange(Args(self.config_file))

    def test_freezed_config(self):
        async def go():
            freezed_config = await get_freezed_state_exchange_config(self.exchange)
            with open(self.config_file, 'r') as f:
                file_config = json.load(f)
            self.assertEqual(freezed_config, file_config)
        self.loop.run_until_complete(go())

    def test_portfolio(self):
        pass


if __name__ == '__main__':
    unittest.main()
