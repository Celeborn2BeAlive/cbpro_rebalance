import unittest
import json
import pprint
import asyncio
from main import make_exchange, get_freezed_state_exchange_config_cmd, get_portfolio_cmd

# Todo: instead of testing functions from main.py, we should directly test the output of the program
# Ideally tests should be specified in a json file


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
            result = await get_freezed_state_exchange_config_cmd(self.exchange)
            with open(self.config_file, 'r') as f:
                expected_result = json.load(f)
            self.assertEqual(result, expected_result)
        self.loop.run_until_complete(go())

    def test_portfolio(self):
        async def go():
            result = await get_portfolio_cmd(self.exchange)
            with open(self.portfolio_file, 'r') as f:
                expected_result = json.load(f)
            self.assertEqual(result, expected_result)
        self.loop.run_until_complete(go())


if __name__ == '__main__':
    unittest.main()
