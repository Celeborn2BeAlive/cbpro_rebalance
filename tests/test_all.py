import unittest
import json
import subprocess
import logging
import pprint
import sys
import os

logging.getLogger().setLevel(logging.INFO)
VERBOSE = os.environ['VERBOSE'] == "1" if 'VERBOSE' in os.environ else False


class Tests20191215(unittest.TestCase):
    def setUp(self):
        self.config_file = 'test-data/freezed-config-2019-12-15.json'
        self.portfolio_file = 'test-data/portfolio-2019-12-15.json'
        self.bad_portfolio_file = 'test-data/bad-portfolio.json'
        self.allocations_file = 'test-data/allocations-2019-12-15.json'
        self.target_allocations_file = 'test-data/target-allocations-2019-12-15.json'
        self.target_allocations_file_no_EUR = 'test-data/target-allocations-no-EUR-2019-12-15.json'
        self.target_allocations_file_no_EUR_BTC_XRP = 'test-data/target-allocations-no-EUR-BTC-XRP-2019-12-15.json'

        self.base_args = [sys.executable, 'main.py', '--config', self.config_file,
                          '--exchange-type', 'freezed']

    def run_test(self, args, expected_result_file, should_fail=False):
        if VERBOSE:
            arg_line = ' '.join(args)
            logging.info(f'Running {arg_line}')
        cp = subprocess.run(args,
                            universal_newlines=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        if cp.returncode != 0 and not should_fail:
            logging.error(cp.stderr)
        if cp.returncode == 0 and should_fail:
            logging.error(cp.stderr)

        if should_fail:
            self.assertNotEqual(cp.returncode, 0)
        else:
            self.assertEqual(cp.returncode, 0)

        if expected_result_file:
            result = json.loads(cp.stdout)
            with open(expected_result_file, 'r') as f:
                expected_result = json.load(f)
            if result != expected_result:
                print("EXPECTED:", file=sys.stderr)
                pprint.pprint(expected_result, stream=sys.stderr)
                print("RETURNED:", file=sys.stderr)
                pprint.pprint(result, stream=sys.stderr)
            self.assertEqual(result, expected_result)

    def test_freeze(self):
        args = self.base_args + ['freeze']
        self.run_test(args, self.config_file)

    def test_portfolio(self):
        args = self.base_args + ['get-portfolio']
        self.run_test(args, self.portfolio_file)

    def test_allocations(self):
        args = self.base_args + ['get-allocations', self.portfolio_file]
        self.run_test(args, self.allocations_file)

    def test_target_allocations(self):
        args = self.base_args + \
            ['get-target-allocations', 'equidistrib',
                self.portfolio_file]
        self.run_test(args, self.target_allocations_file)

    def test_target_allocations_no_EUR(self):
        args = self.base_args + \
            ['get-target-allocations', 'equidistrib',
                self.portfolio_file, '--exclude', 'EUR']
        self.run_test(args, self.target_allocations_file_no_EUR)

    def test_target_allocations_no_EUR_BTC_XRP(self):
        args = self.base_args + \
            ['get-target-allocations', 'equidistrib',
                self.portfolio_file, '--exclude', 'EUR,BTC,XRP']
        self.run_test(args, self.target_allocations_file_no_EUR_BTC_XRP)

    def test_target_allocations_bad_portfolio(self):
        args = self.base_args + \
            ['get-target-allocations', 'equidistrib',
                self.bad_portfolio_file]
        self.run_test(args, None, should_fail=True)


class LiveTest(unittest.TestCase):
    def setUp(self):
        self.config_file = 'live-data/credentials.json'
        self.base_args = [sys.executable, 'main.py', '--config', self.config_file,
                          '--exchange-type', 'live']

    def test_portfolio(self):
        args = self.base_args + ['get-portfolio']
        self.run_test(args, self.portfolio_file)

    def run_test(self, args, callback=None):
        if VERBOSE:
            arg_line = ' '.join(args)
            logging.info(f'Running {arg_line}')
        cp = subprocess.run(args,
                            universal_newlines=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        if cp.returncode != 0:
            logging.error(cp.stderr)

        self.assertEqual(cp.returncode, 0)

        result = json.loads(cp.stdout)
        if VERBOSE:
            pprint.pprint(result, stream=sys.stderr)
        self.assertNotEqual(result, "")
        if callback:
            callback(result)

    # def test_freeze(self):
    #     args = self.base_args + ['freeze']
    #     self.run_test(args)

    def test_portfolio(self):
        args = self.base_args + ['get-portfolio']
        self.run_test(args)


if __name__ == '__main__':
    unittest.main()
