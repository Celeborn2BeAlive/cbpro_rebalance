import unittest
import json
import subprocess
import logging
import pprint


class Tests20191215(unittest.TestCase):
    def setUp(self):
        self.config_file = 'test-data/freezed-config-2019-12-15.json'
        self.portfolio_file = 'test-data/portfolio-2019-12-15.json'
        self.allocations_file = 'test-data/allocations-2019-12-15.json'

        self.base_args = ['python', 'main.py', '--config', self.config_file,
                          '--exchange-type', 'freezed']

    def run_test(self, args, expected_result_file, use_logging=False):
        if use_logging:
            arg_line = ' '.join(args)
            logging.info(f'Running {arg_line}')
        cp = subprocess.run(args,
                            universal_newlines=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        if cp.returncode != 0 and use_logging:
            logging.error(cp.stderr)

        self.assertEqual(cp.returncode, 0)

        json_string = cp.stdout
        result = json.loads(json_string)
        with open(expected_result_file, 'r') as f:
            expected_result = json.load(f)
        if use_logging:
            pprint.pprint(expected_result)
            pprint.pprint(result)
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


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    unittest.main()
