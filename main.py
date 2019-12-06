from collections import defaultdict
import argparse
import logging
import os
import asyncio
import json
import math
from collections import defaultdict

from copra.rest import Client as RestClient


async def main():
    args, args_parser = parse_cli_args()
    init_logging(args)

    with open(args.creds, 'r') as f:
        creds = json.load(f)

    async with RestClient(asyncio.get_event_loop(), auth=True, key=creds['apiKey'],
                          secret=creds['apiSecret'], passphrase=creds['passPhrase']) as rest_client:
        products = await rest_client.products()
        product_ids = {p['id']: p for p in products}

        QUOTE_CURRENCY = 'EUR'

        def fun_with_graph():
            BANNED_CURRENCIES = ['USD']

            products_graph = Digraph()
            for p in product_ids:
                base, quote = p.split('-')
                if base in BANNED_CURRENCIES or quote in BANNED_CURRENCIES:
                    continue
                products_graph.addNode(base, quote)
                products_graph.addEdge(base, quote, 1.0)
                products_graph.addEdge(quote, base, 1.0)

            print(products_graph.min_path('EUR', 'USDC'))

        async def get_prices_dict(rest_client, portfolio):
            prices = {}
            for base_currency in portfolio:
                await asyncio.sleep(0.2)
                if base_currency == QUOTE_CURRENCY:
                    prices[base_currency] = '1.0'
                    continue
                wanted_product_id = f'{base_currency}-{QUOTE_CURRENCY}'
                if wanted_product_id in product_ids:
                    prices[base_currency] = (await rest_client.ticker(f'{base_currency}-{QUOTE_CURRENCY}'))['price']
                    continue
                btc_product_id = f'{base_currency}-BTC'
                if btc_product_id in product_ids:
                    btc_price = float((await rest_client.ticker(btc_product_id))['price'])
                else:
                    btc_product_id = f'BTC-{base_currency}'
                    btc_price = 1.0 / float((await rest_client.ticker(btc_product_id))['price'])

                base_product_id = f'BTC-{QUOTE_CURRENCY}'
                base_price = float((await rest_client.ticker(base_product_id))['price'])

                prices[base_currency] = f'{(base_price * btc_price):.2}'
            return prices

        if args.action == 'get-portfolio':
            accounts = await rest_client.accounts()
            portfolio = {item['currency']: item['balance']
                         for item in accounts if float(item['balance']) != 0}
            json_output(portfolio, args.out)

            return

        if args.action == 'get-allocations':
            with open(args.portfolio) as f:
                portfolio = json.load(f)
            prices = await get_prices_dict(rest_client, portfolio)
            total = 0.0
            values = {}
            float_prices = {}
            float_amount = {}
            for base_currency in portfolio:
                price = float(prices[base_currency])
                float_prices[base_currency] = price
                amount = float(portfolio[base_currency])
                float_amount[base_currency] = amount
                value = price * amount
                values[base_currency] = value
                total += value
            allocations = {}
            for base_currency in portfolio:
                allocations[base_currency] = values[base_currency] / total

            json_output({
                'total': total,
                'prices': float_prices,
                'portfolio': float_amount,
                'values': values,
                'allocations': allocations
            }, args.out)

            return

        if args.action == 'get-target-allocations':
            with open(args.portfolio) as f:
                portfolio = json.load(f)

            if args.method == 'equidistrib':
                target_allocations = {}
                total = 0.0
                for currency in portfolio:
                    if currency == QUOTE_CURRENCY:
                        target_allocations[currency] = 0.0
                    else:
                        target_allocations[currency] = 1.0
                        total += 1.0
                for currency in portfolio:
                    target_allocations[currency] /= total
                json_output(target_allocations, args.out)
            else:
                logging.error("Only 'equidistrib' is supported for now")

            return

        if args.action == 'get-orders':
            with open(args.allocations) as f:
                allocations = json.load(f)
            with open(args.target_allocations) as f:
                target_allocations = json.load(f)

            target_values = {}
            value_to_sell = {}
            amount_to_sell = {}
            for c in target_allocations:
                target_values[c] = target_allocations[c] * allocations['total']
                current_value = allocations['values'][c] if c in allocations['values'] else 0.0
                value_to_sell[c] = current_value - target_values[c]
                amount_to_sell[c] = value_to_sell[c] / allocations['prices'][c]

            json_output(value_to_sell)
            json_output(amount_to_sell)

            sum_positive = 0
            sum_negative = 0
            for c in target_allocations:
                if value_to_sell[c] > 0.0:
                    sum_negative += value_to_sell[c]
                else:
                    sum_positive += value_to_sell[c]
            print(sum_positive, sum_negative)

            TARGET_CURRENCY = 'BTC'

            orders = []

            ruled_amount_to_sell = {}
            # Start by computing sell orders, to accumulate BTCs
            for currency in amount_to_sell:
                if currency == TARGET_CURRENCY:
                    continue
                if amount_to_sell[currency] <= 0.0:
                    continue
                if f'{currency}-{TARGET_CURRENCY}' in product_ids:
                    # we need to sell on that market
                    product_id = f'{currency}-{TARGET_CURRENCY}'
                    product = product_ids[product_id]
                    side = 'sell'
                    key = 'base_increment'
                elif f'{TARGET_CURRENCY}-{currency}' in product_ids:
                    # we need to buy on that market
                    product_id = f'{TARGET_CURRENCY}-{currency}'
                    product = product_ids[product_id]
                    side = 'buy'
                    key = 'quote_increment'
                else:
                    logging.error(
                        f'{TARGET_CURRENCY}-{currency} market does not exist')
                decimals = product[key].split('.')[1]
                if '1' in decimals:
                    decimals_count = 1 + len(decimals.split('1')[0])
                else:
                    decimals_count = 0
                amount = int(amount_to_sell[currency] *
                             10**decimals_count) / 10**decimals_count
                ruled_amount_to_sell[currency] = amount
                order = {
                    'type': 'market',
                    'side': side,
                    'product_id': product_id,
                    'amount': amount,
                    'value': amount * allocations['prices'][c]
                }
                if side == 'sell':
                    order['size'] = str(amount)
                else:
                    order['funds'] = str(amount)
                orders.append(order)

            for currency in amount_to_sell:
                if currency == TARGET_CURRENCY:
                    continue
                if amount_to_sell[currency] >= 0.0:
                    continue
                amount_to_buy = -amount_to_sell[currency]
                if f'{currency}-{TARGET_CURRENCY}' in product_ids:
                    # we need to sell on that market
                    product_id = f'{currency}-{TARGET_CURRENCY}'
                    product = product_ids[product_id]
                    side = 'buy'
                    key = 'base_increment'
                elif f'{TARGET_CURRENCY}-{currency}' in product_ids:
                    # we need to buy on that market
                    product_id = f'{TARGET_CURRENCY}-{currency}'
                    product = product_ids[product_id]
                    side = 'sell'
                    key = 'quote_increment'
                else:
                    logging.error(
                        f'{TARGET_CURRENCY}-{currency} market does not exist')
                decimals = product[key].split('.')[1]
                if '1' in decimals:
                    decimals_count = 1 + len(decimals.split('1')[0])
                else:
                    decimals_count = 0
                amount = int(amount_to_buy *
                             10**decimals_count) / 10**decimals_count
                ruled_amount_to_sell[currency] = -amount
                order = {
                    'type': 'market',
                    'side': side,
                    'product_id': product_id,
                    'amount': amount,
                    'value': amount * allocations['prices'][c]
                }
                if side == 'buy':
                    order['size'] = str(amount)
                else:
                    order['funds'] = str(amount)
                orders.append(order)

            json_output(ruled_amount_to_sell)
            json_output(orders)

            # encoutered issues:
            # - some market throw an exception copra.rest.client.APIRequestError: Limit only mode [400] (EOS-BTC)
            # - bounds need to be respected
            # - todo: should use limit orders (and monitor them)
            # - need to update portfolio according to what has been bought

            if False:
                for i in range(len(orders), len(orders)):
                    if orders[i]['amount'] > 0.0:
                        print(i, orders[i])
                        if 'size' in orders[i]:
                            req_size = float(orders[i]['size'])
                            min_size = float(
                                product_ids[orders[i]['product_id']]['base_min_size'])
                            max_size = float(
                                product_ids[orders[i]['product_id']]['base_max_size'])
                            if req_size < min_size or req_size > max_size:
                                logging.error(
                                    'Requested size is not in bounds')
                                continue

                            r = await rest_client.market_order(orders[i]['side'], orders[i]['product_id'],
                                                               size=orders[i]['size'])
                        elif 'funds' in orders[i]:
                            req_funds = float(orders[i]['funds'])
                            min_funds = float(
                                product_ids[orders[i]['product_id']]['min_market_funds'])
                            max_funds = float(
                                product_ids[orders[i]['product_id']]['max_market_funds'])
                            if req_funds < min_funds or req_funds > max_funds:
                                logging.error(
                                    'Requested funds is not in bounds')
                                continue
                            r = await rest_client.market_order(orders[i]['side'], orders[i]['product_id'],
                                                               funds=orders[i]['funds'])
                        else:
                            logging.error('No size or funds in order')
                            continue
                        print(r)
                        await asyncio.sleep(0.5)
                        while True:
                            r2 = await rest_client.get_order(r['id'])
                            if r2['status'] == 'done':
                                break
                            await asyncio.sleep(0.2)
                        print(f'order {i} executed')

            return

        args_parser.print_help()


def json_output(data: dict, outfile=None):
    if outfile:
        with open(outfile, 'w') as f:
            json.dump(data, f, indent=4)
    else:
        print(json.dumps(data, indent=4))


def init_logging(args):
    if args.log_file:
        if os.path.splitext(args.log_file)[1] == '.html':
            logging.basicConfig(filename=args.log_file, filemode='w',
                                format='%(asctime)s - %(levelname)s - %(message)s<br>', level=logging.INFO)
        else:
            logging.basicConfig(filename=args.log_file, filemode='w',
                                format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
        logging.info(f'Logging to file {args.log_file}')
    else:
        logging.basicConfig(
            format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
        logging.info('Logging to error output')


def parse_cli_args():
    def parse_string_list(string):
        return string.split(',')

    parser = argparse.ArgumentParser(description='Desc')
    parser.add_argument('-l', '--log-file', help='Path to log file.')
    parser.add_argument(
        '--creds', required=True, help='Path to json file containing credentials (apiKey, apiSecret, passPhrase)')

    commands = parser.add_subparsers(
        title='commands', dest='action')

    command_parser = commands.add_parser('get-portfolio')
    command_parser.add_argument('-o', '--out', help='Output json file')

    command_parser = commands.add_parser('get-allocations')
    command_parser.add_argument(
        'portfolio', help='Portfolio stored in json file. Can be obtained with get-portfolio command.')
    command_parser.add_argument('-o', '--out', help='Output json file')

    command_parser = commands.add_parser('get-target-allocations')
    command_parser.add_argument(
        'method', help='Method to use to compute allocation.')
    command_parser.add_argument(
        'portfolio', help='Portfolio stored in json file. Can be obtained with get-portfolio command.')
    command_parser.add_argument('-o', '--out', help='Output json file')

    command_parser = commands.add_parser('get-orders')
    command_parser.add_argument(
        'allocations', help='Allocations stored in json file. Can be obtained with get-allocations command.')
    command_parser.add_argument(
        'target_allocations', help='Target allocation stored in json file. Can be obtained with get-target-allocations.'
    )
    command_parser.add_argument('-o', '--out', help='Output json file')

    return parser.parse_args(), parser


"""An efficient algorithm to find shortest paths between nodes in a graph."""


class Digraph(object):
    def __init__(self, nodes=[]):
        self.nodes = set()
        self.neighbours = defaultdict(set)
        self.dist = {}

    def addNode(self, *nodes):
        [self.nodes.add(n) for n in nodes]

    def addEdge(self, frm, to, d=1e309):
        self.addNode(frm, to)
        self.neighbours[frm].add(to)
        self.dist[frm, to] = d

    def dijkstra(self, start, maxD=1e309):
        """Returns a map of nodes to distance from start and a map of nodes to
        the neighbouring node that is closest to start."""
        # total distance from origin
        tdist = defaultdict(lambda: 1e309)
        tdist[start] = 0
        # neighbour that is nearest to the origin
        preceding_node = {}
        unvisited = self.nodes

        while unvisited:
            current = unvisited.intersection(tdist.keys())
            if not current:
                break
            min_node = min(current, key=tdist.get)
            unvisited.remove(min_node)

            for neighbour in self.neighbours[min_node]:
                d = tdist[min_node] + self.dist[min_node, neighbour]
                if tdist[neighbour] > d and maxD >= d:
                    tdist[neighbour] = d
                    preceding_node[neighbour] = min_node

        return tdist, preceding_node

    def min_path(self, start, end, maxD=1e309):
        """Returns the minimum distance and path from start to end."""
        tdist, preceding_node = self.dijkstra(start, maxD)
        dist = tdist[end]
        backpath = [end]
        try:
            while end != start:
                end = preceding_node[end]
                backpath.append(end)
            path = list(reversed(backpath))
        except KeyError:
            path = None

        return dist, path

    def dist_to(self, *args): return self.min_path(*args)[0]
    def path_to(self, *args): return self.min_path(*args)[1]


if __name__ == "__main__":
    asyncio.run(main())
