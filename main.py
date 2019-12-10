from collections import defaultdict
import argparse
import logging
import os
import asyncio
import json
import math
from datetime import datetime
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

        time = datetime.now()
        time_str = time.strftime('%Y-%m-%d-%H:%M:%S')

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

        if args.action == 'get-portfolio':
            json_output({'time': time_str, 'portfolio': await get_portfolio(rest_client)}, args.out)
            return

        if args.action == 'get-allocations':
            with open(args.portfolio) as f:
                portfolio = json.load(f)['portfolio']
            prices = await get_prices_dict(rest_client, portfolio, QUOTE_CURRENCY, product_ids)
            prices_btc = await get_prices_dict(rest_client, portfolio, 'BTC', product_ids)

            allocations = compute_allocations(prices, portfolio)
            allocations_btc = compute_allocations(prices_btc, portfolio)

            json_output({
                'time': time_str,
                'allocations': allocations,
                'allocations_btc': allocations_btc
            }, args.out)

            return

        if args.action == 'get-target-allocations':
            with open(args.portfolio) as f:
                portfolio = json.load(f)

            if args.method == 'equidistrib':
                json_output(compute_equidistrib_allocations(
                    portfolio, set([QUOTE_CURRENCY])), args.out)
            else:
                logging.error("Only 'equidistrib' is supported for now")

            return

        if args.action == 'get-orders':
            with open(args.portfolio) as f:
                portfolio = json.load(f)['portfolio']
            with open(args.target_allocations) as f:
                target_allocations = json.load(f)

            output = {}

            output['portfolio'] = portfolio
            output['target_allocations'] = target_allocations

            SWAP_CURRENCY = 'BTC'
            output['swap_currency'] = SWAP_CURRENCY

            prices = await get_prices_dict(rest_client, portfolio, SWAP_CURRENCY, product_ids)
            allocations = compute_allocations(prices, portfolio)

            output['prices'] = prices
            output['allocations'] = allocations

            value_to_sell = compute_value_to_sell(
                target_allocations, allocations['values'], allocations['total'])
            amount_to_sell = compute_amount_to_sell(
                value_to_sell, allocations['prices'])

            output['value_to_sell'] = value_to_sell
            output['amount_to_sell'] = amount_to_sell

            output['check_sums'] = check_sums(value_to_sell)

            orders = []

            ruled_size_to_sell = {}
            # Start by computing sell orders, to accumulate BTCs
            for currency in amount_to_sell:
                if currency == SWAP_CURRENCY:
                    continue
                if amount_to_sell[currency] <= 0.0:
                    continue
                if f'{currency}-{SWAP_CURRENCY}' in product_ids:
                    # we need to sell on that market
                    product_id = f'{currency}-{SWAP_CURRENCY}'
                    side = 'sell'
                    size = amount_to_sell[currency]
                    wanted_price = allocations['prices'][currency]
                elif f'{SWAP_CURRENCY}-{currency}' in product_ids:
                    # we need to buy on that market
                    product_id = f'{SWAP_CURRENCY}-{currency}'
                    side = 'buy'
                    size = amount_to_sell[currency] * prices[currency]
                    wanted_price = 1 / allocations['prices'][currency]
                else:
                    logging.error(
                        f'{SWAP_CURRENCY}-{currency} market does not exist')

                product = product_ids[product_id]
                size = round_to_increment(size, product['base_increment'])

                min_size = float(product['base_min_size'])
                max_size = float(product['base_max_size'])
                if size < min_size:
                    size = 0
                elif size > max_size:
                    size = max_size

                ruled_size_to_sell[currency] = size

                if ruled_size_to_sell[currency] == 0.0:
                    continue

                order = {
                    'product_id': product_id,
                    'side': side,
                    'size': size,
                    'currency': currency,
                    'wanted_price': wanted_price
                }

                orders.append(order)

            for currency in amount_to_sell:
                if currency == SWAP_CURRENCY:
                    continue
                if amount_to_sell[currency] >= 0.0:
                    continue
                amount_to_buy = -amount_to_sell[currency]
                if f'{currency}-{SWAP_CURRENCY}' in product_ids:
                    # we need to sell on that market
                    product_id = f'{currency}-{SWAP_CURRENCY}'
                    side = 'buy'
                    size = amount_to_buy
                    wanted_price = allocations['prices'][currency]
                elif f'{SWAP_CURRENCY}-{currency}' in product_ids:
                    # we need to buy on that market
                    product_id = f'{SWAP_CURRENCY}-{currency}'
                    side = 'sell'
                    size = amount_to_buy * prices[currency]
                    wanted_price = 1 / allocations['prices'][currency]
                else:
                    logging.error(
                        f'{SWAP_CURRENCY}-{currency} market does not exist')

                product = product_ids[product_id]
                size = round_to_increment(size, product['base_increment'])

                min_size = float(product['base_min_size'])
                max_size = float(product['base_max_size'])
                if size < min_size:
                    size = 0
                elif size > max_size:
                    size = max_size

                ruled_size_to_sell[currency] = size

                if ruled_size_to_sell[currency] == 0.0:
                    continue

                order = {
                    'product_id': product_id,
                    'side': side,
                    'size': size,
                    'currency': currency,
                    'wanted_price': wanted_price
                }

                orders.append(order)

            output['ruled_size_to_sell'] = ruled_size_to_sell

            # todo: this should be proportional to increment / current_price, for example EOS-BTC was at 0.277% of increment at test time
            PERCENTAGE_THRESHOLD = 0.5
            output['percentage_treshold'] = PERCENTAGE_THRESHOLD

            output['orders'] = orders

            if args.do_it:
                logging.info('Start trading')
            else:
                logging.info('Only compute limit orders')

            limit_orders = []
            for order in orders:
                product_id = order['product_id']
                side = order['side']
                size = order['size']
                wanted_price = order['wanted_price']

                increment_string = product_ids[product_id]['quote_increment']
                quote_increment = float(increment_string)

                ticker = await rest_client.ticker(product_id)

                base_order_price = float(
                    ticker['bid']) if side == 'sell' else float(ticker['ask'])

                corrected_order_price = base_order_price
                increment_multiplier = 1

                while corrected_order_price == base_order_price:
                    increment = increment_multiplier * quote_increment
                    increment_multiplier += 1
                    corrected_order_price = base_order_price + \
                        increment if side == 'sell' else base_order_price - increment
                    corrected_order_price = round_to_increment(
                        corrected_order_price, increment_string)

                order_price = corrected_order_price

                percentage_diff = 100 * \
                    (order_price - wanted_price) / wanted_price

                try_it = True

                if side == 'sell' and percentage_diff < -PERCENTAGE_THRESHOLD:
                    try_it = False
                if side == 'buy' and percentage_diff > PERCENTAGE_THRESHOLD:
                    try_it = False

                limit_order = {
                    'base_order': order,
                    'ticker': ticker,
                    'order_price': order_price,
                    'base_order_price': base_order_price,
                    'corrected_order_price': corrected_order_price,
                    'increment_multiplier': increment_multiplier,
                    'percentage_diff': percentage_diff,
                    'try_it': try_it
                }

                if args.do_it:
                    if try_it:
                        # note: can throw with copra.rest.client.APIRequestError: Post only mode [400]
                        # if price has changed fast
                        try:
                            response = await rest_client.limit_order(
                                side, product_id, order_price, size,
                                time_in_force='GTT', cancel_after='hour',
                                post_only=True)
                            limit_order['submit_response'] = response
                            logging.info(response)
                        except Exception as e:
                            logging.error(e)
                            logging.info(order_price)
                            logging.info(quote_increment)

                limit_orders.append(limit_order)

            if args.do_it:
                for order in limit_orders:
                    if order['try_it']:
                        try:
                            id = order['submit_response']['id']
                            response = await rest_client.get_order(id)
                            order['query_response'] = response
                            if response['status'] == 'done':
                                logging.info(f'{id} executed')
                            else:
                                logging.info(f'{id} pending, cancelling')
                                response = await rest_client.cancel(id)
                                order['cancel_response'] = response
                        except Exception as e:
                            logging.error(e)

            output['limit_orders'] = limit_orders

            json_output(output)

            return

        args_parser.print_help()


def round_to_increment(amount, increment_string):
    decimals = increment_string.split('.')[1]
    if '1' in decimals:
        decimals_count = 1 + len(decimals.split('1')[0])
    else:
        decimals_count = 0
    return int(amount * 10**decimals_count) / 10**decimals_count


def check_sums(value_to_sell):
    sum_positive = 0
    sum_negative = 0
    for c in value_to_sell:
        if value_to_sell[c] > 0.0:
            sum_negative += value_to_sell[c]
        else:
            sum_positive += value_to_sell[c]
    return sum_positive + sum_negative


def compute_value_to_sell(target_allocations, current_values, current_total):
    value_to_sell = {}
    for c in target_allocations:
        target_value = target_allocations[c] * current_total
        current_value = current_values[c] if c in current_values else 0.0
        value_to_sell[c] = current_value - target_value
    return value_to_sell


def compute_amount_to_sell(value_to_sell, current_prices):
    amount_to_sell = {}
    for c in value_to_sell:
        amount_to_sell[c] = value_to_sell[c] / current_prices[c]
    return amount_to_sell


def compute_equidistrib_allocations(portfolio, exclude_symbols=set()):
    target_allocations = {}
    total = 0.0
    for currency in portfolio:
        if currency in exclude_symbols:
            target_allocations[currency] = 0.0
        else:
            target_allocations[currency] = 1.0
            total += 1.0
    for currency in portfolio:
        target_allocations[currency] /= total
    return target_allocations


def compute_allocations(prices, portfolio):
    total = 0.0
    values = {}
    float_prices = {}
    float_amount = {}
    for base_currency in prices:
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
    return {
        'total': total,
        'prices': float_prices,
        'portfolio': float_amount,
        'values': values,
        'allocations': allocations
    }


async def get_portfolio(rest_client):
    accounts = await rest_client.accounts()
    return {item['currency']: item['balance']
            for item in accounts if float(item['balance']) != 0}


async def get_prices_dict(rest_client, portfolio, quote_currency, product_ids):
    prices = {}
    for base_currency in portfolio:
        await asyncio.sleep(0.2)
        if base_currency == quote_currency:
            prices[base_currency] = 1.0
            continue
        wanted_product_id = f'{base_currency}-{quote_currency}'
        if wanted_product_id in product_ids:
            prices[base_currency] = float((await rest_client.ticker(wanted_product_id))['price'])
            continue
        reverse_product_id = f'{quote_currency}-{base_currency}'
        if reverse_product_id in product_ids:
            price_of_quote_in_base = float((await rest_client.ticker(reverse_product_id))['price'])
            price_of_base_in_quote = 1.0 / price_of_quote_in_base
            prices[base_currency] = price_of_base_in_quote
            continue
        if base_currency == 'BTC':
            logging.error(
                f'No market to convert {base_currency} to {quote_currency}')
            exit(-1)

        btc_product_id = f'{base_currency}-BTC'
        if btc_product_id in product_ids:
            btc_price = float((await rest_client.ticker(btc_product_id))['price'])
        else:
            btc_product_id = f'BTC-{base_currency}'
            btc_price = 1.0 / float((await rest_client.ticker(btc_product_id))['price'])

        base_product_id = f'BTC-{quote_currency}'
        base_price = float((await rest_client.ticker(base_product_id))['price'])

        prices[base_currency] = base_price * btc_price
    return prices


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
        'portfolio', help='Portfolio stored in json file. Can be obtained with get-portfolio command.')
    command_parser.add_argument(
        'target_allocations', help='Target allocation stored in json file. Can be obtained with get-target-allocations.'
    )
    command_parser.add_argument(
        '--do-it', action='store_true', help='Performs rebalance.'
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
