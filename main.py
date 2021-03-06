import argparse
import logging
import os
import asyncio
import json
import math
from datetime import datetime, timedelta

from exchange import make_exchange

# QUESTIONS THE SYSTEM NEED TO ANSWER
# - What is the percentage of my portfolio in stable coins ? in EUR ? in stable + EUR ?
# - What is the value of my portfolio at a given date ?
#   - need the history
# - Display evolution of allocations as curve, vs curve of each coin

# - Display open orders
# - Stop active watching -> send order and store states, one hour later rerun and check order states
# - Display size in EUR on orders, + fees

# - Need a simple watch system: exchange + currency + price + direction => mail + sms
# - Or each day a report of the evolution of the portfolio ?

# WHAT NEEDS TO BE FIXED BECAUSE IT IS HARD TO DEAL WITH !!
# - If error, orders is not written
# - I need a true log file as well as the json
# - It is doing to much things -> we should just output orders in a file
# - Then this file is read and executed
# - Print link to market and order pages on coinbase pro

# TODO
# - * watch and repost orders when they are cancelled
# - * new portfolio computation based on executed orders
# - * loop on previous portfolio to implement hourly rebalance with threshold based rebalancing
# - * Compute portfolio values in fiat currency (each line and the total)
# - * Cancel all orders if errors
# - * more logging during trading (about wanted price vs corrected price)
# - * buy BTC orders first (or sell for BTC), then buy others orders (it not enough BTC in wallet to handle rebalance)
# - * test if product is "limit_only": true; if not use market order

# - handle buying of new assets (right now we can only buy/sell assets from the portfolio)
# - better computation of threshold on percentage diff (should depend on increment and current price)
# - better tradeof between buying at bid/ask price vs market order, depending on increment and current price ratio
# - Unit testing
# - Exchange interface + BacktestExchange
# - Trader class
# - Markowitz rebalancing
# - Binance implementation of Exchange interface

# Example of response for limit_order():
# {
#     "id": "f15e831d-192f-4f6c-b9ec-21456b3aba9e",
#     "price": "0.000359",
#     "size": "0.1",
#     "product_id": "EOS-BTC",
#     "side": "buy",
#     "stp": "dc",
#     "type": "limit",
#     "time_in_force": "GTT",
#     "expire_time": "2019-12-11T01:55:38.276Z",
#     "post_only": true,
#     "created_at": "2019-12-10T01:55:38.278001Z",
#     "fill_fees": "0",
#     "filled_size": "0",
#     "executed_value": "0",
#     "status": "pending",
#     "settled": false
# }

# Examples of response for get_order():
# {
#     "id": "f15e831d-192f-4f6c-b9ec-21456b3aba9e",
#     "price": "0.00035900",
#     "size": "0.10000000",
#     "product_id": "EOS-BTC",
#     "profile_id": "b8d559a1-9dd4-4cd3-9204-2a2965f3894f",
#     "side": "buy",
#     "type": "limit",
#     "time_in_force": "GTT",
#     "expire_time": "2019-12-11T01:55:38.276",
#     "post_only": true,
#     "created_at": "2019-12-10T01:55:38.278001Z",
#     "fill_fees": "0.0000000000000000",
#     "filled_size": "0.00000000",
#     "executed_value": "0.0000000000000000",
#     "status": "open",
#     "settled": false
# },
# {
#     "id": "f15e831d-192f-4f6c-b9ec-21456b3aba9e",
#     "price": "0.00035900",
#     "size": "0.10000000",
#     "product_id": "EOS-BTC",
#     "profile_id": "b8d559a1-9dd4-4cd3-9204-2a2965f3894f",
#     "side": "buy",
#     "type": "limit",
#     "time_in_force": "GTT",
#     "expire_time": "2019-12-11T01:55:38.276",
#     "post_only": true,
#     "created_at": "2019-12-10T01:55:38.278001Z",
#     "done_at": "2019-12-10T02:30:12.665Z",
#     "done_reason": "filled",
#     "fill_fees": "0.0000001795000000",
#     "filled_size": "0.10000000",
#     "executed_value": "0.0000359000000000",
#     "status": "done",
#     "settled": true
# }
#
# Note about filled_size: according to my tests, it increment until reaching the size of the order

MAX_LIMIT_ORDER_TRIAL_COUNT = 5


async def get_freezed_state_exchange_config_cmd(exchange):
    output_dict = {
        'time': await exchange.time(),
        'products': await exchange.products(),
        'accounts': await exchange.accounts(),
        'currencies': await exchange.currencies(),
        'fees': await exchange.fees(),
        'tickers': {}
    }
    product_ids = {p['id']: p for p in output_dict['products']}
    for p in product_ids:
        output_dict['tickers'][p] = await exchange.ticker(p)
    return output_dict


async def get_portfolio_cmd(exchange, args):
    return {'time': await exchange.time(), 'portfolio': await get_portfolio(exchange, args.exclude)}


async def get_allocations_cmd(exchange, args):
    products = await exchange.products()
    product_ids = {p['id']: p for p in products}
    with open(args.portfolio) as f:
        portfolio = json.load(f)['portfolio']
        prices = await get_prices_dict(exchange, portfolio, 'EUR', product_ids)
        prices_btc = await get_prices_dict(exchange, portfolio, 'BTC', product_ids)

        allocations = compute_allocations(prices, portfolio)
        allocations_btc = compute_allocations(prices_btc, portfolio)

        return {
            'time': await exchange.time(),
            'allocations': {
                'EUR': allocations,
                'BTC': allocations_btc
            }
        }


async def get_target_allocations_cmd(exchange, args):
    currencies = [item['id'] for item in await exchange.currencies()]
    with open(args.portfolio) as f:
        portfolio = json.load(f)['portfolio']

    for c in portfolio:
        if not c in currencies:
            logging.error(f"Currency {c} not found for that exchange")
            exit(1)

    if args.method == 'equidistrib':
        return compute_equidistrib_allocations(
            portfolio, args.exclude)
    else:
        logging.error("Only 'equidistrib' is supported for now")
        exit(1)


async def change_portfolio_cmd(args):
    with open(args.portfolio) as f:
        d = json.load(f)
        time = d['time']
        portfolio = d['portfolio']
    for currency, value in zip(args.currencies, args.values):
        portfolio[currency] = str(value)
    return {'time': time, 'portfolio': portfolio}


async def main():
    args, args_parser = parse_cli_args()
    init_logging(args)

    if args.action == 'change-portfolio':
        json_output(await change_portfolio_cmd(args), args.out)
        return

    async with make_exchange(args) as exchange:
        if args.action == 'freeze':
            json_output(await get_freezed_state_exchange_config_cmd(exchange), args.out)
            return

        if args.action == 'get-portfolio':
            json_output(await get_portfolio_cmd(exchange, args), args.out)
            return

        if args.action == 'get-allocations':
            json_output(await get_allocations_cmd(exchange, args), args.out)
            return

        if args.action == 'get-target-allocations':
            json_output(await get_target_allocations_cmd(exchange, args), args.out)
            return

        time_str = await exchange.time()

        products = await exchange.products()
        product_ids = {p['id']: p for p in products}

        QUOTE_CURRENCY = 'EUR'

        if args.action == 'get-orders':
            fees = await exchange.fees()
            taker_fees = float(fees['taker_fee_rate'])

            with open(args.portfolio) as f:
                portfolio = json.load(f)['portfolio']
            with open(args.target_allocations) as f:
                target_allocations = json.load(f)

            total_target_alloc = 0
            for c in target_allocations:
                total_target_alloc += float(target_allocations[c])
            for c in target_allocations:
                target_allocations[c] = float(
                    target_allocations[c]) / total_target_alloc

            exchange_portfolio = await get_portfolio(exchange, [])
            for currency in portfolio:
                amount = float(portfolio[currency])
                exchange_amount = float(
                    exchange_portfolio[currency]) if currency in exchange_portfolio else 0
                if amount > exchange_amount:
                    logging.error(
                        f'Invalid portfolio: not enough {currency} on the exchange (asking for {amount}, having {exchange_amount} on the exchange)')
                    exit(-1)
                if amount < 0:  # each amount < 0 in the portfolio mean we want to take the maximum possible quantity, we replace with the exchange amount
                    portfolio[currency] = exchange_portfolio[currency]

            output = {}

            output['portfolio'] = portfolio
            output['target_allocations'] = target_allocations

            SWAP_CURRENCY = 'BTC'
            output['swap_currency'] = SWAP_CURRENCY

            for c in target_allocations:
                if not c in portfolio:
                    portfolio[c] = "0.0"

            prices = await get_prices_dict(exchange, portfolio, SWAP_CURRENCY, product_ids)
            allocations = compute_allocations(prices, portfolio)

            # todo dirty: patch for copra.rest.client.APIRequestError: Insufficient funds [400]
            new_EUR_alloc = int(
                allocations['portfolio']['EUR'] * 10) / 10
            logging.info(
                f"Patching EUR allocation {(allocations['portfolio']['EUR'])} to {new_EUR_alloc} to avoid errors.")
            allocations['portfolio']['EUR'] = new_EUR_alloc

            diff_allocs_perc = {}
            for c in target_allocations:
                diff_allocs_perc[c] = 100 * (
                    target_allocations[c] - allocations['allocations'][c]) / allocations['allocations'][c] if allocations['allocations'][c] > 0.0 else 100.0

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

            for c in ruled_size_to_sell:
                if ruled_size_to_sell[c] == 0.0:
                    diff_allocs_perc[c] = 0.0

            output['diff_alloc_percentage'] = diff_allocs_perc

            max_diff_allocs_perc = 0
            for c in diff_allocs_perc:
                max_diff_allocs_perc = max(
                    max_diff_allocs_perc, diff_allocs_perc[c])

            if max_diff_allocs_perc < args.rebalance_threshold:
                logging.info(
                    f'Max allocation difference is {max_diff_allocs_perc} % while threshold is {args.rebalance_threshold}: no rebalance needed.')
                json_output(output)
                return

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

                ticker = await exchange.ticker(product_id)

                base_order_price = float(
                    ticker['bid']) if side == 'sell' else float(ticker['ask'])

                total_order_price = base_order_price * size
                order_fees = total_order_price * taker_fees
                total_order_price += order_fees
                base_currency, quote_currency = product_id.split(
                    '-')[0], product_id.split('-')[1]

                # Safety check
                if side == 'buy':
                    # Cannot spend more than our quantity of quote currency
                    if total_order_price > allocations['portfolio'][quote_currency]:
                        logging.info(
                            f'Trying to buy {size} {base_currency} for {base_order_price} for a total of {total_order_price} {quote_currency}.')
                        logging.info(
                            f'But you only have {allocations["portfolio"][quote_currency]} {quote_currency}.')

                        size = 0.995 * allocations['portfolio'][quote_currency] / \
                            base_order_price

                        product = product_ids[product_id]
                        size = round_to_increment(
                            size, product['base_increment'])

                        min_size = float(product['base_min_size'])
                        max_size = float(product['base_max_size'])
                        if size < min_size:
                            size = 0
                        elif size > max_size:
                            size = max_size

                        ruled_size_to_sell[currency] = size

                        order['size'] = size

                        total_order_price = base_order_price * size
                        logging.info(
                            f'Changing for: buy {size} {base_currency} for {base_order_price} for a total of {total_order_price} {quote_currency}.')
                elif side == 'sell':
                    # Cannot sell more that our quantity of base currency (this one should not happen)
                    if size > allocations['portfolio'][base_currency]:
                        size = allocations['portfolio'][base_currency]

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
                elif side == 'buy' and percentage_diff > PERCENTAGE_THRESHOLD:
                    try_it = False

                limit_order = {
                    'base_order': order,
                    'ticker': ticker,
                    'order_price': order_price,
                    'base_order_price': base_order_price,
                    'corrected_order_price': corrected_order_price,
                    'order_fees': order_fees,
                    'increment_multiplier': increment_multiplier,
                    'percentage_diff': percentage_diff,
                    'try_it': try_it,
                    'done': False,
                    'submit_response': None,
                    'query_responses': []
                }
                limit_orders.append(limit_order)

            if args.do_it:
                for limit_order in limit_orders:
                    if limit_order['try_it']:
                        order = limit_order['base_order']
                        product_id = order['product_id']
                        side = order['side']
                        size = order['size']
                        order_price = limit_order['corrected_order_price']

                        success = False
                        trial_count = 0
                        while not success and trial_count < MAX_LIMIT_ORDER_TRIAL_COUNT:
                            trial_count += 1
                            try:
                                logging.info(
                                    f'Submitting order {side} {product_id} {order_price} {size} (total = {(order_price * size)})')
                                response = await exchange.limit_order(
                                    side, product_id, order_price, size)
                                if response:
                                    limit_order['submit_response'] = response
                                    logging.info(response)
                                    success = True
                            except Exception as e:
                                logging.error(e)
                                logging.info(order_price)
                                logging.info(quote_increment)

            submitted_count = len(
                [order for order in limit_orders if 'submit_response' in order])
            output['submitted_count'] = submitted_count

            if args.do_it:
                epoch = 0
                executed_count = 0

                while executed_count != submitted_count:
                    logging.info(
                        f'Epoch {epoch} executed count = {executed_count} / {submitted_count}')
                    for order in limit_orders:
                        if order['try_it'] and not order['done']:
                            id = order['submit_response']['id']
                            response = await exchange.get_order(id)
                            if response:
                                previous_response = None
                                if len(order['query_responses']) > 0:
                                    previous_response = order['query_responses'][-1].copy()
                                    del previous_response['time']
                                if len(order['query_responses']) == 0 or previous_response != response:
                                    response['time'] = await exchange.time()
                                    order['query_responses'].append(
                                        response)
                                if response['status'] == 'done':
                                    logging.info(f'{id} executed')
                                    # #todo compute new portfolio
                                    order['done'] = True
                                    executed_count += 1
                            else:
                                logging.info(f'{id} canceled')
                                order['done'] = True
                                order['error'] = True
                                executed_count += 1
                    epoch += 1
                    await asyncio.sleep(1)

                output['epoch'] = epoch

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


def compute_equidistrib_allocations(portfolio, exclude_symbols=[]):
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
    return {
        'total': total,
        'prices': float_prices,
        'portfolio': float_amount,
        'values': values,
        'allocations': allocations
    }


async def get_portfolio(exchange, exclude_symbols):
    accounts = await exchange.accounts()
    return {item['currency']: item['balance'] if not item['currency'] in exclude_symbols else "0"
            for item in accounts if float(item['balance']) != 0}


async def get_prices_dict(exchange, portfolio, quote_currency, product_ids):
    prices = {}
    for base_currency in portfolio:
        if base_currency == quote_currency:
            prices[base_currency] = 1.0
            continue
        wanted_product_id = f'{base_currency}-{quote_currency}'
        if wanted_product_id in product_ids:
            prices[base_currency] = float((await exchange.ticker(wanted_product_id))['price'])
            continue
        reverse_product_id = f'{quote_currency}-{base_currency}'
        if reverse_product_id in product_ids:
            price_of_quote_in_base = float((await exchange.ticker(reverse_product_id))['price'])
            price_of_base_in_quote = 1.0 / price_of_quote_in_base
            prices[base_currency] = price_of_base_in_quote
            continue
        if base_currency == 'BTC':
            logging.error(
                f'No market to convert {base_currency} to {quote_currency}')
            exit(-1)

        btc_product_id = f'{base_currency}-BTC'
        if btc_product_id in product_ids:
            btc_price = float((await exchange.ticker(btc_product_id))['price'])
        else:
            btc_product_id = f'BTC-{base_currency}'
            btc_price = 1.0 / float((await exchange.ticker(btc_product_id))['price'])

        base_product_id = f'BTC-{quote_currency}'
        base_price = float((await exchange.ticker(base_product_id))['price'])

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
    logging.getLogger().setLevel(logging.INFO)


def parse_cli_args():
    def parse_string_list(string):
        return string.split(',')

    def parse_number_list(string):
        return [float(i) for i in string.split(',')]

    parser = argparse.ArgumentParser(description='Desc')
    parser.add_argument('-l', '--log-file', help='Path to log file.')
    parser.add_argument('--rebalance-threshold',
                        type=float, default=0.0, help='Threshold to trigger rebalancing')
    parser.add_argument(
        '--exchange-type', default="none", help='Type of exchange among "none, freezed, backtest, live"'
    )
    parser.add_argument(
        '--config', help='Path to json file containing config for the specified exchange (apiKey, apiSecret, passPhrase) for live, freezed config for freezed')

    commands = parser.add_subparsers(
        title='commands', dest='action')

    command_parser = commands.add_parser('get-portfolio')
    command_parser.add_argument('-o', '--out', help='Output json file')
    command_parser.add_argument(
        '--exclude', type=parse_string_list, default=[], help='What currencies to exclude (comma separated list)')

    command_parser = commands.add_parser('change-portfolio')
    command_parser.add_argument(
        'portfolio', help='Portfolio stored in json file. Can be obtained with get-portfolio command.')
    command_parser.add_argument(
        'currencies', type=parse_string_list, help='List of currencies to change')
    command_parser.add_argument(
        'values', type=parse_number_list, help='Values to set (for each currency)')
    command_parser.add_argument('-o', '--out', help='Output json file')

    command_parser = commands.add_parser('freeze')
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
    command_parser.add_argument(
        '--exclude', type=parse_string_list, default=[], help='What quote currencies to exclude (comma separated list)')
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


if __name__ == "__main__":
    asyncio.set_event_loop(asyncio.new_event_loop())
    asyncio.run(main())
