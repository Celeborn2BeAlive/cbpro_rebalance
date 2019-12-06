import argparse
import logging
import os


def main():
    args, args_parser = parse_cli_args()
    init_logging(args)
    logging.log(logging.INFO, "Hello World")

    if args.action == 'parser1':
        parser1()
    elif args.action == 'parser2':
        parser2()
    else:
        args_parser.print_help()


def parser1():
    logging.info('parser1')
    return


def parser2():
    logging.info('parser2')
    return


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
        logging.info('Logging to standard output')


def parse_cli_args():
    def parse_string_list(string):
        return string.split(',')

    parser = argparse.ArgumentParser(description='Desc')
    parser.add_argument('-l', '--log-file', help='Path to log file.')

    commands = parser.add_subparsers(
        title='commands', dest='action')

    parser1 = commands.add_parser('parser1')
    parser1.add_argument('arg1', help='arg1')

    parser2 = commands.add_parser('parser2')
    parser2.add_argument('arg2', help='arg2')
    parser2.add_argument('-o1', '--optional-arg1',
                         type=int, help='optional_arg')
    parser2.add_argument('-o2', '--optional-arg2',
                         type=parse_string_list, help='otional_string_list')

    return parser.parse_args(), parser


if __name__ == "__main__":
    main()
