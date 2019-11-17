import argparse
import logging


def main():
    args = parse_cli_args()
    init_logging(args)
    logging.log(logging.INFO, "Hello World")


def init_logging(args):
    logging.getLogger().setLevel(logging.INFO)
    pass


def parse_cli_args():
    parser = argparse.ArgumentParser()
    return parser.parse_args()


if __name__ == "__main__":
    main()
