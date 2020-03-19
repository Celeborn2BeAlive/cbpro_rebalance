#!/bin/bash

source venv_cbpro_rebalance_3_7_6/Scripts/activate

OUTFOLDER=$1

mkdir -p $OUTFOLDER

PORTFOLIO_FILE=$2

python main.py --config live-data/credentials.json --exchange-type live get-allocations $PORTFOLIO_FILE > $OUTFOLDER/allocations.json
python main.py --config live-data/credentials.json --exchange-type live get-target-allocations equidistrib $PORTFOLIO_FILE --exclude EUR,ZEC > $OUTFOLDER/target-allocations.json

if [[ -z $3 ]]; then
  python main.py --config live-data/credentials.json --exchange-type live get-orders $PORTFOLIO_FILE $OUTFOLDER/target-allocations.json > $OUTFOLDER/orders.json
else
  python main.py --config live-data/credentials.json --exchange-type live get-orders $PORTFOLIO_FILE $OUTFOLDER/target-allocations.json --do-it > $OUTFOLDER/orders-do-it.json
fi