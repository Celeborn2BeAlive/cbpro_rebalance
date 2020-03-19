#!/bin/bash

source venv_cbpro_rebalance_3_7_6/Scripts/activate

OUTFOLDER=$1

mkdir -p $OUTFOLDER

python main.py --config live-data/credentials.json --exchange-type live get-portfolio --exclude EUR > $OUTFOLDER/portfolio.json
python main.py --config live-data/credentials.json --exchange-type live get-allocations $OUTFOLDER/portfolio.json > $OUTFOLDER/allocations.json
python main.py --config live-data/credentials.json --exchange-type live get-target-allocations equidistrib $OUTFOLDER/portfolio.json --exclude EUR,ZEC > $OUTFOLDER/target-allocations.json

if [[ -z $2 ]]; then
  python main.py --config live-data/credentials.json --exchange-type live get-orders $OUTFOLDER/portfolio.json $OUTFOLDER/target-allocations.json > $OUTFOLDER/orders.json
else
  python main.py --config live-data/credentials.json --exchange-type live get-orders $OUTFOLDER/portfolio.json $OUTFOLDER/target-allocations.json --do-it > $OUTFOLDER/orders-do-it.json
fi