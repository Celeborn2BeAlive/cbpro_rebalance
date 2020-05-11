#!/bin/bash

source venv_cbpro_rebalance_3_7_6/Scripts/activate

OUTFOLDER=invest/`date +%Y-%m-%d`
INVEST_QTY=$1

mkdir -p $OUTFOLDER

if [[ -z $2 ]]; then
  python main.py --config live-data/credentials.json --exchange-type live get-portfolio > $OUTFOLDER/01_portfolio_full.json
  python main.py change-portfolio $OUTFOLDER/01_portfolio_full.json EUR $INVEST_QTY > $OUTFOLDER/02_portfolio_to_rebalance.json
  ./run_rebalance_on_portfolio.sh $OUTFOLDER/03_simu $OUTFOLDER/02_portfolio_to_rebalance.json
else
  ./run_rebalance_on_portfolio.sh $OUTFOLDER/04_real $OUTFOLDER/02_portfolio_to_rebalance.json --do-it
  python main.py --config live-data/credentials.json --exchange-type live get-portfolio > $OUTFOLDER/05_portfolio_new.json
  python main.py --config live-data/credentials.json --exchange-type live get-allocations $OUTFOLDER/05_portfolio_new.json > $OUTFOLDER/06_allocations_new.json
fi