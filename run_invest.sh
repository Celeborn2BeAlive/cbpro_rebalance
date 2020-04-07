#!/bin/bash

source venv_cbpro_rebalance_3_7_6/Scripts/activate

OUTFOLDER=$1
INVEST_QTY=$2

mkdir -p $OUTFOLDER

if [[ -z $3 ]]; then
  python main.py --config live-data/credentials.json --exchange-type live get-portfolio > $OUTFOLDER/01_portfolio_full.json
  python main.py change-portfolio $OUTFOLDER/01_portfolio_full.json EUR $INVEST_QTY > $OUTFOLDER/02_portfolio_to_rebalance.json
  ./run_rebalance_on_portfolio.sh $OUTFOLDER/03_simu $OUTFOLDER/02_portfolio_to_rebalance.json
else
  ./run_rebalance_on_portfolio.sh $OUTFOLDER/04_real $OUTFOLDER/02_portfolio_to_rebalance.json --do-it
fi