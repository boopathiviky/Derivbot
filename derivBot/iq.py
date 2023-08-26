# -*- coding: utf-8 -*-
"""
Spyder Editor

Este é um arquivo de script temporário.
"""

import logging
from deriv_api import DerivAPI
import asyncio
from deriv_api.subscription_manager import SubscriptionManager, get_msg_type
from reactivex.subject import Subject
from reactivex import Observable
import os
import sys
import pandas as pd
import MetaTrader5 as mt5
import numpy as np
import collections
import more_itertools
import time
import datetime as datetime

async def login(verbose = False, iq = None, checkConnection = False):
    try:
        iq = DerivAPI(app_id=38654)
        authorize = await iq.authorize('oss8mMGuvvB5mJc')
        print('successfully connected')
    except Exception as e:
        print(e)
    return iq


async def higher(Money):
    iq = await login()
    proposal = await iq.proposal({"proposal": 1, "amount": Money, "barrier": "+0.1", "basis": "payout",
                               "contract_type": "CALL", "currency": "USD", "duration": 120, "duration_unit": "s",
                               "symbol": "R_100"
})
    proposal_id = proposal.get('proposal').get('id')
    buy = await iq.buy({"buy": proposal_id, "price": Money})
    # source_proposal: Observable = await api.subscribe({"proposal": 1, "amount": 100, "barrier": "+0.1", "basis": "payout",
    #                                    "contract_type": "CALL", "currency": "USD", "duration": 160,
    #                                    "duration_unit": "s",
    #                                    "symbol": "R_100",
    #                                    "subscribe": 1
    #                                    })
    # source_proposal.subscribe(lambda proposal: print(proposal['proposal']['id']))
    id=proposal['proposal']['id']
    print("contract buy successfully")
    return id
async def checkprofit():
    iq = await login()
    profit_table = await iq.profit_table({"profit_table": 1, "description": 1, "sort": "DESC"})
    profit = profit_table['profit_table']['transactions'][0]['sell_price']
    return profit
async def lower(Money):
    iq = await login()
    proposal = await iq.proposal({"proposal": 1, "amount": Money, "barrier": "+0.1", "basis": "payout",
                               "contract_type": "PUT", "currency": "USD", "duration": 120, "duration_unit": "s",
                               "symbol": "R_100"
})

    proposal_id = proposal.get('proposal').get('id')
    buy = await iq.buy({"buy": proposal_id, "price": Money})
    # source_proposal: Observable = await api.subscribe({"proposal": 1, "amount": 100, "barrier": "+0.1", "basis": "payout",
    #                                    "contract_type": "CALL", "currency": "USD", "duration": 160,
    #                                    "duration_unit": "s",
    #                                    "symbol": "R_100",
    #                                    "subscribe": 1
    #                                    })
    # source_proposal.subscribe(lambda proposal: print(proposal['proposal']['id']))
    id=proposal['proposal']['id']
    print("contract sell successfully")

    return id
  

    
async def get_balance():
    iq = await login()
    account = await iq.balance()
    print(account)
    return account['balance']['balance']



def get_connection():
    # establish MetaTrader 5 connection to a specified trading account
    if not mt5.initialize(login=31016367, server="Deriv-Demo",password="Viky123@"):
        print("initialize() failed, error code =",mt5.last_error())
        quit()
    return mt5
def get_data_needed():
    mt5=get_connection()
    rates = mt5.copy_rates_from_pos("Volatility 100 Index", mt5.TIMEFRAME_M1, 0, 5000)
    df = pd.DataFrame(rates)
    df['time']=pd.to_datetime(df['time'], unit='s')
    df=df[['time', 'open', 'high', 'low', 'close','tick_volume']]
    df.columns =['Date','Open', 'High', 'Low', 'Close','tick_volume']
    return df
def get_data():
    mt5=get_connection()
    rates = mt5.copy_rates_from_pos("Volatility 100 Index", mt5.TIMEFRAME_M1, 0, 300)
    df = pd.DataFrame(rates)
    df['time']=pd.to_datetime(df['time'], unit='s')
    df=df[['time', 'open', 'high', 'low', 'close','tick_volume']]
    df.columns =['Date','Open', 'High', 'Low', 'Close','tick_volume']
    return df

# if __name__ == '__main__':
    
#     id= asyncio.run(higher(5))