import numpy as np
import pandas as pd
import datetime

def initialize(context):
    
    
    context.leverage = 1.0
    schedule_function(trade, date_rules.every_day(), time_rules.market_open(minutes=60))
    set_benchmark(symbol('SPY'))
    
    context.data = []
 
def before_trading_start(context,data): 
    
    fundamental_df = get_fundamentals(
        query(
            fundamentals.valuation.market_cap,
        )
        .filter(fundamentals.company_reference.primary_exchange_id == 'NYS')
        .filter(fundamentals.valuation.market_cap != None)
        .order_by(fundamentals.valuation.market_cap.desc()).limit(50))
    update_universe(fundamental_df.columns.values)
    context.stocks = [stock for stock in fundamental_df]
    
def handle_data(context, data):
             
    for stock in context.stocks:
        if stock.security_end_date < get_datetime() + datetime.timedelta(days=5):  # de-listed ?
            context.stocks.remove(stock)
        if stock in security_lists.leveraged_etf_list: # leveraged ETF?
            context.stocks.remove(stock)
            
    # check if data exists
    for stock in context.stocks:
        if stock not in data:
            context.stocks.remove(stock)

def trade(context,data):
    
    prices = history(2000,'1m','price')[context.stocks].dropna(axis=1)
    context.stocks = list(prices.columns.values)
    
    if len(context.stocks) == 0:
        return
    
    log.info(len(context.stocks))
    
    for stock in context.stocks:
        order_target_percent(stock, 1/len(context.stocks))
    
    for stock in data:
        if stock not in context.stocks:
            order_target_percent(stock,0)     
    
    
    
    
