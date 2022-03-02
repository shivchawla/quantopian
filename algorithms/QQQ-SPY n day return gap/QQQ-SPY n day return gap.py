"""
This is a template algorithm on Quantopian for you to adapt and fill in.
"""
import quantopian.algorithm as algo
import numpy as np

def initialize(context):
    """
    Called once at the start of the algorithm.
    """
    # Rebalance every day, 1 hour after market open.
    algo.schedule_function(
        rebalance,
        algo.date_rules.every_day(),
        algo.time_rules.market_open(minutes=10),
    )
    
    # algo.schedule_function(
    #     disallow,
    #     algo.date_rules.every_day(),
    #     algo.time_rules.market_close(hours=3),
    # )

    
    # algo.schedule_function(
    #     closepositions,
    #     algo.date_rules.every_day(),
    #     algo.time_rules.market_close(minutes=10),
    # )

    # Record tracking variables at the end of each day.
    algo.schedule_function(
        record_vars,
        algo.date_rules.every_day(),
        algo.time_rules.market_close(),
    )

    set_slippage(slippage.FixedSlippage(spread=0.0))
    set_commission(commission.PerShare(cost=0, min_trade_cost=0))
    context.initial_val = context.portfolio.portfolio_value
    context.vol = 0
    context.D5 = 0
    context.D6 = 0
    context.PnL = 0


def before_trading_start(context, data):
    """
    Called every day before market open.
    """
    context.QQQ=sid(19920)
    context.SPY=sid(8554)
    context.VAL=sid(25909)
    context.GRW=sid(25910)
    context.TOT=sid(22739)
    context.XLK=sid(19658)
    context.QQQ3x = sid(39214)
    context.SPYS3x = sid(37083)
    context.SPY3x = sid(37514) 
    context.lev = 1.75
    context.intradayLev = 2.75  
    context.allowed = True
    context.eod = False
    context.initial_val = context.portfolio.portfolio_value
    context.delta = 0.1
    
def disallow(context, data):
     context.allowed = False


def rebalance(context, data):
    """
    Execute orders according to our schedule_function() timing.
    """
    pctQQQ_5day = data.history([context.QQQ], fields="price", bar_count=11, frequency="1d").pct_change().iloc[1:].sum()[context.QQQ]
    
    pctQQQ_6day = data.history([context.QQQ], fields="price", bar_count=12, frequency="1d").pct_change().iloc[1:].sum()[context.QQQ]
    
    pctSPY_5day = data.history([context.SPY], fields="price", bar_count=11, frequency="1d").pct_change().iloc[1:].sum()[context.SPY]
    
    pctSPY_6day = data.history([context.SPY], fields="price", bar_count=12, frequency="1d").pct_change().iloc[1:].sum()[context.SPY]
    
    
    print("Diff QQQ - SPY 5 days - previous")
    print(context.D5)
    
    print("Diff QQQ - SPY 6 days")
    print(pctQQQ_6day - pctSPY_6day)
    
    # if context.D5 <= -0.02:
    #     diff = (pctQQQ_6day - pctSPY_6day) - context.D5 
    #     # print("Diff")
    #     # print(diff)
            
    #     context.PnL += diff
        
        # print("P/L")
        # print(context.PnL)
        
    if context.D5 <= -0.01:
        diff = (pctQQQ_6day - pctSPY_6day) - context.D5 
        # print("Diff")
        # print(diff)
            
        context.PnL -= diff
        
        # print("P/L")
        # print(context.PnL)    
        
    context.D5 = pctQQQ_5day - pctSPY_5day
    

def closepositions(context,data):
    lev = context.lev/2
    delta = max(0, min(context.delta, 0.1 - (2/3)*max(0.02, context.vol)/0.2))
    
    # 0.03/0.2  = 0.1 - (0.1/0.15) * 0.15
    print("Delta")
    print(delta)
    # order_target_percent(context.XLK, -lev/2)
    
   

def record_vars(context, data):
    """
    Plot variables at the end of each day.
    """
    record(PnL=context.PnL)


def handle_data(context, data):
    """
    Called every minute.
    """
    pass