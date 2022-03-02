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
    
    algo.schedule_function(
        disallow,
        algo.date_rules.every_day(),
        algo.time_rules.market_close(hours=3),
    )

    
    algo.schedule_function(
        closepositions,
        algo.date_rules.every_day(),
        algo.time_rules.market_close(minutes=10),
    )

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
    context.pval = []
    context.eod = False
    

def before_trading_start(context, data):
    """
    Called every day before market open.
    """
    context.QQQ=sid(19920)
    context.SPY=sid(8554)
    context.VAL=sid(25909)
    context.GRW=sid(25910)
    # context.SPY=sid(41382)
    # context.SPY=sid(2174)
    # context.XLK=sid(19658)
    # context.XLK=sid(8554)
    context.QQQ3x = sid(39214)
    context.SPYS3x = sid(37083)
    context.SPY3x = sid(37514) 
    context.lev = 1.75
    context.intradayLev = 2.75  
    context.allowed = True
    context.eod = False
    context.initial_val = context.portfolio.portfolio_value
    context.delta = 0.1
    context.eod = False
    
    # print("WTFFFFFF")
    # print(context.pval)
    context.pval.append(context.initial_val)
    context.pval = context.pval[-12:]
    # print(context.pval)
    context.dd = drawdown(np.array(context.pval))
    context.factor = min(1, 0.8 - context.dd*5)
   
    
def drawdown(pval):
    
    dd = 0
    if len(pval) > 1:
        # print(np.diff(pval))
        # print(pval[:11])
        ret = np.diff(pval)/pval[:len(pval)-1]
        cumret = np.cumprod(1+ret)
        dd = 0
        for r in cumret:
            if r-1  < dd:
                dd = r - 1
    
        # print("Drawdown")
        # print(dd)
    
    return dd
    
def disallow(context, data):
     context.allowed = False


def rebalance(context, data):
    """
    Execute orders according to our schedule_function() timing.
    """
    hist = data.history([context.QQQ], fields="close", bar_count=2, frequency="1d")
    print(hist)
     
    # histm = data.history([context.QQQ], fields="price", bar_count=10, frequency="1m")
    
    # print(histm)
     
    lp = hist[context.QQQ].values[1]
    print(lp)
    # ltp = histm[context.QQQ].values[0]
    ltp = data.current(context.QQQ, 'price')
    print(ltp)
    
    pct = 3*((ltp/lp) - 1)
    
    
    hist_d = data.history([context.QQQ], fields="price", bar_count=30, frequency="1d")
    
    context.vol = hist_d.pct_change().iloc[1:].std()[context.QQQ]
     
    context.initial_val = context.portfolio.portfolio_value
    f=context.factor
    if pct > 0.01:        
        order_target_percent(context.QQQ, context.intradayLev*f)
        order_target_percent(context.SPY, 0)
        # order_target_percent(context.XLK, 0)
    elif pct < -0.03:       
        order_target_percent(context.QQQ, f*context.intradayLev/2)
        order_target_percent(context.SPY, -f*context.intradayLev/2)
           
    else:
        order_target_percent(context.QQQ, 0)
        order_target_percent(context.SPY, 0)
    #     order_target_percent(context.XLK, 0)

  
def closepositions(context,data):
    lev = context.lev/2
    # delta = max(0, min(context.delta, 0.1 - (2/3)*max(0.02, context.vol)/0.2))
    
    # 0.03/0.2  = 0.1 - (0.1/0.15) * 0.15
    # print("Delta")
    # print(delta)
    f = context.factor 
    order_target_percent(context.QQQ, lev*f)
    order_target_percent(context.SPY, -lev*f)
    # order_target_percent(context.VAL, (-lev + delta)/2)
    # order_target_percent(context.XLK, -lev/2)
    
    context.allowed=False
    context.eod = True

def record_vars(context, data):
    """
    Plot variables at the end of each day.
    """
    record(factor = context.factor)


def handle_data(context, data):
    """
    Called every minute.
    """
            
    positions = context.portfolio.positions
    port_val = context.portfolio.cash
    
    if len(positions) > 0:
    # and context.eod == False:
        # and context.allowed==True:
        for sid in positions:
            pos = positions[sid]
            port_val += data.current(sid, 'price') * pos.amount 
            
        ini_val = context.initial_val
        
        r = port_val/ini_val
        
        if  (r < 0.995): 
        # or (context.vol > 0.03 and r > 1.01): 
            nOrdersQQQ = len(get_open_orders(context.QQQ))
            nOrdersSPY = len(get_open_orders(context.SPY))
            # nOrdersXLK = len(get_open_orders(context.XLK))
            
            lev = context.lev/2
            # delta = max(0, min(context.delta, 0.1 - (2/3)*max(0.02, context.vol)/0.2))
            
            f = context.factor
            # print("Leverage")
            # print(lev*f)
            if nOrdersQQQ == 0 and nOrdersSPY == 0:
                # if context.eod is True:
                    order_target_percent(context.QQQ, lev*f)
                    order_target_percent(context.SPY, -lev*f)  
                # else:
                #     order_target_percent(context.QQQ, 0)
                #     order_target_percent(context.SPY, 0) 
                    

            # context.allowed = False