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
        algo.time_rules.market_close(minutes=5),
    )
    
    # algo.schedule_function(
    #     disallow,
    #     algo.date_rules.every_day(),
    #     algo.time_rules.market_close(hours=3),
    # )

    
    # algo.schedule_function(
    #     closepositions,
    #     algo.date_rules.every_day(),
    #     algo.time_rules.market_close(minutes=2),
    # )

    # Record tracking variables at the end of each day.
    algo.schedule_function(
        record_vars,
        algo.date_rules.every_day(),
        algo.time_rules.market_close(),
    )

    set_slippage(slippage.FixedSlippage(spread=0.0))
    set_commission(commission.PerShare(cost=0, min_trade_cost=0))
    set_benchmark(sid(39214))
    # set_benchmark(sid(24))
    # set_benchmark(sid(19920))
    
    context.initial_val = context.portfolio.portfolio_value
    context.open_price = None
    context.open_ret = None
    context.ret_1D = None
    context.lev = 1
    context.eod = False
    context.intraday = False
    

def before_trading_start(context, data):
    """
    Called every day before market open.
    """
    context.QQQ=sid(19920)
    # context.QQQ = sid(24)
    # context.QQQ3x = sid(39214)
    context.TQQQ = sid(39214)
    
    hist = data.history([context.QQQ], fields="price", bar_count=2, frequency="1d")
    
    returns = hist.pct_change()
    context.ret_1D = returns[context.QQQ].values[-1]
    context.last_close = hist[context.QQQ].values[-1]
    
    context.open_price = None;
    context.open_ret = None;
    context.eod = False
    context.intraday = False
        
    
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
    return dd



def rebalance(context, data):
    """
    Execute orders according to our schedule_function() timing.
    """
    QQQ = context.QQQ
    TQQQ = context.TQQQ
    
    lp = context.last_close
    ltp = data.current(QQQ, 'price')
    ret_today = 3*((ltp/lp) - 1)
    
    ret_1D = 3*context.ret_1D
    
    # print("Returns")
    # print(ret_1D)    
    # print(ret_today)

    context.eod = True
     
    if abs(ret_1D) < 0.04 and abs(ret_today) < 0.04:
        context.lev = 3
        order_target_percent(TQQQ, 1)    
    else:
        context.lev = 1
        order_target_percent(QQQ, 1)
    
    # print("Leverage")
    # print(context.lev)
    
    # order_target_percent(QQQ, context.lev)
        
  
def record_vars(context, data):
    """
    Plot variables at the end of each day.
    """
    record(lev = context.lev)


def handle_data(context, data):
    """
    Called every minute.
    """        
    QQQ = context.QQQ
    TQQQ = context.TQQQ
    
    if (context.open_ret is None) or (context.open_price is None):
        context.open_price = data.current(QQQ, 'price');
        context.open_ret = 3*(context.open_price/context.last_close - 1)
      
    positions = context.portfolio.positions
    
    if len(positions) > 0 and context.eod == False and context.intraday == False:
        
        pos = QQQ in positions  ? positions[QQQ] : None 
        pos_val_QQQ = QQQ in positions ? data.current(QQQ, 'price') * pos.amount : 0 
        ini_val_QQQ = QQQ in positions ? pos.cost_basis * pos.amount : 0
        
        pos = TQQQ in positions ? positions[TQQQ] : None
        pos_val_TQQQ = TQQQ in positions ? data.current(TQQQ, 'price') * pos.amount : 0 
        ini_val_TQQQ = TQQQ in positions ? pos.cost_basis * pos.amount : 0
            
        rQQQ = QQQ in positions ? pos_val_QQQ/ini_val_QQQ : 0
        rTQQQ =  TQQQ in positions ? pos_val_TQQQ/ini_val_TQQQ : 0
       
        if (TQQQ in positions and context.lev == 3 and rTQQQ <= 0.99 and context.open_ret > 0) or (context.lev == 3 and r <= 0.96): 
            
            nOrdersTQQQ = len(get_open_orders(TQQQ))
            
            if nOrdersTQQQ == 0:
                order_target_percent(TQQQ, 0)
                
        elif (context.lev == 3 and rTQQQ >= 1.01 and context.open_ret < 0) or (context.lev == 1 and rQQQ >= 1.0033 and context.open_ret < 0): 
            
            nOrdersTQQQ = len(get_open_orders(TQQQ))
            context.intraday = True
            if nOrdersTQQQ == 0:
                order_target_percent(TQQQ, 1)
                
        elif (context.lev == 3 and rTQQQ <= 0.99 and context.open_ret > 0) or (context.lev == 1 and rQQQ <= 0.9966 and context.open_ret > 0): 
            
            nOrdersQQQ = len(get_open_orders(QQQ))
            context.intraday = True
            if nOrdersQQQ == 0:
                order_target_percent(TQQQ, -1)
                
    elif len(positions) == 1 and context.intraday == True:
        pos = positions[TQQQ]
        pos_val = data.current(TQQQ, 'price') * pos.amount 
        ini_val = pos.cost_basis * pos.amount 
                    
        r = pos_val/ini_val
        
        if r < 0.99 :
            nOrdersQQQ = len(get_open_orders(QQQ))
            if nOrdersQQQ == 0:
                order_target_percent(QQQ, 1)