"""
This is a template algorithm on Quantopian for you to adapt and fill in.
"""
import quantopian.algorithm as algo

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


def before_trading_start(context, data):
    """
    Called every day before market open.
    """
    context.QQQ=sid(19920)
    context.SPY=sid(8554)
    context.SPY=sid(41382)
    context.XLK=sid(19658)
    context.QQQ3x = sid(39214)
    context.SPYS3x = sid(37083)
    context.SPY3x = sid(37514) 
    context.lev = 3
    context.allowed = True
    context.eod = False
    context.initial_val = context.portfolio.portfolio_value
    
def disallow(context, data):
     context.allowed = False


def rebalance(context, data):
    """
    Execute orders according to our schedule_function() timing.
    """
    hist = data.history([context.QQQ3x], fields="price", bar_count=12, frequency="1m")
    
    pct = hist.pct_change().iloc[1:].sum()[context.QQQ3x]
    
    hist_d = data.history([context.QQQ], fields="price", bar_count=30, frequency="1d")
    
    context.vol = hist_d.pct_change().iloc[1:].std()[context.QQQ]
     
    context.initial_val = context.portfolio.portfolio_value
    
    if pct > 0.01:        
        order_target_percent(context.QQQ, context.lev)
        order_target_percent(context.SPY, 0)
        order_target_percent(context.XLK, 0)

    else:
        order_target_percent(context.QQQ, 0)
        order_target_percent(context.SPY, 0)
        order_target_percent(context.XLK, 0)

  
def closepositions(context,data):
    lev = context.lev/2   
    order_target_percent(context.QQQ, lev)
    order_target_percent(context.SPY, -lev/2)
    order_target_percent(context.XLK, -lev/2)
    
    context.allowed=False
    context.eod = True

def record_vars(context, data):
    """
    Plot variables at the end of each day.
    """
    pass


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
            nOrdersXLK = len(get_open_orders(context.XLK))
            
            lev = context.lev/2
            if nOrdersQQQ == 0 and nOrdersSPY == 0 and nOrdersXLK == 0:
                order_target_percent(context.QQQ, lev)
                order_target_percent(context.SPY, -lev/2)       
                order_target_percent(context.XLK, -lev/2)

            # context.allowed = False