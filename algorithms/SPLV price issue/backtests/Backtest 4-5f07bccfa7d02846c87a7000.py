"""
This is a template algorithm on Quantopian for you to adapt and fill in.
"""
import quantopian.algorithm as algo

def initialize(context):
    """
    Called once at the start of the algorithm.
    """    
    algo.schedule_function(
        startpositions,
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
    context.lev = 1.0


def before_trading_start(context, data):
    """
    Called every day before market open.
    """
    context.SPLV=sid(41382)
        

  
def startpositions(context,data):
    order_target_percent(context.SPLV, context.lev)
    
 
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
        
        if  (r < 0.8) or (r > 1.2): 
            order_target_percent(context.SPLV, 0)