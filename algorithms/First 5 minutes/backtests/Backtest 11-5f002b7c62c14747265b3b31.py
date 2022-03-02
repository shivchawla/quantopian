"""
This is a template algorithm on Quantopian for you to adapt and fill in.
"""
import quantopian.algorithm as algo
from quantopian.pipeline import Pipeline
from quantopian.pipeline.data.builtin import USEquityPricing
from quantopian.pipeline.filters import QTradableStocksUS


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
    context.fake_pos = None
    context.vol = 0
    # Create our dynamic stock selector.
    # algo.attach_pipeline(make_pipeline(), 'pipeline')


def make_pipeline():
    """
    A function to create our dynamic stock selector (pipeline). Documentation
    on pipeline can be found here:
    https://www.quantopian.com/help#pipeline-title
    """

    # Base universe set to the QTradableStocksUS
    base_universe = QTradableStocksUS()

    # Factor of yesterday's close price.
    yesterday_close = USEquityPricing.close.latest

    pipe = Pipeline(
        columns={
            'close': yesterday_close,
        },
        screen=base_universe
    )
    return pipe


def before_trading_start(context, data):
    """
    Called every day before market open.
    """
    # context.output = algo.pipeline_output('pipeline')

    # These are the securities that we are interested in trading each day.
    # context.security_list = context.output.index
    context.LONG = sid(39214)
    context.LONG_SPY = sid(37514) 
    context.SHORT = sid(39211)
    context.bench = sid(19920)
    context.allowed = True
    print("New Day")

    
def disallow(context, data):
     context.allowed = False


def rebalance(context, data):
    """
    Execute orders according to our schedule_function() timing.
    """
    hist = data.history([context.LONG], fields="price", bar_count=12, frequency="1m")
    
    pct = hist.pct_change().iloc[1:].sum()[context.LONG]
    
   
    hist_d = data.history([context.bench], fields="price", bar_count=30, frequency="1d")
    
    
    context.vol = hist_d.pct_change().iloc[1:].std()[context.bench]
    
    print("Volatility")
    print(context.vol)
    
    context.initial_val = context.portfolio.portfolio_value
    
    order_target_percent(context.LONG_SPY, 0)
    
    # context.fake_pos =  (context.initial_val/data.current(context.LONG, 'price'))
    # order_target_percent(context.bench, 0)
    
    
    # print(pct)
    l = 1.0
    if pct > 0.01:
        if context.vol < 0.015:
            l=1.0
        elif context.vol < 0.02:
            l=1
        elif context.vol < 0.025:
            l=1
        else:
            l=1
        
        order_target_percent(context.LONG, l)
        # order_target_percent(context.LONG_SPY, -l)
    else:
        order_target_percent(context.LONG, 0)
  
def closepositions(context,data):
    order_target_percent(context.LONG, 1)
    order_target_percent(context.SHORT, 0)
    order_target_percent(context.LONG_SPY, -1)
    # order_target_percent(context.bench, 1)

    context.allowed = False

def record_vars(context, data):
    """
    Plot variables at the end of each day.
    """
    pass


def handle_data(context, data):
    """
    Called every minute.
    """
    # if (context.fake_pos is not None) and (context.allowed == True):
    #     pval = context.fake_pos * data.current(context.LONG, 'price')
    #     ival = context.initial_val
    #     positions = context.portfolio.positions
    #     oo = get_open_orders(context.LONG)
          
    #     if (pval/ival > 1.01 ) and (len(positions) == 0) and len(oo)==0:
    #          print("ordering More")
    #          order_target_percent(context.LONG, 1)
          
    positions = context.portfolio.positions
    port_val = context.portfolio.cash
    if len(positions) > 0:
        
        for sid in positions:
            pos = positions[sid]
            port_val += data.current(sid, 'price') * pos.amount 
        
        ini_val = context.initial_val
        r = port_val/ini_val
        
        # if r > 1.01:
        #     print("Profit")
        #     print(r)
        
        if  (r < 0.995) or (context.vol > 0.03 and r > 1.01): 
        # or (r > 1.06):
       	    
            # print("Liquidating")
            # print("Buy Price")
            # print(positions[context.LONG].cost_basis)
            # print("Sale Price")
            # print(data.current(context.LONG, 'price'))
            order_target_percent(context.LONG, 1)
            order_target_percent(context.LONG_SPY, -1)
            # order_target_percent(context.LONG, 1)
            context.allowed = False