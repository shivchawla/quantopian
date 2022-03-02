"""
This is a template algorithm on Quantopian for you to adapt and fill in.
"""
import quantopian.algorithm as algo
from quantopian.pipeline import Pipeline, CustomFactor
from quantopian.pipeline.data.builtin import USEquityPricing
from quantopian.pipeline.filters import QTradableStocksUS
from quantopian.pipeline.factors import AverageDollarVolume


def initialize(context):
    """
    Called once at the start of the algorithm.
    """
    # Rebalance every day, 1 hour after market open.
    algo.schedule_function(
        closepositions,
        algo.date_rules.every_day(),
        algo.time_rules.market_close(minutes=10),
    )
    

    # # Record tracking variables at the end of each day.
    # algo.schedule_function(
    #     record_vars,
    #     algo.date_rules.every_day(),
    #     algo.time_rules.market_close(),
    # )
    
    
    # # Create our dynamic stock selector.
    algo.attach_pipeline(make_pipeline(), 'pipeline')

class CloseToLowReturnsMean(CustomFactor):
    inputs = [USEquityPricing.close, USEquityPricing.low]
    window_length = 66
    
    def compute(self, today, assets, out, close, low):  
        # [0:-1] is needed to remove last close since diff is one element shorter  
        # daily_returns = np.diff(close, axis = 0) / close[0:-1]
        daily_returns = (low[1:] - close[0:-1])/close[0:-1]
        out[:] = daily_returns.mean(axis = 0)
        

class CloseToHighReturnsMean(CustomFactor):
    inputs = [USEquityPricing.close, USEquityPricing.high]
    window_length = 66
    
    def compute(self, today, assets, out, close, high):  
        # [0:-1] is needed to remove last close since diff is one element shorter  
        # daily_returns = np.diff(close, axis = 0) / close[0:-1]
        daily_returns = (high[1:] - close[0:-1])/close[0:-1]
        out[:] = daily_returns.mean(axis = 0)
        
    
def make_pipeline():
    """
    A function to create our dynamic stock selector (pipeline). Documentation
    on pipeline can be found here:
    https://www.quantopian.com/help#pipeline-title
    """

    # Base universe set to the QTradableStocksUS
    dollar_volume = AverageDollarVolume(window_length=30)
    high_dollar_volume = dollar_volume.top(50)

    # Factor of yesterday's close price.
    yesterday_close = USEquityPricing.close.latest
    cthMean = CloseToHighReturnsMean()
    ctlMean = CloseToLowReturnsMean()
    
    pipe = Pipeline(
        columns={
            'close': yesterday_close,
            'cthMean': cthMean,
            'ctlMean': ctlMean,
        },
        screen=high_dollar_volume
    )
    
    return pipe


def before_trading_start(context, data):
    """
    Called every day before market open.
    """
    context.output = algo.pipeline_output('pipeline')

    # These are the securities that we are interested in trading each day.
    context.security_list = context.output.index
    
    context.high = {}
    context.low = {}


def closepositions(context, data):
    """
    Execute orders according to our schedule_function() timing.
    """
    positions = context.portfolio.positions
    
    for sec in positions:
        order_target_percent(sec, 0) 
    

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
    for sec in context.security_list:
        if sec not in positions:
            
            high = 0
            low = 0
            
            if sec in context.high:
                high = max(data.current(sec, 'high'), context.high[sec])
            else:
                high = data.current(sec, 'high')
            
            if sec in context.low:
                low = max(data.current(sec, 'low'), context.low[sec])
            else:
                low = data.current(sec, 'low')
            
                           
            close = context.output['close'][sec]
            ctlReturn = (low - close)/close
            cthReturn = (high - close)/close
        
            cthMean = context.output['cthMean'][sec]
            ctlMean = context.output['ctlMean'][sec]
        
            # print("High", high)
            # print("Low", low)
            # print("Close", close)
            # print("CtLReturn", ctlReturn)
            # print("CtHReturn", cthReturn)
            # print("CtLMean", ctlMean)
            # print("CtHMean", cthMean)
    
            if cthReturn > cthMean:
                print("Security Crossed High: ", sec)
                order_target_percent(sec, 0.1)
        
            if ctlReturn < ctlMean:
                print("Security Crossed Low: ", sec)
                order_target_percent(sec, -0.1)
            
            context.low[sec] = low
            context.high[sec] = high
         
            