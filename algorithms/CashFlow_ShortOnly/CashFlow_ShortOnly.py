import numpy as np
import math
import quantopian.optimize as opt
import quantopian.algorithm as algo
from quantopian.pipeline.experimental import risk_loading_pipeline
from quantopian.pipeline import CustomFactor, Pipeline
from quantopian.pipeline.data import Fundamentals
from quantopian.pipeline.data.builtin import USEquityPricing
from quantopian.pipeline.filters import QTradableStocksUS
from quantopian.pipeline.factors import SimpleBeta

nStocks = 50

def signalize(df):
   return ((df.rank() - 0.5)/df.count()).replace(np.nan,0.5)


def initialize(context):
    """
    Called once at the start of the algorithm.
    """

    context.max_leverage = 1.0/2
    context.max_pos_size = 0.01
    context.max_turnover = 0.95

    # Rebalance every day, 1 hour after market open.
    algo.schedule_function(
        rebalance,
        algo.date_rules.every_day(),
        algo.time_rules.market_close(hours=1),
    )

    # Record tracking variables at the end of each day.
    algo.schedule_function(
        record_vars,
        algo.date_rules.every_day(),
        algo.time_rules.market_close(),
    )

    # Create our dynamic stock selector.
    algo.attach_pipeline(make_pipeline(), 'pipeline')

    algo.attach_pipeline(
        risk_loading_pipeline(),
        'risk_pipe'
    )
    
    beta = SimpleBeta(
        target=sid(8554),
        regression_length=260,
    )

    pipe = Pipeline(
        columns={
            'beta': beta,
        },
        screen=beta.notnull(),
    )
    algo.attach_pipeline(pipe, 'beta_pipe')

    

class Volatility(CustomFactor):  
    inputs = [USEquityPricing.close]  
    window_length = 20  
    def compute(self, today, assets, out, close):  
        # [0:-1] is needed to remove last close since diff is one element shorter  
        daily_returns = np.diff(close, axis = 0) / close[0:-1]  
        out[:] = daily_returns.std(axis = 0) * math.sqrt(252)

# Create custom factor subclass to calculate a market cap based on yesterday's
# close
class MarketCap(CustomFactor):
    # Pre-declare inputs and window_length
    inputs = [USEquityPricing.close, Fundamentals.shares_outstanding]
    window_length = 1

    # Compute market cap value
    def compute(self, today, assets, out, close, shares):
        out[:] = close[-1] * shares[-1]

class EnterpriseValue(CustomFactor):
    # Pre-declare inputs and window_length
    inputs = [Fundamentals.enterprise_value]
    window_length = 1

    # Compute market cap value
    def compute(self, today, assets, out, ev):
        out[:] = ev[-1]

        
class LogEV(CustomFactor):
    # Pre-declare inputs and window_length
    inputs = [Fundamentals.enterprise_value]
    window_length = 1

    # Compute market cap value
    def compute(self, today, assets, out, ev):
        out[:] = np.log(ev[-1])
 
        
class LogMarketCap(CustomFactor):
    # Pre-declare inputs and window_length
    inputs = [USEquityPricing.close, Fundamentals.shares_outstanding]
    window_length = 1

    # Compute market cap value
    def compute(self, today, assets, out, close, shares):
        out[:] = np.log(close[-1] * shares[-1])


class PCFRatio(CustomFactor):   

    # Pre-declare inputs and window_length
    inputs = [Fundamentals.pcf_ratio] 
    window_length = 4
    
    # Compute factor1 value
    def compute(self, today, assets, out, var):
        out[:] = var[-2]

     
class FcfToEV(CustomFactor):
    # Pre-declare inputs and window_length
    inputs = [Fundamentals.enterprise_value, Fundamentals.free_cash_flow, USEquityPricing.close, Fundamentals.shares_outstanding] 
    window_length = 2
    
    # Compute factor1 value
    def compute(self, today, assets, out, ev, var, close, shares):
        out[:] = var[-2]/np.sqrt(ev[-2]*close[-2]*shares[-2])

class FcfToMcap(CustomFactor):
    # Pre-declare inputs and window_length
    inputs = [Fundamentals.market_cap, Fundamentals.free_cash_flow] 
    window_length = 2
    
    # Compute factor1 value
    def compute(self, today, assets, out, ev, var):
        out[:] = var[-2]/ev[-2]

        
class AvgPrice(CustomFactor):
    # Pre-declare inputs and window_length
    inputs = [USEquityPricing.close] 
    window_length = 22
    
    # Compute factor1 value
    def compute(self, today, assets, out, close):
        out[:] = np.mean(close[~np.isnan(close)])


class CashFlow(CustomFactor):
    # Pre-declare inputs and window_length
    inputs = [Fundamentals.operating_cash_flow, Fundamentals.shares_outstanding] 
    window_length = 2
    
    # Compute factor1 value
    def compute(self, today, assets, out, var, shares):
        out[:] = var[-2] / shares[-2]

        
        
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

    mkt_cap = MarketCap(mask = base_universe)
    log_mkt_cap = LogMarketCap(mask = base_universe)
    log_ev = LogEV(mask = base_universe)

    vol = Volatility(mask = base_universe)

    pcf_ratio = PCFRatio(mask = base_universe)
    fcfToEv = FcfToEV(mask = base_universe)
    fcfToMcap = FcfToMcap(mask = base_universe)
    
    w_pcf_ratio = pcf_ratio / (log_mkt_cap * vol)
    mkt_cap_top = mkt_cap.bottom(200)

    # Filter stocks out with Mcap < mcap of 100th stock in S&P500
    pipe_screen = base_universe & ~mkt_cap_top
    # & cash_return_negative

    pipe_columns = {
        'mkt_cap': mkt_cap,
        'log_mkt_cap': log_mkt_cap,
        'vol': vol,
        'pcf_ratio': pcf_ratio,
        'fcfToMcap':fcfToMcap,
        'fcfToEv':fcfToEv,
        'w_pcf_ratio': w_pcf_ratio,
    }

    return Pipeline(columns=pipe_columns, screen=pipe_screen)

def transform(df, field, multiplier=1):
    return signalize(multiplier*df[field])

        
def before_trading_start(context, data):
    """
    Called every day before market open.
    """
    output = algo.pipeline_output('pipeline')
    output['combined'] = transform(output, 'fcfToEv') 
    # output['combined'] = transform(output, 'pcf_ratio', -1)
    
    output['signal_rank'] = transform(output, 'combined', 1)
   
    output = output.sort_values(by = 'signal_rank', ascending=True)
    alpha = output['signal_rank']
    alpha = alpha.sub(alpha.mean())
    context.alpha = alpha
    # print  output[:nStocks]['signal_rank']
    # print alpha.head(10)
    # print alpha.tail(10)
    
    context.risk_factor_betas = algo.pipeline_output(
      'risk_pipe'
    ).dropna()

    context.beta_pipeline = algo.pipeline_output('beta_pipe')



def rebalance(context, data):
    
    alpha = context.alpha
    beta_pipeline = context.beta_pipeline

    if not alpha.empty:
        # Create MaximizeAlpha objective
        objective = opt.MaximizeAlpha(alpha)

        # Create position size constraint
        constrain_pos_size = opt.PositionConcentration.with_equal_bounds(
            -context.max_pos_size,
            0
        )

        # Constrain target portfolio's leverage
        max_leverage = opt.MaxGrossExposure(context.max_leverage)

        # Ensure long and short books
        # are roughly the same size
        dollar_neutral = opt.DollarNeutral()

        # Constrain portfolio turnover
        max_turnover = opt.MaxTurnover(context.max_turnover)
        
        factor_risk_constraints = opt.experimental.RiskModelExposure(
            context.risk_factor_betas,
            max_volatility = 0.2,
            version=opt.Newest
        )
        
        beta_neutral = opt.FactorExposure(
            beta_pipeline[['beta']],
            min_exposures={'beta': -0.55},
            max_exposures={'beta': -0.45},
        )

        
        # Rebalance portfolio using objective
        # and list of constraints
        algo.order_optimal_portfolio(
            objective=objective,
            constraints=[
                constrain_pos_size,
                max_leverage,
                # dollar_neutral,
                # max_turnover,
                # factor_risk_constraints,
                beta_neutral
            ]
        )


def record_vars(context, data):
    """
    Plot variables at the end of each day.
    """
    pass


def handle_data(context, data):
    """
    Called every minute.
    """
    pass