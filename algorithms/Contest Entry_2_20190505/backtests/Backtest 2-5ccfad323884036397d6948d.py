import numpy as np
import math
import pandas as pd
import quantopian.optimize as opt
import quantopian.algorithm as algo
from quantopian.pipeline.experimental import risk_loading_pipeline
from quantopian.pipeline import CustomFactor, Pipeline
from quantopian.pipeline.data import Fundamentals
from quantopian.pipeline.data.builtin import USEquityPricing
from quantopian.pipeline.filters import QTradableStocksUS
from quantopian.pipeline.factors import SimpleBeta
from quantopian.pipeline.data.psychsignal import stocktwits

def signalize(df):
   return ((df.rank() - 0.5)/df.count()).replace(np.nan,0.5)


def initialize(context):
    """
    Called once at the start of the algorithm.
    """

    context.max_leverage = 1.075
    context.max_pos_size = 0.01
    context.max_turnover = 0.95
    context.max_beta = 0.15
    context.max_net_exposure = 0.075
    context.max_volatility = 0.1

    # Rebalance every day, 1 hour after market open.
    algo.schedule_function(
        rebalance,
        algo.date_rules.every_day(),
        algo.time_rules.market_open(hours=1),
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


class Factor1(CustomFactor):   

    # Pre-declare inputs and window_length
    inputs = [Fundamentals.pcf_ratio] 
    window_length = 2
    
    # Compute factor1 value
    def compute(self, today, assets, out, var):
        out[:] = var[-2]

        
class Factor2(CustomFactor):   

    # Pre-declare inputs and window_length
    inputs = [Fundamentals.ps_ratio] 
    window_length = 2
    
    # Compute factor2 value
    def compute(self, today, assets, out, var):
        out[:] = var[-2]
    
     
class Factor3(CustomFactor):
    # Pre-declare inputs and window_length
    inputs = [Fundamentals.enterprise_value, Fundamentals.free_cash_flow, USEquityPricing.close, Fundamentals.shares_outstanding, Fundamentals.total_assets] 
    window_length = 2
    
    # Compute factor3 value
    def compute(self, today, assets, out, ev, var, close, shares, ta):
        out[:] = var[-2]/(ev[-2]*close[-2]*shares[-2]*ta[-2])**(1./3.)

        
class Factor4(CustomFactor):
    # Pre-declare inputs and window_length
    inputs = [Fundamentals.enterprise_value, Fundamentals.free_cash_flow, USEquityPricing.close, Fundamentals.shares_outstanding, Fundamentals.total_assets] 
    window_length = 2
    
    # Compute factor4 value
    def compute(self, today, assets, out, ev, var, close, shares, ta):
        out[:] = ta[-2]/(ev[-2]*close[-2]*shares[-2])**(1./2.)

         
class Factor5(CustomFactor):
    """
    TEM = standard deviation of past 6 quarters' reports
    """
    inputs = [Fundamentals.capital_expenditure, Fundamentals.enterprise_value] 
    window_length = 390
    
    # Compute factor5 value
    def compute(self, today, assets, out, capex, ev):
        values = capex/ev
        out[:] = values.std(axis = 0)
        

class Factor6(CustomFactor):  
        inputs = [Fundamentals.sales_yield]  
        window_length = 2

        # Compute factor6 value  
        def compute(self, today, assets, out, syield):  
            out[:] =  syield[-2]

        
class Factor7(CustomFactor):  
        inputs = [Fundamentals.earning_yield]  
        window_length = 2

        # Compute factor6 value  
        def compute(self, today, assets, out, syield):  
            out[:] =  syield[-2]

#Contest Entry#2 
class Factor8(CustomFactor):
        inputs = [USEquityPricing.high, USEquityPricing.low, USEquityPricing.close, stocktwits.bull_scored_messages, stocktwits.bear_scored_messages, stocktwits.total_scanned_messages]
        window_length = 21
        window_safe = True
        def compute(self, today, assets, out, high, low, close, bull, bear, total):
            v = np.nansum((high-low)/close, axis=0)
            out[:] = v*np.nansum(total*(bear-bull), axis=0)

def make_pipeline():
    """
    A function to create our dynamic stock selector (pipeline). Documentation
    on pipeline can be found here:
    https://www.quantopian.com/help#pipeline-title
    """

    # Base universe set to the QTradableStocksUS
    base_universe = QTradableStocksUS()

    mkt_cap = MarketCap(mask = base_universe)
    log_mkt_cap = LogMarketCap(mask = base_universe)
    log_ev = LogEV(mask = base_universe)
    vol = Volatility(mask = base_universe)
    
    f1 = Factor1(mask = base_universe)
    f2 = Factor2(mask = base_universe)
    f3 = Factor3(mask = base_universe)
    f4 = Factor4(mask = base_universe)
    f5 = Factor5(mask = base_universe)
    f6 = Factor6(mask = base_universe)
    f7 = Factor7(mask = base_universe)
    f8 = Factor8(mask = base_universe)
    
    # Filter stocks out with Mcap < mcap of 100th stock in S&P500
    pipe_screen = base_universe 

    pipe_columns = {
        'mkt_cap': mkt_cap,
        'log_mkt_cap': log_mkt_cap,
        'log_ev':log_ev,
        'vol': vol,
        'f1': f1,
        'f2': f2,
        'f3': f3,
        'f4': f4,
        'f5': f5,
        'f6': f6,
        'f7': f7,
        'f8': f8,
    }

    return Pipeline(columns=pipe_columns, screen=pipe_screen)

def transform(df, field, multiplier=1):
    return signalize(multiplier*df[field])

        
def before_trading_start(context, data):
    """
    Called every day before market open.
    """
    output = algo.pipeline_output('pipeline')
    
    mk = transform(output, 'log_mkt_cap', 1)
    vl = transform(output, 'vol', -1)
    
    alpha1 = signalize(transform(output, 'f1', 1)*mk)        
    alpha2 = signalize(signalize(transform(output, 'f2', 1)*mk)*vl)
    alpha3 = transform(output, 'f3',1)
    alpha4 = transform(output, 'f4', -1)
    alpha5 = transform(output, 'f5', -1) 
    alpha6 = transform(output, 'f6', -1)
    alpha7 = transform(output, 'f7', -1)
    alpha8 = signalize(transform(output, 'f8', 1)*vl)

    alpha1 = alpha1.sub(alpha1.mean())
    alpha2 = alpha2.sub(alpha2.mean())
    alpha3 = alpha3.sub(alpha3.mean())
    alpha4 = alpha4.sub(alpha4.mean())
    alpha5 = alpha5.sub(alpha5.mean())
    alpha6 = alpha6.sub(alpha6.mean())
    alpha7 = alpha7.sub(alpha7.mean())
    alpha8 = alpha8.sub(alpha8.mean())
    alpha3[alpha3 < 0] = 0
    
    output['signal_rank'] = alpha1 + alpha2 + alpha3 + alpha4 + alpha5 + alpha6 + alpha7 + alpha8
 
    alpha = output['signal_rank']
    context.alpha = alpha
    
    output['min_weights'] = -context.max_pos_size
    output['max_weights'] = context.max_pos_size
    
    context.min_weights = output['min_weights']
    context.max_weights = output['max_weights']
    
    context.risk_factor_betas = algo.pipeline_output(
      'risk_pipe'
    ).dropna()

    context.beta_pipeline = algo.pipeline_output('beta_pipe')


def rebalance(context, data):
    
    alpha = context.alpha
    beta_pipeline = context.beta_pipeline
    min_weights = context.min_weights.copy(deep = True)
    max_weights = context.max_weights.copy(deep = True)
    
    if not alpha.empty:
        # Create MaximizeAlpha objective
        objective = opt.MaximizeAlpha(alpha)

        constrain_pos_size = opt.PositionConcentration(
            min_weights,
            max_weights
        )

        # Constrain target portfolio's leverage
        max_leverage = opt.MaxGrossExposure(context.max_leverage)

        # Ensure long and short books
        # are roughly the same size
        dollar_neutral = opt.DollarNeutral(context.max_net_exposure)
        
        # Constrain portfolio turnover
        max_turnover = opt.MaxTurnover(context.max_turnover)
        
        factor_risk_constraints = opt.experimental.RiskModelExposure(
            context.risk_factor_betas,
            max_volatility = context.max_volatility,
            version=opt.Newest
        )
        
        beta_neutral = opt.FactorExposure(
            beta_pipeline[['beta']],
            min_exposures={'beta': -context.max_beta},
            max_exposures={'beta': context.max_beta},
        )

        algo.order_optimal_portfolio(
            objective = objective,
            constraints = [
                constrain_pos_size,
                max_leverage,
                dollar_neutral,
                max_turnover,
                factor_risk_constraints,
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