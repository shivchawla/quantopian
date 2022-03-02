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

def signalize(df):
   return ((df.rank() - 0.5)/df.count()).replace(np.nan,0.5)


def initialize(context):
    """
    Called once at the start of the algorithm.
    """

    context.max_leverage = 1.0
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
    inputs = [Fundamentals.pcf_ratio, USEquityPricing.close] 
    window_length = 2
    
    # Compute factor1 value
    def compute(self, today, assets, out, var, close):
        out[:] = var[-2]

        
class PSRatio(CustomFactor):   

    # Pre-declare inputs and window_length
    inputs = [Fundamentals.ps_ratio] 
    window_length = 2
    
    # Compute factor1 value
    def compute(self, today, assets, out, var):
        out[:] = var[-2]

        
class FEYield(CustomFactor):   

    # Pre-declare inputs and window_length
    inputs = [Fundamentals.forward_earning_yield] 
    window_length = 2
    
    # Compute factor1 value
    def compute(self, today, assets, out, var):
        out[:] = var[-2]

     
class FcfToEV(CustomFactor):
    # Pre-declare inputs and window_length
    inputs = [Fundamentals.enterprise_value, Fundamentals.free_cash_flow, USEquityPricing.close, Fundamentals.shares_outstanding, Fundamentals.total_assets] 
    window_length = 2
    
    # Compute factor1 value
    def compute(self, today, assets, out, ev, var, close, shares, ta):
        out[:] = var[-2]/(ev[-2]*close[-2]*shares[-2]*ta[-2])**(1./3.)

        
class TaToEV(CustomFactor):
    # Pre-declare inputs and window_length
    inputs = [Fundamentals.enterprise_value, Fundamentals.free_cash_flow, USEquityPricing.close, Fundamentals.shares_outstanding, Fundamentals.total_assets] 
    window_length = 2
    
    # Compute factor1 value
    def compute(self, today, assets, out, ev, var, close, shares, ta):
        out[:] = ta[-2]/(ev[-2]*close[-2]*shares[-2])**(1./2.)

        
class ReToEV(CustomFactor):
    # Pre-declare inputs and window_length
    inputs = [Fundamentals.enterprise_value, Fundamentals.retained_earnings, USEquityPricing.close, Fundamentals.shares_outstanding, Fundamentals.total_assets] 
    window_length = 2
    
    # Compute factor1 value
    def compute(self, today, assets, out, ev, var, close, shares, ta):
        out[:] = var[-2]/(ev[-2]*close[-2]*shares[-2]*ta[-2])**(1./3.)


class RevenueToEV(CustomFactor):
    # Pre-declare inputs and window_length
    inputs = [Fundamentals.enterprise_value, Fundamentals.total_revenue, USEquityPricing.close, Fundamentals.shares_outstanding, Fundamentals.total_assets] 
    window_length = 2
    
    # Compute factor1 value
    def compute(self, today, assets, out, ev, var, close, shares, ta):
        out[:] = var[-2]/ev[-2]
        # *close[-2]*shares[-2]*ta[-2])**(1./3.)

        
class SalesYield(CustomFactor):   

    # Pre-declare inputs and window_length
    inputs = [Fundamentals.forward_earning_yield] 
    window_length = 2
    
    # Compute factor1 value
    def compute(self, today, assets, out, var):
        out[:] = var[-2]

        
class ForwardPERatio(CustomFactor):   

    # Pre-declare inputs and window_length
    inputs = [Fundamentals.forward_pe_ratio] 
    window_length = 2
    
    # Compute factor1 value
    def compute(self, today, assets, out, var):
        out[:] = var[-2]

     
class PretaxMargin(CustomFactor):   

    # Pre-declare inputs and window_length
    inputs = [Fundamentals.pretax_margin] 
    window_length = 1
    
    # Compute factor1 value
    def compute(self, today, assets, out, var):
        out[:] = var[-1]

class TEM(CustomFactor):
    """
    TEM = standard deviation of past 6 quarters' reports
    """
    inputs = [Fundamentals.capital_expenditure, Fundamentals.total_assets, Fundamentals.enterprise_value] 
    window_length = 390
    
    def compute(self, today, assets, out, capex, total_assets, ev):
        values = capex/ev
        out[:] = values.std(axis = 0)
        
        
class Yield(CustomFactor):  
        inputs = [Fundamentals.earning_yield]  
        window_length = 2  
        def compute(self, today, assets, out, syield):  
            out[:] =  syield[-2]
 
class Quality(CustomFactor):     
        inputs = [Fundamentals.gross_profit, Fundamentals.total_assets]
        window_length = 3*252
 
        def compute(self, today, assets, out, gross_profit, total_assets):
            norm = gross_profit / total_assets
            out[:] = (norm[-1] - np.mean(norm, axis=0)) / np.std(norm, axis=0)

            
class Momentum(CustomFactor):
        inputs = [USEquityPricing.open, USEquityPricing.high, USEquityPricing.low, USEquityPricing.close]
        window_length = 252
 
        def compute(self, today, assets, out, open, high, low, close):
 
            p = (open + high + low + close)/4
 
            out[:] = ((p[-21] - p[-252])/p[-252] -
                      (p[-1] - p[-21])/p[-21])

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
    
    pcf_ratio = PCFRatio(mask = base_universe)
    ps_ratio = PSRatio(mask = base_universe)
    fcfToEv = FcfToEV(mask = base_universe)
    sales_yield = SalesYield(mask = base_universe)
    taToEv = TaToEV(mask = base_universe)
    
    capex_vol = TEM(mask=base_universe)
    
    total_yield = Yield(mask=base_universe)
    
    # Filter stocks out with Mcap < mcap of 100th stock in S&P500
    pipe_screen = base_universe 

    pipe_columns = {
        'mkt_cap': mkt_cap,
        'log_mkt_cap': log_mkt_cap,
        'log_ev':log_ev,
        'vol': vol,
        'pcf_ratio': pcf_ratio,
        'ps_ratio': ps_ratio,
        'sales_yield':sales_yield,
        'total_yield': total_yield,
        'taToEv': taToEv,
        'fcfToEv':fcfToEv,
        'capex_vol': capex_vol
    }

    return Pipeline(columns=pipe_columns, screen=pipe_screen)

def transform(df, field, multiplier=1):
    return signalize(multiplier*df[field])

        
def before_trading_start(context, data):
    """
    Called every day before market open.
    """
    output = algo.pipeline_output('pipeline')
    print output['capex_vol']
    
    mk = transform(output, 'log_mkt_cap', 1)
    vl = transform(output, 'vol', -1)
    
    alpha1 = transform(output, 'taToEv', -1)
    alpha2 = transform(output, 'capex_vol', -1) 
    alpha3 = signalize(transform(output, 'ps_ratio', 1)*mk)
    alpha4 = signalize(signalize(transform(output, 'pcf_ratio', 1)*mk)*vl)
    alpha5 = transform(output, 'fcfToEv',1)
    alpha6 = transform(output, 'sales_yield', -1)
    # alpha7 = transform(output, 'total_yield', -1)
    
    output['combined'] = alpha1 + alpha2 + alpha3 + alpha4 + alpha5 
    # + alpha6 
    # + alpha7 
    output['signal_rank'] = transform(output, 'combined', 1)

    alpha = output['signal_rank']
    alpha = alpha.sub(alpha.mean())
    context.alpha = alpha
    
    output['min_weights'] = -context.max_pos_size
    output['max_weights'] = 0
    
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
        # dollar_neutral = opt.DollarNeutral()

        # Constrain portfolio turnover
        max_turnover = opt.MaxTurnover(context.max_turnover)
        
        factor_risk_constraints = opt.experimental.RiskModelExposure(
            context.risk_factor_betas,
            max_volatility = 0.2,
            version=opt.Newest
        )
        
        beta_neutral = opt.FactorExposure(
            beta_pipeline[['beta']],
            min_exposures={'beta': -1.1},
            max_exposures={'beta': -0.9},
        )

        algo.order_optimal_portfolio(
            objective = objective,
            constraints = [
                constrain_pos_size,
                max_leverage,
                # dollar_neutral,
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