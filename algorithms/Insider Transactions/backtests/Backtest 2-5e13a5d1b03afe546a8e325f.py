import numpy as np
import math
import pandas as pd
import quantopian.optimize as opt
import quantopian.algorithm as algo
from quantopian.pipeline.experimental import risk_loading_pipeline
from quantopian.pipeline import CustomFactor, Pipeline
from quantopian.pipeline.data import Fundamentals
from quantopian.pipeline.data.builtin import USEquityPricing
from quantopian.pipeline.filters import QTradableStocksUS, StaticAssets
from quantopian.pipeline.factors import SimpleBeta, Returns, PercentChange,BusinessDaysSincePreviousEvent, BusinessDaysUntilNextEvent, VWAP
from quantopian.pipeline.data.psychsignal import stocktwits
from scipy import stats, linalg
from quantopian.pipeline.factors import AverageDollarVolume
import cvxpy as cp
from scipy import sparse as scipysp
from quantopian.pipeline.data.factset.estimates import PeriodicConsensus
import quantopian.pipeline.data.factset.estimates as fe
from quantopian.pipeline.data import factset
from sklearn import preprocessing
from scipy.stats.mstats import winsorize
from quantopian.pipeline.data.factset.ownership import Form3AggregatedTrades
from quantopian.pipeline.data.factset.ownership import Form4and5AggregatedTrades


WIN_LIMIT = 0
QL = 80

def preprocess(a):
    
    a = a.astype(np.float64)
    a[np.isinf(a)] = np.nan
    a = np.nan_to_num(a - np.nanmean(a))
    a = winsorize(a, limits=[WIN_LIMIT,WIN_LIMIT])
    
    return preprocessing.scale(a)

def signalize(df):
   # return ((df.rank() - 0.5)/df.count()).replace(np.nan,0.5)
    z = (df.rank() - 0.5)/df.count()
    return z.replace(np.nan, z.mean())
  
def initialize(context):
    set_slippage(slippage.FixedSlippage(spread=0.0))
    set_commission(commission.PerShare(cost=0, min_trade_cost=0))

    """
    Called once at the start of the algorithm.
    """
    context.max_leverage = 1.001
    context.min_leverage = 0.999
    context.max_pos_size = 0.4
    context.max_turnover = 0.95
    context.max_beta = 0.05
    context.max_net_exposure = 0.001
    context.max_volatility = 0.05
    context.max_sector_exposure = 0.02
    context.max_style_exposure = 0.38
    context.target_mkt_beta = 0
    context.normalizing_constant = 0.0714 
    
    context.beta_calc_days = 126
    context.init = True
    
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
        
class TxnsCount(CustomFactor):
    window_length = 3
    window_safe = True
    def compute(self, today, assets, out, b):  
        b=np.nan_to_num(b)
        m = b[-1]
        
        idx_zero = np.where(m == 0)[0]
        m[idx_zero] = b[-2][idx_zero]

        idx_zero = np.where(m == 0)[0]
        m[idx_zero] = b[-3][idx_zero]

        out[:] = m
        
def make_pipeline():
    """
    A function to create our dynamic stock selector (pipeline). Documentation
    on pipeline can be found here:
    https://www.quantopian.com/help#pipeline-title
    """
    # Base universe set to the QTradableStocksUS
    base_universe = QTradableStocksUS()

    mkt_cap = MarketCap(mask = base_universe)
    vol = Volatility(mask = base_universe)
    
    insider_txns_form3_7d = Form3AggregatedTrades.slice(False, 7)
    insider_txns_form4and5_7d = Form4and5AggregatedTrades.slice( False, 7)

    insider_txns_form3_90d = Form3AggregatedTrades.slice(False, 90)
    insider_txns_form4and5_90d = Form4and5AggregatedTrades.slice(False, 90)

    #Get unique buyers/sellers    
    filers_form3_7d = insider_txns_form3_7d.num_unique_filers
    buyers_form4and5_7d = insider_txns_form4and5_7d.num_unique_buyers
    sellers_form4and5_7d = insider_txns_form4and5_7d.num_unique_sellers
    
    filers_form3_90d = insider_txns_form3_90d.num_unique_filers
    buyers_form4and5_90d = insider_txns_form4and5_90d.num_unique_buyers
    sellers_form4and5_90d = insider_txns_form4and5_90d.num_unique_sellers

    
    f3_prev = BusinessDaysSincePreviousEvent(inputs=[insider_txns_form3_7d.asof_date])  
    f5_prev = BusinessDaysSincePreviousEvent(inputs=[insider_txns_form4and5_7d.asof_date])  


    # Filter stocks out with Mcap < mcap of 100th stock in S&P500
    pipe_screen = base_universe & (filers_form3_90d.latest.notnull() | buyers_form4and5_90d.latest.notnull() | sellers_form4and5_90d.latest.notnull())

    pipe_columns = {
        'mkt_cap': mkt_cap,
        'vol': vol,
        
        'f3_prev_days': f3_prev,
        'f5_prev_days': f5_prev,
        
        'f3_buyers': TxnsCount(inputs=[filers_form3_7d]),
        'f5_buyers': TxnsCount(inputs=[buyers_form4and5_7d]),
        'f5_sellers': TxnsCount(inputs=[sellers_form4and5_7d]),
        
    }

    return Pipeline(columns=pipe_columns, screen=pipe_screen)



def before_trading_start(context, data):
    """
    Called every day before market open.
    """
    output = algo.pipeline_output('pipeline')
    
    mkt_cap_rank = signalize(-output['mkt_cap'])

    output['alpha1'] = (output['f5_buyers'] - output['f5_sellers'])/output['f5_prev_days']
    output['alpha1'] = signalize(output['alpha1']*mkt_cap_rank)

    
    output['alpha2'] = (output['f5_buyers'] - output['f5_sellers'])
    output['alpha2'] = output[output['alpha2'] != 0]
    output['alpha2'] = output[output['f5_prev_days'] < 7]
    output['alpha2'] = signalize(output['alpha2']*mkt_cap_rank)
    
    
    output['alpha3'] = (output['f5_buyers'] - output['f5_sellers'])/(output['f5_buyers'] + output['f5_sellers'])
    output['alpha3'] = output[output['alpha3'] != 0]
    output['alpha3'] = output[output['alpha3'].notnull()]
    output['alpha3'] = output[output['f5_prev_days'] < 7]
    output['alpha3'] = signalize(output['alpha3']*mkt_cap_rank)
    
    
    print(output['alpha1'].corr(output['alpha2']))
    
    context.alpha = output['alpha1']
    
    output['min_weights'] = -context.max_pos_size
    output['max_weights'] = context.max_pos_size
    
    context.min_weights = output['min_weights']
    context.max_weights = output['max_weights']
    
    context.risk_factor_betas = algo.pipeline_output(
      'risk_pipe'
    ).dropna()
    

def rebalance(context, data):
    
    alpha = context.alpha
                       
    if not alpha.empty:
        
        alpha = alpha.sub(alpha.mean())
        try_opt = alpha/alpha.abs().sum()
        
        min_weights = context.min_weights.copy(deep = True)
        max_weights = context.max_weights.copy(deep = True)
        
        constrain_pos_size = opt.PositionConcentration(
            min_weights,
            max_weights
        )

        max_leverage = opt.MaxGrossExposure(context.max_leverage)

        # Ensure long and short books
        # are roughly the same size
        dollar_neutral = opt.DollarNeutral(context.max_net_exposure)
        
        # Constrain portfolio turnover
        max_turnover = opt.MaxTurnover(context.max_turnover)
        
        algo.order_optimal_portfolio(
            opt.TargetWeights(try_opt.dropna()),
            constraints = [
                # constrain_pos_size,
                max_leverage,
                dollar_neutral,
            ])
       
        

def record_vars(context, data):
    """
    Plot variables at the end of each day.
    """
    record("Num Positions", len(context.portfolio.positions))
    record("Leverage", context.account.leverage)
    pass

def handle_data(context, data):
    """
    Called every minute.
    """
    pass