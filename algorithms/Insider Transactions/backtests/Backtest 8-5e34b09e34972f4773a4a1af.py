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
    # print("In signlaize")
    # print(df.head(10))
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

    insider_txns_form3_30d = Form3AggregatedTrades.slice(False, 30)
    insider_txns_form4and5_30d = Form4and5AggregatedTrades.slice(False, 30)

    insider_txns_form3_90d = Form3AggregatedTrades.slice(False, 90)
    insider_txns_form4and5_90d = Form4and5AggregatedTrades.slice(False, 90)

    #Get unique buyers/sellers    
    filers_form3_7d = insider_txns_form3_7d.num_unique_filers
    buyers_form4and5_7d = insider_txns_form4and5_7d.num_unique_buyers
    sellers_form4and5_7d = insider_txns_form4and5_7d.num_unique_sellers

    
    filers_form3_30d = insider_txns_form3_30d.num_unique_filers
    buyers_form4and5_30d = insider_txns_form4and5_30d.num_unique_buyers
    sellers_form4and5_30d = insider_txns_form4and5_30d.num_unique_sellers

    filers_form3_90d = insider_txns_form3_90d.num_unique_filers
    buyers_form4and5_90d = insider_txns_form4and5_90d.num_unique_buyers
    sellers_form4and5_90d = insider_txns_form4and5_90d.num_unique_sellers

    
    f3_prev = BusinessDaysSincePreviousEvent(inputs=[insider_txns_form3_30d.asof_date])  
    f5_prev = BusinessDaysSincePreviousEvent(inputs=[insider_txns_form4and5_30d.asof_date])  


    # Filter stocks out with Mcap < mcap of 100th stock in S&P500
    pipe_screen = base_universe & (filers_form3_90d.latest.notnull() | buyers_form4and5_90d.latest.notnull() | sellers_form4and5_90d.latest.notnull())

    pipe_columns = {
        'mkt_cap': mkt_cap,
        'vol': vol,
        
        'f3_prev_days': f3_prev,
        'f5_prev_days': f5_prev,
        
        'f3_buyers': TxnsCount(inputs=[filers_form3_30d]),
        'f5_buyers': TxnsCount(inputs=[buyers_form4and5_30d]),
        'f5_sellers': TxnsCount(inputs=[sellers_form4and5_30d]),
        
    }

    return Pipeline(columns=pipe_columns, screen=pipe_screen)



def before_trading_start(context, data):
    """
    Called every day before market open.
    """
    output = algo.pipeline_output('pipeline')
    
    # mkt_cap_rank = signalize(-output['mkt_cap'])

    # output['alpha1'] = (output['f5_buyers'] - output['f5_sellers'])/output['f5_prev_days']
    # output['alpha1'] = signalize(output['alpha1']*mkt_cap_rank)

    
    output['alpha2'] = output['f5_buyers'] - output['f5_sellers']
    output['alpha2'] = output[(output['alpha2'] != 0) & (output['alpha2'].notnull())]

    # output['alpha1'] = output['f3_buyers']
    # output = output[(output['alpha1'] != 0) & (output['alpha1'].notnull())]

    
    # print("-5")
    # print(output["mkt_cap"].head(20))
    
    # print("-4")
    # print(output["vol"].head(20))

    # print("-3")
    # print(output[(output['alpha2'] != 0)].head(20))
    
    # print("-2")
    # print(output[(output['alpha2'].notnull())].head(20))
          
    # print("-1")
    # print(output["f3_buyers"].head(20))
    
    # print("0")
    # print(output["f5_buyers"].head(20))

    # print("00")
    # print(output["f5_sellers"].head(20))
    
    # print("000")
    # print(output['f3_prev_days'].head(20))
    
    # print("0000")
    # print(output['f5_prev_days'].head(20))
    
    # print("1")
    # print(output['alpha2'].head(20))
    
    #30d
    #0 not enough data
    #1 38.52/0.09/1.09
    #2 7.58/0.03/0.32
    #3 4.94/0.02/0.21
    #4 13.13/0.03/0.45
    #5 1.98/0.01/0.1
    #6 18.33/0.05/0.56
    #7 -16.85/-0.04/-0.44
    #8 17.81/0.04/0.44/-20.38
    #9 -26.68/-0.08/-0.86/-31.95
    #10 27.86/0.07/0.62/-14.07
    #11 23.38/0.06/0.51/-13.02
    #12 -20.54/-0.05/-0.45/-23.2
    #13 10.7/0.03/0.39/-10.13
    #14 9.44/0.03/0.28/-10.13
    #15 30.87/0.08/0.64/-20.83
    #16 44.6/0.10/0.9/-13.47
    #17 -10.69/-0.02/-0.19/-19.12
    #18 -2.19/0/0/-17.13
    
    #7d
    #0 -10.61/-0.03/-0.87/-11.8
    #1 -9.17/-0.01/-0.2/-23.12
    #2 39.29/0.09/0.86/-12.33
    #3 -1.71/0.01/-27.74
    #4 -6.45/-0.01/-16.18
    
    #90d
    #0 -6.24/-0.02/-0.87/-8.69
    #1 14.93/0.04/0.65/-7.74
    #2 -6.47/-0.02/-14.62
    #3 
    
    #30d (f5_buyers)
    #1 16.28/0.04/0.81/-5.49
    #1 
    
   
    #30d (f3_buyers)
    #0 -12.92/-0.02/-28.42
    #1  -9.97/-0.01/-20.31
    #2 -ve
    #3 -ve
    #4 -ve
    #6 
    
    a0 = output[(output['f5_prev_days'] >= 1) & (output['f5_prev_days'] < 2)]
    # a1 = output[(output['f5_prev_days'] >= 2) & (output['f5_prev_days'] < 3)]
    # a2 = output[(output['f5_prev_days'] >= 3) & (output['f5_prev_days'] < 4)]
    a3 = output[(output['f5_prev_days'] >= 4) & (output['f5_prev_days'] < 5)]
    # a4 = output[(output['f5_prev_days'] >= 5) & (output['f5_prev_days'] < 6)]
    a5 = output[(output['f5_prev_days'] >= 6) & (output['f5_prev_days'] < 7)]
    # a6 = output[(output['f5_prev_days'] >= 7) & (output['f5_prev_days'] < 8)]

    # output['alpha2'] = signalize(0.5*signalize(a0)).add(0.1*signalize(a1), fill_value=0).add(0.1*signalize(a2), fill_value=0).add(0.1*signalize(a3), fill_value = 0).add(0.1*signalize(a4), fill_value=0).add(0.1*signalize(a5), fill_value=0))

    output['alpha2'] = signalize(0.6*signalize(a0).add(0.2*signalize(a3), fill_value=0.5).add(0.2*signalize(a5), fill_value=0.5))
    
    # a0 = output[(output['f3_prev_days'] >= 0) & (output['f3_prev_days'] < 1)]
    # a1 = output[(output['f3_prev_days'] >= 1) & (output['f3_prev_days'] < 2)]
    # a2 = output[(output['f3_prev_days'] >= 2) & (output['f3_prev_days'] < 3)]
    # a3 = output[(output['f3_prev_days'] >= 3) & (output['f3_prev_days'] < 4)]
    # a4 = output[(output['f3_prev_days'] >= 4) & (output['f3_prev_days'] < 5)]
    # a5 = output[(output['f3_prev_days'] >= 5) & (output['f3_prev_days'] < 6)]
    # a6 = output[(output['f3_prev_days'] >= 6) & (output['f3_prev_days'] < 7)]

        
    # output['alpha1'] = -signalize(0.5*signalize(-a0['alpha1']).add(0.1*signalize(-a1['alpha1']), fill_value=0).add(0.1*signalize(-a2['alpha1']), fill_value=0).add(0.1*signalize(-a3['alpha1']), fill_value = 0).add(0.1*signalize(-a4['alpha1']), fill_value=0).add(0.1*signalize(-a5['alpha1']), fill_value=0).add(0.1*signalize(-a6['alpha1']), fill_value=0))
    
    
        # output['alpha1'] = -signalize(0.5*signalize(-a0['alpha1']).add(0.1*signalize(-a1['alpha1']), fill_value=0).add(0.1*signalize(-a2['alpha1']), fill_value=0).add(0.1*signalize(-a3['alpha1']), fill_value = 0).add(0.1*signalize(-a4['alpha1']), fill_value=0).add(0.1*signalize(-a5['alpha1']), fill_value=0).add(0.1*signalize(-a6['alpha1']), fill_value=0))

    
    
    # output['alpha'] = output[(output['f5_prev_days'] >= 1) & (output['f5_prev_days'] < 2)]
    
    
    # a2 = output[(output['f5_prev_days'] >= 2) & (output['f5_prev_days'] < 3)]
    # a3 = output[(output['f5_prev_days'] >= 3) & (output['f5_prev_days'] < 4)]
    # a4 = output[(output['f5_prev_days'] >= 4) & (output['f5_prev_days'] < 5)]
    # a5 = output[(output['f5_prev_days'] >= 5) & (output['f5_prev_days'] < 6)]
    # a6 = output[(output['f5_prev_days'] >= 6) & (output['f5_prev_days'] < 7)]
    
    # output['k'] = output['f3_buyers']
    # output.loc[output['f5_prev_days'] != 1, 'k'] = np.nan
    # # output.loc[(output['f5_prev_days'] >= 10) & (output['f5_prev_days'] <= 16)  , 'k'] = -1.0
    
    # print("WTF")
    # print(output["f5_prev_days"])

    # print("2")
    # print(output[['f3_buyers', 'f3_prev_days', 'f5_prev_days']].sort_values(by = 'f3_buyers', ascending=False).head(50)) 

    
    # print("3")
    # print(output['k'].head(50)) 

    # output['alpha1'] = signalize(output['alpha1'])
    
    # print(output['alpha1'].head(50)) 

    # output['alpha3'] = signalize(output['k'])
    # # print(output['alpha2'].head(50))
    # output['alpha2'] = signalize(output['alpha2'])
    
    
    # print("sig")
    # print(output['alpha2'].sort_values(ascending=False).head(50))
    
    # print("sig_k")
    # print(output['alpha3'].sort_values(ascending=False).head(50))
    
    
    # print(a1.head(10))
    # print(output['alpha2'].head(50))
    # signalize(a1['alpha2'].add(signalize(a2['alpha2']), fill_value=0))    
        
    # output['alpha2'] = signalize(signalize(a1['alpha2']) + signalize(a2['alpha2']) + signalize(a3['alpha2']) + signalize(a4['alpha2']) + signalize(a5['alpha2']) + signalize(a6['alpha2']))
    
    # output = output[
    #     ((output['f5_prev_days'] >= 1) & (output['f5_prev_days'] < 7))]
    
    # output['alpha2'] = signalize(output['alpha2'])
    
    
    # output['alpha3'] = (output['f5_buyers'] - output['f5_sellers'])/(output['f5_buyers'] + output['f5_sellers'])
    # output['alpha3'] = output[output['alpha3'] != 0]
    # output['alpha3'] = output[output['alpha3'].notnull()]
    # output['alpha3'] = output[output['f5_prev_days'] < 7]
    # output['alpha3'] = signalize(output['alpha3']*mkt_cap_rank)
    
    
    # print(output['alpha1'].corr(output['alpha2']))
    
    context.alpha = output['alpha2']
    
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
        
        print(try_opt.sort_values(ascending = False).head(10))
        print(try_opt.sort_values(ascending = True).head(10))
        
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