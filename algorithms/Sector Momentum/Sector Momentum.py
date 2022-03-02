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
from quantopian.pipeline.factors import SimpleBeta, Returns, PercentChange
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
from sklearn.decomposition import PCA


WIN_LIMIT = 0
QL = 80
ML = 22

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
    context.max_pos_size = 0.35
    context.max_turnover = 0.95
    context.max_beta = 0.05
    context.max_net_exposure = 0.001
    context.max_volatility = 0.05
    context.max_sector_exposure = 0.02
    context.max_style_exposure = 0.38
    context.target_mkt_beta = 0
    context.normalizing_constant = 0.0714 
    
    context.etfs = symbols('VAW', 'VCR', 'VFH', 'VNQ', 'VDC', 'VHT', 'VPU', 'VOX', 'VDE', 'VIS', 'VGT')
    
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
    algo.attach_pipeline(make_pipeline(context), 'pipeline')   
    
    algo.attach_pipeline(
        risk_loading_pipeline(),
        'risk_pipe'
    )
    
    mkt_beta = SimpleBeta(
        target=sid(8554),
        regression_length=context.beta_calc_days,
    )
    
    value_beta = SimpleBeta(
        target=sid(22010),
        regression_length=context.beta_calc_days,
    )
    
    growth_beta = SimpleBeta(
        target=sid(22009),
        regression_length=context.beta_calc_days,
    )
    
    large_beta = SimpleBeta(
        target=sid(22148),
        regression_length=context.beta_calc_days,
    )
    
    small_beta = SimpleBeta(
        target=sid(21508),
        regression_length=context.beta_calc_days,
    )
    
    beta_pipe = Pipeline(
        columns={
            'mkt_beta': mkt_beta,
            'value_beta': value_beta,
            'growth_beta': growth_beta,
            'small_beta': small_beta,
            'large_beta': large_beta,
        },
        screen = QTradableStocksUS() & 
        mkt_beta.notnull() & 
        value_beta.notnull() & 
        growth_beta.notnull() & 
        small_beta.notnull() & 
        large_beta.notnull()
    )
    
    algo.attach_pipeline(beta_pipe, 'beta_pipe')
    
    
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
        

class Momentum(CustomFactor):
    inputs = [USEquityPricing.close]
    window_length = ML*12  
    def compute(self, today, assets, out, close):  
        # [0:-1] is needed to remove last close since diff is one element shorter  
        out[:] = np.nanmean([
            # close[-1]/close[-ML], #/-2.5
            # close[-1*ML]/close[-2*ML], #-3.6/-8.31 
            # close[-2*ML]/close[-3*ML],  #2.82/14.95(0.03)
            # close[-3*ML]/close[-4*ML], #-0.75/-12.21
            # close[-4*ML]/close[-5*ML],  #-3.83/-10.48
            # close[-5*ML]/close[-6*ML], #-10.55/-1.33
            # close[-6*ML]/close[-7*ML], #-3.5/6.09
            # close[-7*ML]/close[-8*ML], #-1.06/5.62
            # close[-8*ML]/close[-9*ML], #2.34/23.51(0.03)
            # close[-9*ML]/close[-10*ML],  #-3.89/-4.36
            # close[-10*ML]/close[-11*ML], #5.2/15.93(0.02)
            # close[-11*ML]/close[-12*ML] #4.76/11.14
            ], axis=0)
            

class mean_rev(CustomFactor):   
    inputs = [USEquityPricing.high,USEquityPricing.low,USEquityPricing.close]
    window_length = 60
    window_safe = True
    def compute(self, today, assets, out, high, low, close):

        p = (high+low+close)/3

        m = len(close[0,:])
        n = len(close[:,0])

        b = np.zeros(m)
        a = np.zeros(m)

        for k in range(10,n+1):
            price_rel = np.nanmean(p[-k:,:],axis=0)/p[-1,:]
            wt = np.nansum(price_rel)
            b += wt*price_rel
            price_rel = 1.0/price_rel
            wt = np.nansum(price_rel)
            a += wt*price_rel

            out[:] = preprocess(b-a)
    
def make_pipeline(context):
    """
    A function to create our dynamic stock selector (pipeline). Documentation
    on pipeline can be found here:
    https://www.quantopian.com/help#pipeline-title
    """
    # Base universe set to the QTradableStocksUS
    base_universe = StaticAssets(context.etfs)

    mkt_cap = MarketCap(mask = base_universe)
    vol = Volatility(mask = base_universe)
    adv = AverageDollarVolume(mask = base_universe, window_length = 22)
    # sector = Sector(mask = base_universe)

    f1 = Momentum(mask = base_universe)  
    f2 = mean_rev(mask = base_universe)
    f3 = -Volatility(mask = base_universe)
    
    # Filter stocks out with Mcap < mcap of 100th stock in S&P500
    pipe_screen = base_universe 
    
    pipe_columns = {
        'mkt_cap': mkt_cap,
        'vol': vol,
        'adv':adv,
        
        'f1':f1,  
        'f2':f2,  
        'f3': f3,
    }

    return Pipeline(columns=pipe_columns, screen=pipe_screen)


    
def transform(df, field, multiplier=1):
    return signalize(multiplier*df[field])

def weightedMean(df, weights):
    return df.mean()

def before_trading_start(context, data):
    """
    Called every day before market open.
    """
    output = algo.pipeline_output('pipeline')
    mkt_cap = output['mkt_cap']
    vol = output['vol']
    adv = output['adv']
    
    for sym in context.etfs:
        if not data.can_trade(sym):
            output = output.drop([sym])

        
    context.alpha = signalize(output['f1']) 
    
    output['min_weights'] = -context.max_pos_size
    output['max_weights'] = context.max_pos_size
    
    context.min_weights = output['min_weights']
    context.max_weights = output['max_weights']
    
    # context.risk_factor_betas = algo.pipeline_output(
    #   'risk_pipe'
    # ).dropna()
    
    # context.beta_pipeline = algo.pipeline_output('beta_pipe')
    # context.volatility = vol
    # context.adv = adv
    

def rebalance(context, data):
    
    alpha = context.alpha
                       
    if not alpha.empty:
        
        alpha = alpha.sub(alpha.mean())
        try_opt = alpha/alpha.abs().sum()
        
        
        # Running optimization implementation (testing)         
        # my_opt_weights = optimize(context, alpha, factor_loadings, beta, cov_style, cov_sector, variance, adv_wt, try_opt)
        # my_opt = pd.Series(my_opt_weights, index=validSecurities) 
        # algo.order_optimal_portfolio(
        #     opt.TargetWeights(try_opt),
        #     constraints = [])
                
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
                constrain_pos_size,
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