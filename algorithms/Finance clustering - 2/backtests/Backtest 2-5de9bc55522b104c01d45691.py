from quantopian.algorithm import attach_pipeline, pipeline_output
from quantopian.pipeline import CustomFactor, Pipeline
from quantopian.pipeline.factors import SimpleBeta, Returns, PercentChange, BusinessDaysSincePreviousEvent
from quantopian.pipeline.data.builtin import USEquityPricing
from quantopian.pipeline.data import Fundamentals
import quantopian.optimize as opt
import quantopian.algorithm as algo
from sklearn import preprocessing
from quantopian.pipeline.experimental import risk_loading_pipeline
from quantopian.pipeline.filters import QTradableStocksUS
from quantopian.pipeline.data.factset import RBICSFocus
from quantopian.pipeline.data.psychsignal import stocktwits
from scipy.stats.mstats import winsorize
from zipline.utils.numpy_utils import (
    repeat_first_axis,
    repeat_last_axis,
)
from quantopian.pipeline.data import factset
import quantopian.pipeline.data.factset.estimates as fe

from scipy.stats.mstats import gmean
from sklearn.cluster import SpectralClustering
 
import numpy as np
import pandas as pd

from collections import Counter

WIN_LIMIT = 0
N_FACTORS = None
N_FACTOR_WINDOW = 5 # trailing window of alpha factors exported to before_trading_start
MAX_NET_EXPOSURE = 0.001
MAX_BETA_EXPOSURE = 0.05
QL=66

N1=10
N2=50

def preprocess(a):
    
    a = a.astype(np.float64)
    a[np.isinf(a)] = np.nan
    a = np.nan_to_num(a - np.nanmean(a))
    a = winsorize(a, limits=[WIN_LIMIT,WIN_LIMIT])
    
    return preprocessing.scale(a)

class mean_rev(CustomFactor):   #alpha4
    inputs = [USEquityPricing.high,USEquityPricing.low,USEquityPricing.close]
    window_length = 30
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
            

class MaxGap(CustomFactor):     #alpha13
    # the biggest absolute overnight gap in the previous 90 sessions
    inputs = [USEquityPricing.close] ; window_length = 90
    window_safe = True
    def compute(self, today, assets, out, close):
        abs_log_rets = np.abs(np.diff(np.log(close),axis=0))
        max_gap = np.max(abs_log_rets, axis=0)
        out[:] = preprocess(max_gap)

class revenue_growth_mean(CustomFactor):  #alpha76
    inputs = [Fundamentals.revenue_growth]  
    window_length = QL*8
    window_safe = True
    def compute(self, today, assets, out, var):  
        arr = [var[-1],
            var[-QL],
            var[-QL*2],
            var[-QL*3],
            var[-QL*4],
            var[-QL*5],
            var[-QL*6]]

        mean = np.mean(arr, axis=0)
        out[:] = preprocess(mean)

class TA_growth_mean(CustomFactor):  
    inputs = [Fundamentals.total_assets]  
    window_length = QL*9
    window_safe = True
    
    def compute(self, today, assets, out, var):  
        arr = [var[-1],
            var[-QL],
            var[-QL*2],
            var[-QL*3],
            var[-QL*4],
            var[-QL*5],
            var[-QL*6],
           var[-QL*7],
         var[-QL*8]]
        
        arr = np.diff(arr, axis=0)/arr[1:]
        mean = np.mean(arr, axis=0)
        out[:] = preprocess(mean)
        

class GainLoansTa_mean(CustomFactor):  
    inputs = [Fundamentals.gainon_saleof_loans, Fundamentals.total_assets]  
    window_length = QL*9
    window_safe = True
    
    def compute(self, today, assets, out, v, ta):  
        var  = v/ta
        arr = [var[-1],
            var[-QL],
            var[-QL*2],
            var[-QL*3],
            var[-QL*4],
            var[-QL*5],
            var[-QL*6],
           var[-QL*7],
         var[-QL*8]]
        
        mean = np.mean(arr, axis=0)
        out[:] = -preprocess(mean)
    
class IncomeDeposits_growth_stability(CustomFactor):  
    inputs = [Fundamentals.interest_income_from_deposits]  
    window_length = QL*9
    window_safe = True
    
    def compute(self, today, assets, out, var):  
        arr = [var[-1],
            var[-QL],
            var[-QL*2],
            var[-QL*3],
            var[-QL*4],
            var[-QL*5],
            var[-QL*6],
           var[-QL*7],
         var[-QL*8]]
        
        arr = np.diff(arr, axis=0)/arr[1:]
        mean = np.mean(arr, axis=0)
        std = np.std(arr, axis=0)
        out[:] = -preprocess(mean/std)
        
   
class NonInrstIncm_growth_mean(CustomFactor):  
    inputs = [Fundamentals.non_interest_income]  
    window_length = QL*9
    window_safe = True
    
    def compute(self, today, assets, out, var):  
        arr = [var[-1],
            var[-QL],
            var[-QL*2],
            var[-QL*3],
            var[-QL*4],
            var[-QL*5],
            var[-QL*6],
           var[-QL*7],
         var[-QL*8]]
        
        arr = np.diff(arr, axis=0)/arr[1:]
        mean = np.mean(arr, axis=0)
        out[:] = -preprocess(mean)

   

def make_factors():
    
    days = BusinessDaysSincePreviousEvent(inputs=[fe.Actuals.slice('SALES', 'qf', 0).asof_date])  
    
    class mean_rev_ind(CustomFactor):   #alpha4
        inputs = [mean_rev(), days]
        window_length = 1
        window_safe = True
        
        def compute(self, today, assets, out, var, days):
            ind = np.zeros(np.shape(days))
            ind[:, ((days[0,:] > N1) & (days[0,:] < N2))] = 1.0
            out[:] = preprocess(var * ind)
            

    class MaxGap_ind(CustomFactor):     #alpha13
        # the biggest absolute overnight gap in the previous 90 sessions
        inputs = [MaxGap(), days]
        window_length = 1    
        window_safe = True
        
        def compute(self, today, assets, out, var, days):
            out[:] = var

    class revenue_growth_mean_ind(CustomFactor):  #alpha76
        inputs = [revenue_growth_mean(), days]  
        window_length = 1
        window_safe = True
        def compute(self, today, assets, out, var, days):  
            out[:] = var
            
    class TA_growth_mean_ind(CustomFactor):  
        inputs = [TA_growth_mean(), days]  
        window_length = 1
        window_safe = True
    
        def compute(self, today, assets, out, var, days):  
            out[:] = var
        

    class GainLoansTa_mean_ind(CustomFactor):  
        inputs = [GainLoansTa_mean(), days]  
        window_length = 1
        window_safe = True
    
        def compute(self, today, assets, out, var, days):  
            out[:] = var
    
    class IncomeDeposits_growth_stability_ind(CustomFactor):  
        inputs = [IncomeDeposits_growth_stability(), days]  
        window_length = 1
        window_safe = True
    
        def compute(self, today, assets, out, var, days):  
            out[:] = var
        
   
    class NonInrstIncm_growth_mean_ind(CustomFactor):  
        inputs = [NonInrstIncm_growth_mean(), days]  
        window_length = 1
        window_safe = True
    
        def compute(self, today, assets, out, var, days):  
            out[:] = var

   
    factors = [
        mean_rev_ind,
        MaxGap_ind,
        revenue_growth_mean_ind,

        NonInrstIncm_growth_mean_ind,
        
        IncomeDeposits_growth_stability_ind,
        GainLoansTa_mean_ind,
        
        TA_growth_mean_ind,

    ]
    
    return factors

class Factor_N_Days_Ago(CustomFactor):

    def compute(self, today, assets, out, input_factor):
        out[:] = input_factor[0]

def factor_pipeline():
    
    factors = make_factors()
    
    sector = RBICSFocus.l1_name.latest
    base_universe = QTradableStocksUS() & sector.eq('Finance')

    pipeline_columns = {}
    for k,f in enumerate(factors):
        for days_ago in range(N_FACTOR_WINDOW):

            pipeline_columns['alpha_'+str(k)+'_'+str(days_ago)] = Factor_N_Days_Ago([f(mask=base_universe)], window_length=days_ago+1, mask=base_universe) 
    
    pipe = Pipeline(columns = pipeline_columns,
    screen = base_universe)
    
    return pipe

def beta_pipeline():
    
    beta = SimpleBeta(target=sid(8554),regression_length=260,
                      allowed_missing_percentage=1.0
                     )
    
    pipe = Pipeline(columns = {'beta': beta},
    screen = QTradableStocksUS())
    return pipe
    
def initialize(context):    
    
    attach_pipeline(risk_loading_pipeline(), 'risk_loading_pipeline')
    attach_pipeline(beta_pipeline(), 'beta_pipeline')
    attach_pipeline(factor_pipeline(), 'factor_pipeline')
    
    # Schedule my rebalance function
    schedule_function(func=rebalance,
                      date_rule=date_rules.every_day(),
                      time_rule=time_rules.market_open(minutes=60),
                      half_days=True)
    # record my portfolio variables at the end of day
    schedule_function(func=recording_statements,
                      date_rule=date_rules.every_day(),
                      time_rule=time_rules.market_close(),
                      half_days=True)
    
    context.init = True
    
    set_commission(commission.PerShare(cost=0, min_trade_cost=0))
    set_slippage(slippage.FixedSlippage(spread=0))

def recording_statements(context, data):
 
    record(num_positions=len(context.portfolio.positions))
    record(leverage=context.account.leverage)
    
def before_trading_start(context, data):
 
    risk_loadings = pipeline_output('risk_loading_pipeline')
    risk_loadings.fillna(risk_loadings.median(), inplace=True)
    context.risk_loadings = risk_loadings
    context.beta_pipeline = pipeline_output('beta_pipeline')
    
    alphas = pipeline_output('factor_pipeline').dropna()
    
    n_factors = len(alphas.columns)/N_FACTOR_WINDOW
    n_stocks = len(alphas.index)
    
    alphas_flattened = np.zeros((n_factors,n_stocks*N_FACTOR_WINDOW))
    
    for f in range(n_factors):
        a = alphas.iloc[:,f*N_FACTOR_WINDOW:(f+1)*N_FACTOR_WINDOW].values
        alphas_flattened[f,:] = np.ravel(a)
    
    clustering = SpectralClustering(n_clusters=3,assign_labels="discretize",random_state=0).fit(alphas_flattened)
    
    weights = np.zeros(n_factors)
    for k,w in enumerate(clustering.labels_):
        weights[k] = Counter(clustering.labels_)[w]
    
    alphas_current = alphas.ix[:,::N_FACTOR_WINDOW]
    
    context.combined_alpha = pd.Series(np.zeros_like(alphas_current.iloc[:,1].values),index=alphas_current.index)
    for k in range(n_factors):
        context.combined_alpha += alphas_current.iloc[:,k]/weights[k]
        
def rebalance(context, data):
    
    alpha = context.combined_alpha
    beta = context.beta_pipeline
    factor_loadings = context.risk_loadings
        
    validSecurities = list(
        set(alpha.index.values.tolist()) & 
        set(beta.index.values.tolist()) &
        set(factor_loadings.index.values.tolist()))
    
    alpha = alpha.loc[validSecurities]
    factor_loadings = factor_loadings.loc[validSecurities, :]
    beta = beta.loc[validSecurities, :]   
    
    #Demean Alpha
    alpha = alpha.sub(alpha.mean())
    try_opt = alpha/alpha.abs().sum()
    try_opt = try_opt.loc[validSecurities]

    
    factor_risk_constraints = opt.experimental.RiskModelExposure(
        factor_loadings,
        version=opt.Newest,
    )
    
    dollar_neutral = opt.DollarNeutral(MAX_NET_EXPOSURE)
    
    beta_neutral = opt.FactorExposure(
        loadings=context.beta_pipeline[['beta']],
        min_exposures={'beta':-MAX_BETA_EXPOSURE},
        max_exposures={'beta':MAX_BETA_EXPOSURE}
        )
    
    algo.order_optimal_portfolio(
            opt.TargetWeights(try_opt),
            constraints = [
                dollar_neutral,
                factor_risk_constraints,
                # beta_neutral
            ]
        )