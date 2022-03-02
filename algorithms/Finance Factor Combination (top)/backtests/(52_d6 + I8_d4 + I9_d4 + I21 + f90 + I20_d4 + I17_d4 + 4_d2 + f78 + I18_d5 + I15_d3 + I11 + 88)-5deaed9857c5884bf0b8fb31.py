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
from quantopian.pipeline.factors import SimpleBeta, Returns, PercentChange, BusinessDaysSincePreviousEvent
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
from quantopian.pipeline.data.factset import RBICSFocus


WIN_LIMIT = 0.00
QL = 66

def preprocess(a):
    
    a = a.astype(np.float64)
    a[np.isinf(a)] = np.nan
    a = np.nan_to_num(a - np.nanmean(a))
    a = winsorize(a, limits=WIN_LIMIT)
    
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
    context.max_pos_size = 0.01
    context.max_turnover = 0.95
    context.max_beta = 0.05
    context.max_net_exposure = 0.001
    context.max_volatility = 0.05
    context.max_sector_exposure = 1.0
    context.max_style_exposure = 1.0
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
            

class Gross_Income_Margin(CustomFactor): #alpha12
    #Gross Income Margin:
    #Gross Profit divided by Net Sales
    #Notes:
    #High value suggests that the company is generating large profits
    inputs = [Fundamentals.cost_of_revenue, Fundamentals.total_revenue]
    window_length = 1
    window_safe = True
    def compute(self, today, assets, out, cost_of_revenue, sales):
        gross_income_margin = sales[-1]/sales[-1] - cost_of_revenue[-1]/sales[-1]
        out[:] = preprocess(-gross_income_margin)
            
            
            
class MaxGap(CustomFactor):     #alpha13
    # the biggest absolute overnight gap in the previous 90 sessions
    inputs = [USEquityPricing.close] ; window_length = 90
    window_safe = True
    def compute(self, today, assets, out, close):
        abs_log_rets = np.abs(np.diff(np.log(close),axis=0))
        max_gap = np.max(abs_log_rets, axis=0)
        out[:] = preprocess(max_gap)

class FCFTA(CustomFactor):  #alpha25
    inputs = [Fundamentals.free_cash_flow,  
             Fundamentals.total_assets]  
    window_length = 1
    window_safe = True
    def compute(self, today, assets, out, fcf, ta):  
        out[:] = preprocess(np.where(fcf[-1]/ta[-1]>0,1,0))

        

class growthscorestability(CustomFactor): #Alpha44
    inputs = [Fundamentals.growth_score]
    window_length = QL*8
    window_safe = True
    def compute(self, today, assets, out, var):
        std = np.std([var[-1], 
            var[-QL], 
            var[-QL*2],
            var[-QL*3],
            var[-QL*4],
            var[-QL*5],
            var[-QL*6],
            var[-QL*7]], axis=0)

        out[:] = preprocess(-std)

class fcf_growth_mean(CustomFactor): #alpha52
    inputs=[
        Fundamentals.fcf_per_share,
        Fundamentals.shares_outstanding,
        Fundamentals.enterprise_value,]
    window_length = QL*8
    window_safe = True
    def compute(self, today, assets, out, fcf, shares, ev):
        var = fcf*shares
        var[np.isinf(var)] = np.nan
        
        arr = [var[-1]/var[-QL] -1, 
            var[-QL]/var[-2*QL] -1,
            var[-QL*2]/var[-3*QL] -1, 
            var[-QL*3]/var[-4*QL] -1, 
            var[-QL*4]/var[-5*QL] -1, 
            var[-QL*5]/var[-6*QL] -1, 
            var[-QL*6]/var[-7*QL] -1]

        mean = np.mean(arr, axis=0)
        out[:] = preprocess(mean)
        

class STA_Stability(CustomFactor):  #alpha59
    inputs = [Fundamentals.operating_cash_flow,  
              Fundamentals.net_income_continuous_operations,  
              Fundamentals.total_assets]  
    window_length = QL*8
    window_safe = True
    def compute(self, today, assets, out, ocf, ni, ta):  
        ta = np.where(np.isnan(ta), 0, ta)  
        ocf = np.where(np.isnan(ocf), 0, ocf)  
        ni = np.where(np.isnan(ni), 0, ni)  
        var = abs(ni - ocf)/ ta
        
        arr = [var[-1],
            var[-QL],
            var[-QL*2],
            var[-QL*3],
            var[-QL*4],
            var[-QL*5],
            var[-QL*6]]

        std = np.std(arr, axis=0)
        mean = np.mean(arr, axis=0)
        
        out[:] = preprocess(mean/std)

        
        
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

class TEM_GROWTH(CustomFactor): #alpha54
    """
    TEM = standard deviation of past 6 quarters' reports
    """
    inputs=[factset.Fundamentals.capex_qf_asof_date,
        factset.Fundamentals.capex_qf,
        factset.Fundamentals.assets]
    window_length = QL*8
    window_safe = True
    def compute(self, today, assets, out, asof_date, capex, total_assets):
        var = capex/total_assets
        var[np.isinf(var)] = np.nan
        arr = [var[-1]/var[-QL] -1, 
            var[-QL]/var[-2*QL] -1,
            var[-QL*2]/var[-3*QL] -1, 
            var[-QL*3]/var[-4*QL] -1, 
            var[-QL*4]/var[-5*QL] -1, 
            var[-QL*5]/var[-6*QL] -1, 
            var[-QL*6]/var[-7*QL] -1]

        std = np.std(arr, axis=0)
        out[:] = preprocess(-std)

        
        
class revenue_growth_stability(CustomFactor):  #alpha78
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

        std = np.std(arr, axis=0)
        mean = np.mean(arr, axis=0)
        
        out[:] = preprocess(mean/std)
        
        
        
class Current_Ratio_Mean(CustomFactor): #alpha65
    inputs=[Fundamentals.current_ratio]  
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
        out[:] = -preprocess(mean)
        

class pcf_ratio_stability(CustomFactor): #alpha71
    inputs = [Fundamentals.pcf_ratio]
    window_length = QL*8
    window_safe = True
    def compute(self, today, assets, out, var):
        std = np.std([1/var[-1], 
            1/var[-QL], 
            1/var[-QL*2],
            1/var[-QL*3],
            1/var[-QL*4],
            1/var[-QL*5],
            1/var[-QL*6],
            1/var[-QL*7]], axis=0)

        out[:] = -preprocess(std)
        

class ps_ratio_stability(CustomFactor): #alpha72
    inputs = [Fundamentals.ps_ratio]
    window_length = QL*8
    window_safe = True
    def compute(self, today, assets, out, var):
        std = np.std([1/var[-1], 
            1/var[-QL], 
            1/var[-QL*2],
            1/var[-QL*3],
            1/var[-QL*4],
            1/var[-QL*5],
            1/var[-QL*6],
            1/var[-QL*7]], axis=0)

        out[:] = -preprocess(std)

        
        
class pretax_margin_stability(CustomFactor):  #alpha84
    inputs = [Fundamentals.pretax_margin]  
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

        std = np.std(arr, axis=0)
        mean = np.mean(arr, axis=0)
        
        out[:] = -preprocess(mean/std)

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
        
        
class TotalInv_stability(CustomFactor):   # Why negative contribution?
    inputs = [Fundamentals.total_investments]  
    window_length = QL*9
    window_safe = True
    
    def compute(self, today, assets, out, ta):  
        var = ta
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
        
        
class TSTA_stability(CustomFactor):  
    inputs = [Fundamentals.trading_securities, Fundamentals.total_assets]  
    window_length = QL*9
    window_safe = True
    
    def compute(self, today, assets, out, ts, ta):  
        var  = ts/ta
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
        std = np.std(arr, axis=0)
        out[:] = preprocess(mean/std)
        
        
        
class CRTa_stability(CustomFactor):  
    inputs = [Fundamentals.cost_of_revenue, Fundamentals.total_assets]  
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
        std = np.std(arr, axis=0)
        out[:] = -preprocess(mean/std)
        

class CR_growth_stability(CustomFactor):  
    inputs = [Fundamentals.cost_of_revenue]  
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
        
        
class CrCrd_growth_mean(CustomFactor):  
    inputs = [Fundamentals.credit_card]  
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
        
        
        
class DivIncmTa_mean(CustomFactor):  
    inputs = [Fundamentals.dividend_income, Fundamentals.total_assets]  
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
        
        
class EqtyIntrstEarng_growth_stability(CustomFactor):  
    inputs = [Fundamentals.earningsfrom_equity_interest_net_of_tax]  
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
        out[:] = preprocess(mean/std)
        
        
        
class FetgTa_mean(CustomFactor):  
    inputs = [Fundamentals.foreign_exchange_trading_gains, Fundamentals.total_assets]  
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

    
    
class GainLoansTa_stability(CustomFactor):  
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
        std = np.std(arr, axis=0)
        out[:] = preprocess(mean/std)
        
        
        
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
        
        
        
class IncomeFdrlFunds_growth_stability(CustomFactor):  
    inputs = [Fundamentals.interest_income_from_federal_funds_sold_and_securities_purchase_under_agreements_to_resell]  
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
        out[:] = preprocess(mean/std)

        
        
class IncomeFdrlFundsTa_stability(CustomFactor):  
    inputs = [Fundamentals.interest_income_from_federal_funds_sold_and_securities_purchase_under_agreements_to_resell, Fundamentals.total_assets]  
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
        std = np.std(arr, axis=0)
        out[:] = -preprocess(mean/std)

        
        
class LeaseIncomeTa_mean(CustomFactor):  
    inputs = [Fundamentals.interest_income_from_leases, Fundamentals.total_assets]  
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
        
        
class IntrstIncmSec_growth_std(CustomFactor):  
    inputs = [Fundamentals.interest_income_from_securities]  
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
        std = np.std(arr, axis=0)
        out[:] = preprocess(-std)
        
        
class IntrstIncmNonOper_growth_mean(CustomFactor):  
    inputs = [Fundamentals.interest_income_non_operating]  
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

class NonInrstIncmTa_mean(CustomFactor):  
    inputs = [Fundamentals.non_interest_income, Fundamentals.total_assets]  
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
        
        
class PolicyHldrIntrst_growth_mean(CustomFactor):  
    inputs = [Fundamentals.policyholder_interest]  
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
        
        
class PolicyDivTa_stability(CustomFactor):  
    inputs = [Fundamentals.policyholder_dividends, Fundamentals.total_assets]  
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
        std = np.std(arr, axis=0)
        out[:] = preprocess(mean/std)
        
        
        
class TotalPremiumTa_stability(CustomFactor):  
    inputs = [Fundamentals.total_premiums_earned, Fundamentals.total_assets]  
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
        std = np.std(arr, axis=0)
        out[:] = preprocess(mean/std)
        
        
class TradingGLTa_stability(CustomFactor):  
    inputs = [Fundamentals.trading_gain_loss, Fundamentals.total_assets]  
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
        std = np.std(arr, axis=0)
        out[:] = -preprocess(mean/std)
       
    
class FnCTa_stability(CustomFactor):  
    inputs = [Fundamentals.fees_and_commissions, Fundamentals.total_assets]  
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
        std = np.std(arr, axis=0)
        out[:] = preprocess(mean/std)
        
        
class CashTa_mean(CustomFactor):  
    inputs = [Fundamentals.cash, Fundamentals.total_assets]  
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
        
        
        
        
def make_pipeline():
    """
    A function to create our dynamic stock selector (pipeline). Documentation
    on pipeline can be found here:
    https://www.quantopian.com/help#pipeline-title
    """
    # Base universe set to the QTradableStocksUS
    sector = RBICSFocus.l1_name.latest
    base_universe = QTradableStocksUS() & sector.eq('Finance')

    #Other Factors
    f4 = mean_rev(mask = base_universe)        
    f12 = Gross_Income_Margin(mask = base_universe)        
    f13 = MaxGap(mask = base_universe)        
    f25 = FCFTA(mask = base_universe)        
    f44 = growthscorestability(mask = base_universe)
    f52 = fcf_growth_mean(mask = base_universe)
    f54 = TEM_GROWTH(mask = base_universe)
    f59 = STA_Stability(mask = base_universe)
    f65 = Current_Ratio_Mean(mask = base_universe)
    f71 = pcf_ratio_stability(mask = base_universe)
    f72 = ps_ratio_stability(mask = base_universe)
    f76 = revenue_growth_mean(mask = base_universe)
    f78 = revenue_growth_stability(mask = base_universe)
    f84 = pretax_margin_stability(mask = base_universe)
    f86 = TA_growth_mean(mask = base_universe)
    
    f88 = TotalInv_stability(mask = base_universe)

    f90 = TSTA_stability(mask = base_universe)
    
    f91 = CR_growth_stability(mask = base_universe)
    f92 = CRTa_stability(mask = base_universe)
    f93 = CrCrd_growth_mean(mask = base_universe)
    f94 = DivIncmTa_mean(mask = base_universe)
    f95 = EqtyIntrstEarng_growth_stability(mask = base_universe)
    
    f97 = FnCTa_stability(mask = base_universe)
        
    f98 = FetgTa_mean(mask = base_universe)
    
    f99 = GainLoansTa_mean(mask = base_universe)
    f100 = GainLoansTa_stability(mask = base_universe)
    f101 = IncomeDeposits_growth_stability(mask = base_universe)
    
    f102 = IncomeFdrlFunds_growth_stability(mask = base_universe)
    f103 = IncomeFdrlFundsTa_stability(mask = base_universe)
    
    f104 = LeaseIncomeTa_mean(mask = base_universe)

    f105 = IntrstIncmSec_growth_std(mask = base_universe)
    f106 = IntrstIncmNonOper_growth_mean(mask = base_universe)
    
    f107 = NonInrstIncmTa_mean(mask = base_universe)
    f108 = NonInrstIncm_growth_mean(mask = base_universe)
    
    f109 = PolicyHldrIntrst_growth_mean(mask = base_universe)
    f110 = PolicyDivTa_stability(mask = base_universe)
    
    f111 = TotalPremiumTa_stability(mask = base_universe) 
    f112 = TradingGLTa_stability(mask = base_universe)
    
    f113 = CashTa_mean(mask = base_universe)
    
    guidance_days_prev = BusinessDaysSincePreviousEvent(inputs=[fe.Actuals.slice('SALES', 'qf', 0).asof_date])  
    
    pipe_screen = base_universe 

    pipe_columns = {
        'f4':f4,
        'f12':f12,
        'f13':f13,
        'f25':f25,
        'f44':f44,
        'f52':f52,
        'f54':f54,
        'f59':f59,
        'f65':f65,
        'f71':f71,
        'f72':f72,
        'f76':f76,
        'f78':f78,
        'f84':f84,
        'f86':f86,
        
        'f88': f88,
        
        'f90':f90,
        
        'f91': f91,
        'f92': f92,
        'f93': f93,
        'f94': f94,
        'f95': f95,
        
        'f97': f97,
               
        'f98': f98,
        
        'f99': f99,
        'f100': f100,
        'f101': f101,
        
        'f102': f102,
        'f103': f103,
        'f104': f104,
        'f105': f105,
        'f106': f106,
        
        'f107': f107,
        'f108': f108,
        
        'f109': f109,
        'f110': f110,
        
        'f111': f111,
        'f112': f112,
        
        'f113': f113,

        'guidance_days_prev': guidance_days_prev
      
    }

    return Pipeline(columns=pipe_columns, screen=pipe_screen)


def transform(df, field, multiplier=1):
    return signalize(multiplier*df[field])

def weightedMean(df, weights):
    return df.mean()

def before_trading_start(context, data):
    output = algo.pipeline_output('pipeline')
    
    alpha_k6 = output['f12'] + output['f13'] + output['f25'] + output['f44'] + output['f59'] + output['f76'] + output['f90'] + output['f78'] + output['f92'] + output['f93']
    
    alpha_4 = output['f4']
    alpha_52 = output['f52']
    alpha_54 = output['f54']

    alpha_86 = output['f86']
    alpha_88 = output['f88']
    
    alpha_90 = output['f90']
 
    alphaI1 = output['f91']
    
    alphaI4 = output['f94']
    alphaI5 = output['f95']
    
    alphaI7 = output['f97']

    alphaI8 = output['f98']
    
    alphaI9 = output['f99']
    alphaI10 = output['f100']
    alphaI11 = output['f101']
    
    alphaI12 = output['f102']
    alphaI13 = output['f103']
    
    alphaI14 = output['f104']
    alphaI15 = output['f105']
    alphaI16 = output['f106']

    alphaI17 = output['f107']
    alphaI18 = output['f108']
    alphaI19 = output['f109']
    alphaI20 = output['f110']
    
    alphaI21 = output['f111']
    alphaI22 = output['f112']
    
    alphaI23 = output['f113']
    
    alpha_s3 = output['f65'] + output['f71'] + output['f72'] + output['f84']
    
    output['d1'] = 0
    output['d2'] = 0
    output['d3'] = 0
    output['d4'] = 0
    output['d5'] = 0
    output['d6'] = 0
    
    N1 = 10
    N2 = 50
    output.loc[(output['guidance_days_prev'] < N1), 'd1'] = 1.0
    output.loc[(output['guidance_days_prev'] > N1) & (output['guidance_days_prev'] < N2), 'd2'] = 1.0
    output.loc[output['guidance_days_prev'] > N2, 'd3'] = 1.0
    output.loc[(output['d2'] == 1) | (output['d3'] == 1), 'd4'] = 1                 
    output.loc[(output['d1'] == 1) | (output['d2'] == 1), 'd5'] = 1.0
    output.loc[(output['d1'] == 1) | (output['d3'] == 1), 'd6'] = 1.0

    # alpha = alpha_k6 + alpha_54*output['d1'] + alpha_4*output['d2'] + (alpha_54 + alpha_78)*output['d3'] + (alpha_s3 + alpha_52)*output['d4'] + alpha_86 + alpha_90

    # alpha =  alpha_4 + alpha_52 + alpha_s3 + alpha_k6 + alpha_54 + alpha_86 + alpha_91 + alpha_94 + alpha_95 + alpha_98 + alpha_99 + alpha_100 + alpha_101 + alpha_102 + alpha_103 + alpha_105 + alpha_106 + alpha_107 + alpha_108 + alpha_109 + alpha_110 + alpha_111 + alpha_112
    
    alpha_4_d2 = alpha_4 * output['d2']
    alpha_52_d6 = alpha_52 * output['d6']
    alpha_54_d6 = alpha_54 * output['d6']
    alpha_s3_d4 = alpha_s3 * output['d4']
    alpha_86_d2 = alpha_86 * output['d2']
    alpha_90_d4 = alpha_90 * output['d4']
    # alphaI4_d2 = alphaI4 * output['d2'] 
    alphaI5_d6 = alphaI5 * output['d6']
    alphaI7_d1 = alphaI7 * output['d1']
    alphaI8_d4 = alphaI8 * output['d4']
    alphaI9_d4 = alphaI9 * output['d4']
    alphaI10_d3 = alphaI10 * output['d3']
    alphaI12_d4 = alphaI12 * output['d4']
    alphaI13_d4 = alphaI13 * output['d2']
    alphaI14_d3 = alphaI14 * output['d3']
    alphaI15_d3 = alphaI15 * output['d3']
    alphaI16_d1 = alphaI16 * output['d1']
    alphaI17_d4 = alphaI17 * output['d4']
    alphaI18_d5 = alphaI18 * output['d5']
    alphaI19_d1 = alphaI19 * output['d1']
    alphaI20_d4 = alphaI20 * output['d4'] #d2
    alphaI22_d2 = alphaI22 * output['d2']
    alphaI23_d1 = alphaI23 * output['d1']

    alpha = (alpha_52_d6 + alphaI8_d4 + alphaI9_d4 + alphaI21 + output['f90'] + alphaI20_d4 + alphaI17_d4 + alpha_4_d2 + output['f78'] + alphaI18_d5 + alphaI15_d3 + alphaI11 + alpha_88)
    
    print (alpha_52_d6 + alphaI8_d4 + alphaI9_d4 + alphaI21 + output['f90'] + alphaI20_d4 + alphaI17_d4 + alpha_4_d2 + output['f78'] + alphaI18_d5 + alphaI15_d3 + alphaI11).corr(alpha_88)
    
    # alpha_4_d2 + alpha_86_d2 +  alphaI1 + alphaI5_d6 +  alphaI8_d4 +  alphaI9_d4 +  alphaI11 + alphaI12_d4 +  alphaI13_d4  + alphaI15_d3 + alphaI17_d4 +  alphaI18_d5  +  alphaI20_d4 +  alphaI21 + alphaI22_d2
    
    #2nd removed general factors
    #alpha_52_d6 
    #alpha_54_d6 
    #alphaI14_d3 
    
    #1st removed d1/d3 based factors 
    # alphaI19_d1 + alphaI10_d3 + alphaI16_d1 + alpha_k6                                   
    alpha2 = alpha + alpha_k6 + alpha_s3_d4
    
    # print alpha.corr(alpha2)

#     alpha = alpha_86_d2 + alpha_88 + alpha_90_d4 + alphaI1 +  \
# alphaI5_d6 +  alphaI7_d1 + alphaI8_d4 + alphaI9_d4 +  alphaI10_d3 +  \
# alphaI11 +  alphaI12_d4 +  alphaI13_d4 +  alphaI14_d3 + alphaI15_d3 + \
# alphaI16_d1 + alphaI17_d4 +  alphaI18_d5 +  alphaI19_d1 +  alphaI20_d4 + \
# alphaI21 + alphaI22_d2 + alphaI23                                    

    output['signal_rank'] = alpha 
    context.alpha = output['signal_rank']
    
    context.risk_factor_betas = algo.pipeline_output(
      'risk_pipe'
    ).dropna()
    
    context.beta_pipeline = algo.pipeline_output('beta_pipe')
    
    
def rebalance(context, data):
    
    alpha = context.alpha
    beta = context.beta_pipeline
    factor_loadings = context.risk_factor_betas
     
    validSecurities = list(
        set(alpha.index.values.tolist()) & 
        set(beta.index.values.tolist()) &
        set(factor_loadings.index.values.tolist()))
    
    alpha = alpha.loc[validSecurities]
    factor_loadings = factor_loadings.loc[validSecurities, :]
    
    beta = beta.loc[validSecurities, :]   
                        
    if not alpha.empty:
        
        alpha = alpha.sub(alpha.mean())
        try_opt = alpha/alpha.abs().sum()
        try_opt = try_opt.loc[validSecurities]
        
        # Running optimization implementation (testing)         
        # algo.order_optimal_portfolio(
        #     opt.TargetWeights(try_opt),
        #     constraints = [])
                
        # min_weights = context.min_weights.copy(deep = True)
        # max_weights = context.max_weights.copy(deep = True)

        # constrain_pos_size = opt.PositionConcentration(
        #     min_weights,
        #     max_weights
        # )

        # max_leverage = opt.MaxGrossExposure(context.max_leverage)

        # # Ensure long and short books
        # # are roughly the same size
        dollar_neutral = opt.DollarNeutral(context.max_net_exposure)
        
        # # Constrain portfolio turnover
        # max_turnover = opt.MaxTurnover(context.max_turnover)
        
        factor_risk_constraints = opt.experimental.RiskModelExposure(
            context.risk_factor_betas,
            
            # min_volatility = -0.1,
            # max_volatility = 0.1,
            # min_size = -0.1,
            # max_size = 0.1,
            # min_short_term_reversal = -0.1,
            # max_short_term_reversal = 0.1,
            # min_value = -0.1,
            # max_value = 0.1,
            # min_momentum = -0.1,
            # max_momentum = 0.1,
            
            # min_energy = -0.01,
            # max_energy = 0.01,
            # min_industrials = -0.01,
            # max_industrials = 0.01,
            # min_health_care = -0.01,
            # max_health_care = 0.01,
            min_real_estate = -0.05,
            max_real_estate = 0.05,
            # min_consumer_cyclical = -0.01,
            # max_consumer_cyclical = 0.01,
            # min_technology = -0.01,
            # max_technology = 0.01,
            # min_basic_materials = -0.01,
            # max_basic_materials = 0.01,
            # min_consumer_defensive = -0.01,
            # max_consumer_defensive = 0.01,
            min_financial_services = -0.05,
            max_financial_services = 0.05,
            # min_communication_services = -0.01,
            # max_communication_services = 0.01,
            version=opt.Newest
        )
        
        beta_neutral = opt.FactorExposure(
            beta[['mkt_beta']],
            min_exposures={'mkt_beta': -context.max_beta},
            max_exposures={'mkt_beta': context.max_beta},
        )
        
        # algo.order_optimal_portfolio(
        #     opt.TargetWeights(my_opt),
        #     constraints = [
        #         constrain_pos_size,
        #         max_leverage,
        #         dollar_neutral,
        #         max_turnover,
        #         factor_risk_constraints,
        #         beta_neutral
        #     ]
        # )
        
        algo.order_optimal_portfolio(
            opt.TargetWeights(try_opt),
            constraints = [
                # constrain_pos_size,
                # max_leverage,
                dollar_neutral,
                # max_turnover,
                factor_risk_constraints,
                # beta_neutral
            ]
        )
       
        
def record_vars(context, data):
    """
    Plot variables at the end of each day.
    """
    record("Num Positions", len(context.portfolio.positions))
    pass

def handle_data(context, data):
    """
    Called every minute.
    """
    pass