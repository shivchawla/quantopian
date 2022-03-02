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
# from quantopian.pipeline.data.factset.estimates import PeriodicConsensus
import quantopian.pipeline.data.factset.estimates as fe
# from quantopian.pipeline.data import factset
from sklearn import preprocessing
from scipy.stats.mstats import winsorize
from quantopian.pipeline.data.factset import RBICSFocus
# from quantopian.pipeline.classifiers.fundamentals import Sector



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
        out[:] = preprocess(mean)

class CashTa_stability(CustomFactor):  
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
        std = np.std(arr, axis=0)
        out[:] = preprocess(mean/std)

# Current Assets to TA
class CurrentAssetsTa_stability(CustomFactor):  
    inputs = [Fundamentals.current_assets, Fundamentals.total_assets]  
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


        
#current_deferred_taxes_assets  to TA
class CDTATa_mean(CustomFactor):  
    inputs = [Fundamentals.current_deferred_taxes_assets, Fundamentals.total_assets]  
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
        out[:] = -preprocess(mean)
        
        

class CDTATa_stability(CustomFactor):  
    inputs = [Fundamentals.current_deferred_taxes_assets, Fundamentals.total_assets]  
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
        
        
#current_deferred_taxes_liabilities to TA mean
#current_deferred_taxes_liabilities growth mean


class CDTLTa_mean(CustomFactor):  
    inputs = [Fundamentals.current_deferred_taxes_liabilities, Fundamentals.total_assets]  
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
        out[:] = preprocess(mean)


class CDTL_growth_mean(CustomFactor):  
    inputs = [Fundamentals.current_deferred_taxes_liabilities]  
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

        
#current_notes_payable -> (alpha94) 
class CNPTa_mean(CustomFactor):  
    inputs = [Fundamentals.current_notes_payable, Fundamentals.total_assets]  
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
        out[:] = preprocess(mean)



#current_provisions -> (alpha95)
class CPTa_std(CustomFactor):  
    inputs = [Fundamentals.current_notes_payable, Fundamentals.total_assets]  
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
        
        std = np.std(arr, axis=0)
        out[:] = preprocess(-std)


#customer_accounts - (alpha91/alpha92  - sum works (dd))
class CustAccnts_growth_dd(CustomFactor):  
    inputs = [Fundamentals.customer_accounts]  
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
        out[:] = preprocess(mean) + preprocess(-std)

        
#deferred_tax_assets - alpha92
class DTA_growth_std(CustomFactor):  
    inputs = [Fundamentals.deferred_tax_assets]  
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
        
    
#depositsby_bank - alpha91/alpha94
class DepositsBank_growth_mean(CustomFactor):  
    inputs = [Fundamentals.depositsby_bank]  
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


class DepositsBankTa_mean(CustomFactor):  
    inputs = [Fundamentals.depositsby_bank, Fundamentals.total_assets]  
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
        

#federal_funds_purchased - alpha96
class FdrlsFundPurchTa_stability(CustomFactor):  
    inputs = [Fundamentals.federal_funds_purchased, Fundamentals.total_assets]  
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



#federal_funds_sold - alpha91/alpha92
class FdrlFundsSold_growth_mean(CustomFactor):  
    inputs = [Fundamentals.federal_funds_sold]  
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

        
class FdrlFundsSold_growth_std(CustomFactor):  
    inputs = [Fundamentals.federal_funds_sold]  
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
        
        
#foreclosed_assets - (-alpha96)
class ForeclosedAssetsTa_stability(CustomFactor):  
    inputs = [Fundamentals.foreclosed_assets, Fundamentals.total_assets]  
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


#investment_properties - alpha93/alpha94/alpha96
class InvPropTa_mean(CustomFactor):  
    inputs = [Fundamentals.investment_properties, Fundamentals.total_assets]  
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

class InvPropTa_std(CustomFactor):  
    inputs = [Fundamentals.investment_properties, Fundamentals.total_assets]  
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
        
        std = np.std(arr, axis=0)
        out[:] = preprocess(-std)
    
class InvPropTa_stability(CustomFactor):  
    inputs = [Fundamentals.investment_properties, Fundamentals.total_assets]  
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

        
        
#investmentin_financial_assets - alpha93/alpha96
class InvFinAssts_growth_stability(CustomFactor):  
    inputs = [Fundamentals.investmentin_financial_assets]  
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

class InvFinAsstsTa_stability(CustomFactor):  
    inputs = [Fundamentals.investmentin_financial_assets, Fundamentals.total_assets]  
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


#investments_and_advances  - alpha95
class InvAndAdvTa_stability(CustomFactor):  
    inputs = [Fundamentals.investments_and_advances, Fundamentals.total_assets]  
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
        
        std = np.std(arr, axis=0)
        out[:] = preprocess(-std)


#investmentsin_associatesat_cost - alpha95
class InvAssTa_stability(CustomFactor):  
    inputs = [Fundamentals.investmentsin_associatesat_cost, Fundamentals.total_assets]  
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
        
        std = np.std(arr, axis=0)
        out[:] = preprocess(-std)


#investmentsin_subsidiariesat_cost - alpha94/alpha96
class InvSubdrTa_mean(CustomFactor):  
    inputs = [Fundamentals.investmentsin_subsidiariesat_cost, Fundamentals.total_assets]  
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

        
class InvSubdrTa_stability(CustomFactor):  
    inputs = [Fundamentals.investmentsin_subsidiariesat_cost, Fundamentals.total_assets]  
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
        out[:] = preprocess(std/mean)



#investmentsin_joint_venturesat_cost - alpah96
class InvJntVentrTa_stability(CustomFactor):  
    inputs = [Fundamentals.investmentsin_joint_venturesat_cost, Fundamentals.total_assets]  
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
        out[:] = preprocess(std/mean)


#long_term_investments - alpha93/alpha95
class LTI_growth_stability(CustomFactor):  
    inputs = [Fundamentals.long_term_investments]  
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

        
class LTITa_std(CustomFactor):  
    inputs = [Fundamentals.long_term_investments, Fundamentals.total_assets]  
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
        
        std = np.std(arr, axis=0)
        out[:] = preprocess(-std)


#long_term_provisions - alpah92
class LTP_growth_std(CustomFactor):  
    inputs = [Fundamentals.long_term_provisions]  
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



#minority_interest_balance_sheet - alpha93
class MIBS_growth_stability(CustomFactor):  
    inputs = [Fundamentals.minority_interest_balance_sheet]  
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


#money_market_investments - (-alpha91)
class MMI_growth_mean(CustomFactor):  
    inputs = [Fundamentals.money_market_investments]  
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
        

#policyholder_funds - alpha95
class PolicyFndsTa_std(CustomFactor):  
    inputs = [Fundamentals.policyholder_funds, Fundamentals.total_assets]  
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
        
        std = np.std(arr, axis=0)
        out[:] = preprocess(-std)


#properties - alpha94
class PropertiesTa_mean(CustomFactor):  
    inputs = [Fundamentals.properties, Fundamentals.total_assets]  
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
        
#restricted_investments - alpha94
class RestInvTa_mean(CustomFactor):  
    inputs = [Fundamentals.restricted_investments, Fundamentals.total_assets]  
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

#securities_and_investments - alpha91
class SecInv_growth_mean(CustomFactor):  
    inputs = [Fundamentals.securities_and_investments]  
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


#security_borrowed - (-alpha96)
class SecuritiesBorrowedTa_stability(CustomFactor):  
    inputs = [Fundamentals.security_borrowed, Fundamentals.total_assets]  
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


#subordinated_liabilities - alpha91/alpha94
class SubordLiab_growth_mean(CustomFactor):  
    inputs = [Fundamentals.subordinated_liabilities]  
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

class SubordLiabTa_mean(CustomFactor):  
    inputs = [Fundamentals.subordinated_liabilities, Fundamentals.total_assets]  
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

#total_equity - alpha96
class TotEqTa_stability(CustomFactor):  
    inputs = [Fundamentals.total_equity, Fundamentals.total_assets]  
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


#total_non_current_assets - (-alpha93)
class TotNCA_growth_stability(CustomFactor):  
    inputs = [Fundamentals.total_non_current_assets]  
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

class TrdngLiabTa_stability(CustomFactor):  
    inputs = [Fundamentals.trading_liabilities, Fundamentals.total_assets]  
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
        

class mean_rev(CustomFactor):   
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

        
class MaxGap(CustomFactor): 
    # the biggest absolute overnight gap in the previous 90 sessions
    inputs = [USEquityPricing.close] ; window_length = 90
    window_safe = True
    def compute(self, today, assets, out, close):
        abs_log_rets = np.abs(np.diff(np.log(close),axis=0))
        max_gap = np.max(abs_log_rets, axis=0)
        out[:] = preprocess(max_gap)

        
class revenue_growth_mean(CustomFactor):  
    inputs = [Fundamentals.total_revenue]  
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
        
    
def make_pipeline():
    """
    A function to create our dynamic stock selector (pipeline). Documentation
    on pipeline can be found here:
    https://www.quantopian.com/help#pipeline-title
    """
    # Base universe set to the QTradableStocksUS
    sector = RBICSFocus.l1_name.latest
    # sector = Fundamentals.morningstar_sector_code
    
    # base_universe = QTradableStocksUS() & Sector().element_of( [103, 104])
    base_universe = QTradableStocksUS() & sector.eq('Finance')

    #Other Factors
    f1a = -1 * CashTa_mean(mask = base_universe)
    f1b = CashTa_stability(mask = base_universe)
    f2 = CurrentAssetsTa_stability(mask = base_universe)
    f3a = CDTATa_mean(mask = base_universe)      
    f3b = CDTATa_stability(mask = base_universe)
    f4a = CDTLTa_mean(mask = base_universe)
    f4b = CDTL_growth_mean(mask = base_universe)
    f5 = -CNPTa_mean(mask = base_universe)
    f6 = CPTa_std(mask = base_universe)
    f7 = CustAccnts_growth_dd(mask = base_universe)
    f8 = DTA_growth_std(mask = base_universe)
    f9a = DepositsBank_growth_mean(mask = base_universe)
    f9b = DepositsBankTa_mean(mask = base_universe)
    f10 = FdrlsFundPurchTa_stability(mask = base_universe)
    f11a = -FdrlFundsSold_growth_mean(mask = base_universe)
    f11b = FdrlFundsSold_growth_std(mask = base_universe)
    f12 = ForeclosedAssetsTa_stability(mask = base_universe)
    f13a = InvPropTa_mean(mask = base_universe)
    f13b = -InvPropTa_std(mask = base_universe)
    f13c = InvPropTa_stability(mask = base_universe)
    f14a = InvFinAssts_growth_stability(mask = base_universe)
    f14b = -InvFinAsstsTa_stability(mask = base_universe)
    f15 = InvAndAdvTa_stability(mask = base_universe)
    f16 = InvAssTa_stability(mask = base_universe)
    f17a = InvSubdrTa_mean(mask = base_universe)
    f17b = InvSubdrTa_stability(mask = base_universe)
    f18 = InvJntVentrTa_stability(mask = base_universe)
    f19a = LTI_growth_stability(mask = base_universe)
    f19b = LTITa_std(mask = base_universe)
    f20 = LTP_growth_std(mask = base_universe)
    f21 = MIBS_growth_stability(mask = base_universe)
    f22 = MMI_growth_mean(mask = base_universe)
    f23 = PolicyFndsTa_std(mask = base_universe)
    f24 = PropertiesTa_mean(mask = base_universe)
    f25 = RestInvTa_mean(mask = base_universe)
    f26 = SecInv_growth_mean(mask = base_universe)
    f27 = -SecuritiesBorrowedTa_stability(mask = base_universe)
    f28a = SubordLiab_growth_mean(mask = base_universe)
    f28b = SubordLiabTa_mean(mask = base_universe)
    f29 = TotEqTa_stability(mask = base_universe)
    f30 = -TotNCA_growth_stability(mask = base_universe)
    f31 = -TrdngLiabTa_stability(mask = base_universe)
    
    
    f86 = TA_growth_mean(mask = base_universe) #1
    
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
    
    f113 = mean_rev(mask = base_universe)
    f114 = MaxGap(mask = base_universe)
    f115 = revenue_growth_mean(mask = base_universe)
    

    guidance_days_prev = BusinessDaysSincePreviousEvent(inputs=[fe.Actuals.slice('SALES', 'qf', 0).asof_date])  
    
    pipe_screen = base_universe 

    pipe_columns = {
              'f1a':f1a,
                'f3b':f3b,
                 'f4b':f4b,
                'f5':f5,
                'f7':f7,
                'f9a':f9a,
                'f10':f10,
                'f11a':f11a,
                'f12':f12,
                'f13a':f13a,
                'f13c':f13c,
                'f14a':f14a,
                'f14b':f14b,
                'f15':f15,
                'f16':f16,
                'f17a':f17a,
                'f19a':f19a,
                'f19b':f19b,
                'f21':f21,
                'f27':f27,
                'f28b':f28b,
                'f30':f30,
        
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
        
        'f113':f113,
        'f114':f114,
        'f115':f115,

        'guidance_days_prev': guidance_days_prev
      
    }

    return Pipeline(columns=pipe_columns, screen=pipe_screen)


def transform(df, field, multiplier=1):
    return signalize(multiplier*df[field])

def weightedMean(df, weights):
    return df.mean()

def before_trading_start(context, data):
    output = algo.pipeline_output('pipeline')
    
    
    output['ind'] = 0
    
    output.loc[(output['guidance_days_prev'] > N1) & (output['guidance_days_prev'] < N2), 'ind'] = 1.0

    # alpha1 = output['f86'] + output['f88'] + output['f90'] + output['f91'] + output['f95'] + output['f97'] + output['f98'] + output['f99'] + output['f100'] + output['f101'] + output['f102'] + output['f103'] + output['f104'] + output['f105'] + output['f106'] + output['f107'] + output['f108'] + output['f109'] + output['f110'] + output['f111'] + output['f112']
        
    
    # alpha2 = output['f1a'] + output['f3b'] +  output['f4b'] +  output['f5'] + output['f7'] +  output['f9a'] +  output['f10'] +  output['f11a'] +  output['f12'] + output['f13a'] +  output['f13c'] +  output['f14a'] +  output['f14b'] + output['f15'] + output['f16'] +  output['f17a'] +  output['f19a'] +  output['f19b'] + output['f21'] +  output['f27'] +  output['f28b'] + output['f30'] 
    
    
    alpha = output['f113'] + output['f114'] + output['f115'] + output['f86'] + output['f99'] + output['f101'] + output['f108'] 
    
    
    # + output['f1a'] + output['f7'] + output['f10'] 
    

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
        
        # # Ensure long and short books
        # # are roughly the same size
        dollar_neutral = opt.DollarNeutral(context.max_net_exposure)
        
        factor_risk_constraints = opt.experimental.RiskModelExposure(
            context.risk_factor_betas,
            
            min_real_estate = -0.05,
            max_real_estate = 0.05,
            min_financial_services = -0.05,
            max_financial_services = 0.05,
            version=opt.Newest
        )
        
        beta_neutral = opt.FactorExposure(
            beta[['mkt_beta']],
            min_exposures={'mkt_beta': -context.max_beta},
            max_exposures={'mkt_beta': context.max_beta},
        )
        
        algo.order_optimal_portfolio(
            opt.TargetWeights(try_opt),
            constraints = [
                dollar_neutral,
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