from quantopian.algorithm import attach_pipeline, pipeline_output
from quantopian.pipeline import Pipeline
from quantopian.pipeline.factors import CustomFactor, SimpleBeta, Returns
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


def preprocess(a):
    
    a = a.astype(np.float64)
    a[np.isinf(a)] = np.nan
    a = np.nan_to_num(a - np.nanmean(a))
    a = winsorize(a, limits=[WIN_LIMIT,WIN_LIMIT])
    
    return preprocessing.scale(a)

def make_factors():
    
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

    factors = [
        mean_rev,
        MaxGap,
        revenue_growth_mean,

        TradingGLTa_stability,
        TotalPremiumTa_stability,
        PolicyDivTa_stability,
        PolicyHldrIntrst_growth_mean,
        NonInrstIncmTa_mean,
        
        NonInrstIncm_growth_mean,
        
        IntrstIncmNonOper_growth_mean,
        IntrstIncmSec_growth_std,
        LeaseIncomeTa_mean,
        IncomeFdrlFundsTa_stability,
        IncomeFdrlFunds_growth_stability,
        
        IncomeDeposits_growth_stability,
        GainLoansTa_mean,
        
        GainLoansTa_stability,
        FetgTa_mean,
        EqtyIntrstEarng_growth_stability,
        DivIncmTa_mean,
        CrCrd_growth_mean,
        CR_growth_stability,
        CRTa_stability,
        TSTA_stability,        
        
        
        TA_growth_mean,
        CashTa_mean,
        CustAccnts_growth_dd,
        FdrlsFundPurchTa_stability
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
            pipeline_columns['alpha_'+str(k)+'_'+str(days_ago)] = Factor_N_Days_Ago([f(mask=QTradableStocksUS())], window_length=days_ago+1, mask=QTradableStocksUS())
    
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