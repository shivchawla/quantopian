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


class FCFStability(CustomFactor): #Alpha43
    inputs = [Fundamentals.fcf_yield]
    window_length = QL*8
    window_safe = True
    def compute(self, today, assets, out, fcf_yield):
        std = np.std([fcf_yield[-1], 
            fcf_yield[-QL], 
            fcf_yield[-QL*2],
            fcf_yield[-QL*3],
            fcf_yield[-QL*4],
            fcf_yield[-QL*5],
            fcf_yield[-QL*6],
            fcf_yield[-QL*7]], axis=0)

        out[:] = preprocess(-std)
        
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

class SalesGrowthStability(CustomFactor): #alpha46
    inputs = [factset.Fundamentals.sales_gr_qf]
    window_length = QL*8
    window_safe = True
    def compute(self, today, assets, out, var):
        mean = np.mean([var[-1], 
            var[-QL], 
            var[-QL*2],
            var[-QL*3],
            var[-QL*4],
            var[-QL*5],
            var[-QL*6],
            var[-QL*7]], axis=0)

        std = np.std([var[-1], 
            var[-QL], 
            var[-QL*2],
            var[-QL*3],
            var[-QL*4],
            var[-QL*5],
            var[-QL*6],
            var[-QL*7]], axis=0)
        
        out[:] = preprocess(mean/std)

        
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

class STA_Mean(CustomFactor):  #alpha60
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

        mean = np.mean(arr, axis=0)
        out[:] = preprocess(mean)

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
        out[:] = preprocess(mean)
        

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

        out[:] = preprocess(std)
        

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

        out[:] = preprocess(std)

        
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
        
        out[:] = preprocess(mean/std)

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
        
        
        
        
        
def make_pipeline():
    """
    A function to create our dynamic stock selector (pipeline). Documentation
    on pipeline can be found here:
    https://www.quantopian.com/help#pipeline-title
    """
    # Base universe set to the QTradableStocksUS
    sector = RBICSFocus.l1_name.latest
    base_universe = QTradableStocksUS() & sector.eq('Finance')

    mkt_cap = MarketCap(mask = base_universe)
    log_mkt_cap = LogMarketCap(mask = base_universe)
    vol = Volatility(mask = base_universe)
    adv = AverageDollarVolume(window_length=66, mask = base_universe)
    
    #Other Factors
    f4 = mean_rev(mask = base_universe)        
    f12 = Gross_Income_Margin(mask = base_universe)        
    f13 = MaxGap(mask = base_universe)        
    f25 = FCFTA(mask = base_universe)        
    f43 = FCFStability(mask = base_universe)
    f44 = growthscorestability(mask = base_universe)
    f46 = SalesGrowthStability(mask = base_universe)
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

    f90 = TSTA_stability(mask = base_universe)
    
    f91 = CR_growth_stability(mask = base_universe)
    f92 = CRTa_stability(mask = base_universe)
    f93 = CrCrd_growth_mean(mask = base_universe)
    f94 = DivIncmTa_mean(mask = base_universe)
    f95 = EqtyIntrstEarng_growth_stability(mask = base_universe)

    
    guidance_days_prev = BusinessDaysSincePreviousEvent(inputs=[fe.Actuals.slice('SALES', 'qf', 0).asof_date])  

    
    pipe_screen = base_universe 

    pipe_columns = {
        'mkt_cap': mkt_cap,
        'log_mkt_cap': log_mkt_cap,
        'vol': vol,
        'adv': adv,
        'f4':f4,
        'f12':f12,
        'f13':f13,
        'f25':f25,
        'f43':f43,
        'f44':f44,
        'f46':f46,
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
        
        'f90':f90,
        
        'f91': f91,
        'f92': f92,
        'f93': f93,
        'f94': f94,
        'f95': f95,
        
        'guidance_days_prev': guidance_days_prev
      
    }

    return Pipeline(columns=pipe_columns, screen=pipe_screen)


def transform(df, field, multiplier=1):
    return signalize(multiplier*df[field])

def weightedMean(df, weights):
    return df.mean()

def before_trading_start(context, data):
    output = algo.pipeline_output('pipeline')
    # mkt_cap = output['mkt_cap']
    vol = output['vol']
    adv = output['adv']
    # mk_rank = transform(output, 'log_mkt_cap', 1)
    # vol_rank = transform(output, 'vol', -1)
    
    alpha_k6 = output['f12'] + output['f13'] + output['f25'] + output['f44'] + output['f59'] + output['f76'] + output['f90'] + output['f78'] + output['f92'] + output['f93']
    
    alpha_4 = output['f4']
    alpha_52 = output['f52']
    alpha_54 = output['f54']
    # alpha_78 = output['f78']
    alpha_86 = output['f86']
    # alpha_90 = output['f90']
    alpha_91 = output['f91']
    # alpha_92 = output['f92']
    # alpha_93 = output['f93']
    alpha_94 = output['f94']
    alpha_95 = output['f95']
    
    alpha_s3 = -(output['f65'] + output['f71'] + output['f72'] + output['f84'])
    
    # output['d2'] = 0
    # output.loc[output['guidance_days_prev'] >= 10, 'd2'] = 1.0
    
    output['d1'] = 0
    output['d2'] = 0
    output['d3'] = 0
    output['d4'] = 0
    
    N1 = 10
    N2 = 50
    output.loc[(output['guidance_days_prev'] < N1), 'd1'] = 1.0
    output.loc[(output['guidance_days_prev'] > N1) & (output['guidance_days_prev'] < N2), 'd2'] = 1.0
    output.loc[output['guidance_days_prev'] > N2, 'd3'] = 1.0
    output.loc[(output['d2'] == 1) | (output['d3'] == 1), 'd4'] = 1                 
    
    # alpha = alpha_k6 + alpha_54*output['d1'] + alpha_4*output['d2'] + (alpha_54 + alpha_78)*output['d3'] + (alpha_s3 + alpha_52)*output['d4'] + alpha_86 + alpha_90

    alpha =  alpha_4 + alpha_52 + alpha_s3 + alpha_k6 + alpha_54 + alpha_86 + alpha_91 + alpha_94 
    # + alpha_95
    
    
    # alpha = (alpha_4 + alpha_52 + alpha_s3 + alpha_78)*output['d2'] + alpha_k6 + alpha_54 

    # alpha = (alpha_s3 + alpha_78)*output['d2'] + alpha_k6 + alpha_54 + alpha_52

    # alpha = alpha_52 + alpha_k6 + alpha_54 + alpha_78 + alpha_s3
    # alpha = alpha_k6 + alpha_54 + alpha_78 + alpha_s3
    # alpha = alpha_k6 + alpha_78 + alpha_s3
    # alpha = (alpha52 + alpha_k6 + alpha_s3)*output['d2']
    
    output['signal_rank'] = alpha 
    context.alpha = output['signal_rank']
    
    context.risk_factor_betas = algo.pipeline_output(
      'risk_pipe'
    ).dropna()
    
    context.beta_pipeline = algo.pipeline_output('beta_pipe')
    context.volatility = vol
    context.adv = adv
    
    
def rebalance(context, data):
    
    alpha = context.alpha
    beta = context.beta_pipeline
    factor_loadings = context.risk_factor_betas
    vol = context.volatility
     
    validSecurities = list(
        set(alpha.index.values.tolist()) & 
        set(beta.index.values.tolist()) &
        set(factor_loadings.index.values.tolist()))
    
    alpha = alpha.loc[validSecurities]
    factor_loadings = factor_loadings.loc[validSecurities, :]
    
    beta = beta.loc[validSecurities, :]   
    vol = vol.loc[validSecurities]
    
    #Variance = Square of volatility
    variance = vol * vol
    
    context.min_turnover = 0.0
    int_port = np.zeros(len(alpha))
                        
    portfolio_value = context.portfolio.portfolio_value
    adv_wt = 0.05*context.adv[validSecurities].fillna(0)/portfolio_value
    allPos = context.portfolio.positions
    currentSecurities = list(allPos.keys())
    defunctSecurities = list(set(currentSecurities) - set(validSecurities))               
    for (i,sec) in enumerate(validSecurities):
        if allPos[sec]:
           int_port[i] = (allPos[sec].amount*allPos[sec].last_sale_price)/portfolio_value
        
    for (i,sec) in enumerate(defunctSecurities):
        if allPos[sec]:
           context.min_turnover += allPos[sec].amount/portfolio_value
    
    context.initial_portfolio = pd.Series(int_port, index=validSecurities)
                        
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
                beta_neutral
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