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
from quantopian.pipeline.factors import SimpleBeta, Returns, PercentChange
from quantopian.pipeline.data.psychsignal import stocktwits
from scipy import stats, linalg
from quantopian.pipeline.factors import AverageDollarVolume
import cvxpy as cp
from scipy import sparse as scipysp
from quantopian.pipeline.data.factset.estimates import PeriodicConsensus
import quantopian.pipeline.data.factset.estimates as fe


def signalize(df):
   return ((df.rank() - 0.5)/df.count()).replace(np.nan,0.5)
  
def initialize(context):
    set_slippage(slippage.FixedSlippage(spread=0.0))
    set_commission(commission.PerShare(cost=0, min_trade_cost=0))

    """
    Called once at the start of the algorithm.
    """
    context.max_leverage = 1.001
    context.min_leverage = 0.999
    context.max_pos_size = 0.0025
    context.max_turnover = 0.95
    context.max_beta = 0.05
    context.max_net_exposure = 0.001
    context.max_volatility = 0.05
    context.max_sector_exposure = 0.02
    context.max_style_exposure = 0.02
    context.target_mkt_beta = 0
    context.normalizing_constant = 0.0714 
    
    context.sectors = [
        'basic_materials', 
        'consumer_cyclical', 
        'financial_services',
        'real_estate',
        'consumer_defensive',
        'health_care',
        'utilities',
        'communication_services',
        'energy',
        'industrials',
        'technology']
    
    context.styles = [
        'momentum', 
        'size', 
        'value', 
        'short_term_reversal', 
        'volatility']
    
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
    inputs = [Fundamentals.forward_earning_yield]  
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

class Factor8(CustomFactor):  
    inputs = [Fundamentals.sales_yield]  
    window_length = 2
    # Compute factor6 value  
    def compute(self, today, assets, out, syield):  
        out[:] =  syield[-2]

class Factor9(CustomFactor):
        inputs = [USEquityPricing.high, USEquityPricing.low, USEquityPricing.close, stocktwits.bull_scored_messages, stocktwits.bear_scored_messages, stocktwits.total_scanned_messages]
        window_length = 21
        window_safe = True
        def compute(self, today, assets, out, high, low, close, bull, bear, total):
            v = np.nansum((high-low)/close, axis=0)
            out[:] = v*np.nansum(total*(bear-bull), axis=0)


class Factor10(CustomFactor):
    inputs = [Fundamentals.capital_expenditure, Fundamentals.cost_of_revenue] 
    window_length = 360
    
    def compute(self, today, assets, out, capex, cr):
        values = capex/cr
        out[:] = values.mean(axis = 0)

class Factor11(CustomFactor):
    inputs = [Fundamentals.revenue_growth] 
#     inputs = [Fundamentals.operation_revenue_growth3_month_avg] 
    window_length = 360
    def compute(self, today, assets, out, rate):
        out[:] = rate.mean(axis = 0)/rate.std(axis = 0)        

class Factor12(CustomFactor):
    inputs = [Fundamentals.gross_margin] 
    window_length = 360
    def compute(self, today, assets, out, rate):
        out[:] = rate.mean(axis = 0)/rate.std(axis = 0)        

class Factor13(CustomFactor):
    inputs = [Fundamentals.quick_ratio] 
    window_length = 360
    def compute(self, today, assets, out, rate):
        out[:] = rate.mean(axis = 0)/rate.std(axis = 0)        

class Factor14(CustomFactor):
    inputs = [Fundamentals.ebitda_margin] 
    window_length = 360
    def compute(self, today, assets, out, rate):
        out[:] = 1/rate.std(axis = 0)        

class Factor15(CustomFactor):
    inputs = [Fundamentals.current_ratio] 
    window_length = 360
    def compute(self, today, assets, out, rate):
        out[:] = rate.mean(axis = 0)/rate.std(axis = 0)        


        
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
    vol = Volatility(mask = base_universe)
    adv = AverageDollarVolume(mask = base_universe, window_length = 22)
    
    fq1_eps_cons = fe.PeriodicConsensus.slice('EPS', 'qf', 1)
    fq1_eps_std = fq1_eps_cons.std_dev.latest
    fq1_eps_mean = fq1_eps_cons.mean.latest
    fq1_eps_median = fq1_eps_cons.median.latest
    fq1_eps_high = fq1_eps_cons.high.latest
    fq1_eps_low = fq1_eps_cons.low.latest
    
    # fq1_eps_change_pct = PercentChange(inputs=[fq1_eps_cons.mean], window_length=5)    

    # fq1_eps_lh_dspn = (fq1_eps_high - fq1_eps_low)/(fq1_eps_high + fq1_eps_low)
    # fq1_eps_mm_diff = fq1_eps_median - fq1_eps_mean
    

    fq1_eps_up = fq1_eps_cons.up.latest
    fq1_eps_dwn = fq1_eps_cons.down.latest
    fq1_eps_num_est = fq1_eps_cons.num_est.latest

    fq1_sales_cons = fe.PeriodicConsensus.slice('SALES', 'qf', 1)
    fq1_sales_std = fq1_sales_cons.std_dev.latest
    fq1_sales_mean = fq1_sales_cons.mean.latest
    fq1_sales_up = fq1_sales_cons.up.latest
    fq1_sales_dwn = fq1_sales_cons.down.latest
    fq1_sales_num_est = fq1_sales_cons.num_est.latest

    
    fe_rec = fe.ConsensusRecommendations
    
    guid_eps_q1 = fe.Guidance.slice('EPS', 'qf', 1)
    guid_eps_q1_low = guid_eps_q1.low.latest
    guid_eps_q1_high = guid_eps_q1.high.latest
    guid_eps_q1_mean = (guid_eps_q1_low + guid_eps_q1_high)/2

    guid_eps_surprise = guid_eps_q1_mean - fq1_eps_mean
    guid_eps_std = (guid_eps_q1_high - guid_eps_q1_low)/(guid_eps_q1_low + guid_eps_q1_high)
    
    price_tgt_cons = fe.LongTermConsensus.slice('PRICE_TGT')
    eps_gr_cons = fe.LongTermConsensus.slice('EPS_LTG')
    price_tgt_mean = price_tgt_cons.mean.latest
    eps_gr_mean = eps_gr_cons.mean.latest
    price_tgt_std = price_tgt_cons.std_dev.latest
    eps_gr_std = eps_gr_cons.std_dev.latest
    
    
    
    fq0_eps_act = fe.Actuals.slice('EPS', 'qf', 0).actual_value.latest
    fqn1_eps_act = fe.Actuals.slice('EPS', 'qf', -1).actual_value.latest
    fqn2_eps_act = fe.Actuals.slice('EPS', 'qf', -2).actual_value.latest
    fqn3_eps_act = fe.Actuals.slice('EPS', 'qf', -3).actual_value.latest
    fqn4_eps_act = fe.Actuals.slice('EPS', 'qf', -4).actual_value.latest
    fqn5_eps_act = fe.Actuals.slice('EPS', 'qf', -5).actual_value.latest
    fqn6_eps_act = fe.Actuals.slice('EPS', 'qf', -6).actual_value.latest
    fqn7_eps_act = fe.Actuals.slice('EPS', 'qf', -7).actual_value.latest
    fqn8_eps_act = fe.Actuals.slice('EPS', 'qf', -8).actual_value.latest
    fqn12_eps_act = fe.Actuals.slice('EPS', 'qf', -12).actual_value.latest
    fqn16_eps_act = fe.Actuals.slice('EPS', 'qf', -16).actual_value.latest
    fqn20_eps_act = fe.Actuals.slice('EPS', 'qf', -20).actual_value.latest


    fq0_sales_act = fe.Actuals.slice('SALES', 'qf', 0).actual_value.latest
    fqn1_sales_act = fe.Actuals.slice('SALES', 'qf', -1).actual_value.latest
    fqn2_sales_act = fe.Actuals.slice('SALES', 'qf', -2).actual_value.latest
    fqn3_sales_act = fe.Actuals.slice('SALES', 'qf', -3).actual_value.latest
    fqn4_sales_act = fe.Actuals.slice('SALES', 'qf', -4).actual_value.latest
    fqn5_sales_act = fe.Actuals.slice('SALES', 'qf', -5).actual_value.latest
    fqn6_sales_act = fe.Actuals.slice('SALES', 'qf', -6).actual_value.latest
    fqn7_sales_act = fe.Actuals.slice('SALES', 'qf', -7).actual_value.latest
    fqn8_sales_act = fe.Actuals.slice('SALES', 'qf', -8).actual_value.latest
    fqn12_sales_act = fe.Actuals.slice('SALES', 'qf', -12).actual_value.latest
    fqn16_sales_act = fe.Actuals.slice('SALES', 'qf', -16).actual_value.latest
    fqn20_sales_act = fe.Actuals.slice('SALES', 'qf', -20).actual_value.latest
    
    #Quarterly Consensus EPS/SALES
    fq0_sales_cons = fe.PeriodicConsensus.slice('SALES', 'qf', 0).mean.latest
    fqn1_sales_cons = fe.PeriodicConsensus.slice('SALES', 'qf', -1).mean.latest
    fqn2_sales_cons = fe.PeriodicConsensus.slice('SALES', 'qf', -2).mean.latest
    fqn3_sales_cons = fe.PeriodicConsensus.slice('SALES', 'qf', -3).mean.latest
    fqn4_sales_cons = fe.PeriodicConsensus.slice('SALES', 'qf', -4).mean.latest
    fqn5_sales_cons = fe.PeriodicConsensus.slice('SALES', 'qf', -5).mean.latest
    fqn6_sales_cons = fe.PeriodicConsensus.slice('SALES', 'qf', -6).mean.latest
    fqn7_sales_cons = fe.PeriodicConsensus.slice('SALES', 'qf', -7).mean.latest
    fqn8_sales_cons = fe.PeriodicConsensus.slice('SALES', 'qf', -8).mean.latest

    
    fq0_sales_surprise = (fq0_sales_act - fq0_sales_cons)/fq0_sales_cons
    fqn1_sales_surprise = (fqn1_sales_act - fqn1_sales_cons)/fqn1_sales_cons
    fqn2_sales_surprise = (fqn2_sales_act - fqn2_sales_cons)/fqn2_sales_cons
    fqn3_sales_surprise = (fqn3_sales_act - fqn3_sales_cons)/fqn3_sales_cons
    fqn4_sales_surprise = (fqn4_sales_act - fqn4_sales_cons)/fqn4_sales_cons
    fqn5_sales_surprise = (fqn5_sales_act - fqn5_sales_cons)/fqn5_sales_cons
    fqn6_sales_surprise = (fqn6_sales_act - fqn6_sales_cons)/fqn6_sales_cons
    fqn7_sales_surprise = (fqn7_sales_act - fqn7_sales_cons)/fqn7_sales_cons
    fqn8_sales_surprise = (fqn8_sales_act - fqn8_sales_cons)/fqn8_sales_cons

    
    fq0_eps_guid = fe.Guidance.slice('EPS', 'qf', 0).high.latest
    fqn1_eps_guid = fe.Guidance.slice('EPS', 'qf', -1).high.latest
    fqn2_eps_guid = fe.Guidance.slice('EPS', 'qf', -2).high.latest
    fqn3_eps_guid = fe.Guidance.slice('EPS', 'qf', -3).high.latest
    fqn4_eps_guid = fe.Guidance.slice('EPS', 'qf', -4).high.latest
    fqn5_eps_guid = fe.Guidance.slice('EPS', 'qf', -5).high.latest
    fqn6_eps_guid = fe.Guidance.slice('EPS', 'qf', -6).high.latest
    fqn7_eps_guid = fe.Guidance.slice('EPS', 'qf', -7).high.latest
    fqn8_eps_guid = fe.Guidance.slice('EPS', 'qf', -8).high.latest

    
    fq0_sales_guid = fe.Guidance.slice('SALES', 'qf', 0).high.latest
    fqn1_sales_guid = fe.Guidance.slice('SALES', 'qf', -1).high.latest
    fqn2_sales_guid = fe.Guidance.slice('SALES', 'qf', -2).high.latest
    fqn3_sales_guid = fe.Guidance.slice('SALES', 'qf', -3).high.latest
    fqn4_sales_guid = fe.Guidance.slice('SALES', 'qf', -4).high.latest
    fqn5_sales_guid = fe.Guidance.slice('SALES', 'qf', -5).high.latest
    fqn6_sales_guid = fe.Guidance.slice('SALES', 'qf', -6).high.latest
    fqn7_sales_guid = fe.Guidance.slice('SALES', 'qf', -7).high.latest
    fqn8_sales_guid = fe.Guidance.slice('SALES', 'qf', -8).high.latest



    fq0_sales_surprise_guid = (fq0_sales_act - fq0_sales_guid)/fq0_sales_guid
    fqn1_sales_surprise_guid = (fqn1_sales_act - fqn1_sales_guid)/fqn1_sales_guid
    fqn2_sales_surprise_guid = (fqn2_sales_act - fqn2_sales_guid)/fqn2_sales_guid
    fqn3_sales_surprise_guid = (fqn3_sales_act - fqn3_sales_guid)/fqn3_sales_guid
    fqn4_sales_surprise_guid = (fqn4_sales_act - fqn4_sales_guid)/fqn4_sales_guid
    fqn5_sales_surprise_guid = (fqn5_sales_act - fqn5_sales_guid)/fqn5_sales_guid
    fqn6_sales_surprise_guid = (fqn6_sales_act - fqn6_sales_guid)/fqn6_sales_guid
    fqn7_sales_surprise_guid = (fqn7_sales_act - fqn7_sales_guid)/fqn7_sales_guid
    fqn8_sales_surprise_guid = (fqn8_sales_act - fqn8_sales_guid)/fqn8_sales_guid

    
    
    # f1 = Factor1(mask = base_universe)
    # f2 = Factor2(mask = base_universe)
    # f3 = Factor3(mask = base_universe)
    # f4 = Factor4(mask = base_universe)
    # f5 = Factor5(mask = base_universe)
    # f6 = Factor6(mask = base_universe)
    # f7 = Factor7(mask = base_universe)
    # f8 = Factor8(mask = base_universe)
    # f9 = Factor9(mask = base_universe)
    # f10 = Factor10(mask = base_universe)
    # f11 = Factor11(mask = base_universe)
    # f12 = Factor12(mask = base_universe)
    # f13 = Factor13(mask = base_universe)
    # f14 = Factor14(mask = base_universe)
    # f15 = Factor15(mask = base_universe)

    # Filter stocks out with Mcap < mcap of 100th stock in S&P500
    pipe_screen = base_universe 

    pipe_columns = {
        'mkt_cap': mkt_cap,
        'log_mkt_cap': log_mkt_cap,
        'vol': vol,
        'adv':adv,
        # 'f1': f1,
        # 'f2': f2,
        # 'f3': f3,
        # 'f4': f4,
        # 'f5': f5,
        # 'f6': f6,
        # 'f7': f7,
        # 'f8': f8,
        # 'f9': f9,
        # 'f10': f10,
        # 'f11': f11,
        # 'f12': f12,
        # 'f13': f13,
        # 'f14': f14,
        # 'f15': f15,
        'fq1_eps_std': fq1_eps_std,
        'fq1_sales_std': fq1_sales_std,
        'fq1_sales_mean': fq1_sales_mean,
        'fq1_eps_up':fq1_eps_up,
        'fq1_eps_dwn':fq1_eps_dwn,
        'fq1_eps_num_est':fq1_eps_num_est,
        
        'fq1_sales_up':fq1_sales_up,
        'fq1_sales_dwn':fq1_sales_dwn,
        'fq1_sales_num_est':fq1_sales_num_est,

        'rec_buy': fe_rec.buy.latest,
        'rec_overweight': fe_rec.over.latest,
        'rec_hold': fe_rec.hold.latest,
        'rec_underweight': fe_rec.under.latest,
        'rec_sell': fe_rec.sell.latest,
        'rec_total': fe_rec.total.latest,
        'no_rec': fe_rec.no_rec.latest,
        'rec_mark': fe_rec.mark.latest,
        'guid_eps_surprise': guid_eps_surprise,
        # 'fq1_eps_mm_diff' :fq1_eps_mm_diff,
        # 'fq1_eps_lh_dspn' :fq1_eps_lh_dspn,
        # 'fq1_eps_change_pct':fq1_eps_change_pct,
        'price_tgt_mean': price_tgt_mean,
        'eps_gr_mean': eps_gr_mean,
        'price_tgt_std': price_tgt_std,
        'eps_gr_std': eps_gr_std,
        'guid_eps_std':guid_eps_std,
        
        # 'fq0_eps_act':fq0_eps_act,
        # 'fqn1_eps_act':fqn1_eps_act,
        # 'fqn2_eps_act':fqn2_eps_act,
        # 'fqn3_eps_act':fqn3_eps_act,
        # 'fqn4_eps_act':fqn4_eps_act,
        # 'fqn5_eps_act':fqn5_eps_act,
        # 'fqn6_eps_act':fqn6_eps_act,
        # 'fqn7_eps_act':fqn7_eps_act,
        # 'fqn8_eps_act':fqn8_eps_act,
        # 'fqn12_eps_act':fqn12_eps_act,
        # 'fqn16_eps_act':fqn16_eps_act,
        # 'fqn20_eps_act':fqn20_eps_act,
            
        # 'fq0_sales_act':fq0_sales_act,
        # 'fqn1_sales_act':fqn1_sales_act,
        # 'fqn2_sales_act':fqn2_sales_act,
        # 'fqn3_sales_act':fqn3_sales_act,
        # 'fqn4_sales_act':fqn4_sales_act,
        # 'fqn5_sales_act':fqn5_sales_act,
        # 'fqn6_sales_act':fqn6_sales_act,
        # 'fqn7_sales_act':fqn7_sales_act,
        # 'fqn8_sales_act':fqn8_sales_act,
        # 'fqn12_sales_act':fqn12_sales_act,
        # 'fqn16_sales_act':fqn16_sales_act,
        # 'fqn20_sales_act':fqn20_sales_act,
        
        'fq0_eps_guid': fq0_eps_guid , 
'fqn1_eps_guid': fqn1_eps_guid,
'fqn2_eps_guid': fqn2_eps_guid,
'fqn3_eps_guid': fqn3_eps_guid,
'fqn4_eps_guid': fqn4_eps_guid,
'fqn5_eps_guid': fqn5_eps_guid,
'fqn6_eps_guid': fqn6_eps_guid,
'fqn7_eps_guid': fqn7_eps_guid,
'fqn8_eps_guid': fqn8_eps_guid,

        
        
        'fq0_sales_guid': fq0_sales_guid,
'fqn1_sales_guid': fqn1_sales_guid,
'fqn2_sales_guid': fqn2_sales_guid,
'fqn3_sales_guid': fqn3_sales_guid,
'fqn4_sales_guid': fqn4_sales_guid,
'fqn5_sales_guid': fqn5_sales_guid,
'fqn6_sales_guid': fqn6_sales_guid,
'fqn7_sales_guid': fqn7_sales_guid,
'fqn8_sales_guid': fqn8_sales_guid,

        
        'fq0_sales_surprise': fq0_sales_surprise,
        'fqn1_sales_surprise': fqn1_sales_surprise,
        'fqn2_sales_surprise': fqn2_sales_surprise,
        'fqn3_sales_surprise': fqn3_sales_surprise,
        'fqn4_sales_surprise': fqn4_sales_surprise,
        'fqn5_sales_surprise': fqn5_sales_surprise,
        'fqn6_sales_surprise': fqn6_sales_surprise,
        'fqn7_sales_surprise': fqn7_sales_surprise,
        'fqn8_sales_surprise': fqn8_sales_surprise,
        
        'fq0_sales_surprise_guid':fq0_sales_surprise_guid,  
'fqn1_sales_surprise_guid':fqn1_sales_surprise_guid,
'fqn2_sales_surprise_guid':fqn2_sales_surprise_guid,
'fqn3_sales_surprise_guid':fqn3_sales_surprise_guid,
'fqn4_sales_surprise_guid':fqn4_sales_surprise_guid,
'fqn5_sales_surprise_guid':fqn5_sales_surprise_guid,
'fqn6_sales_surprise_guid':fqn6_sales_surprise_guid,
'fqn7_sales_surprise_guid':fqn7_sales_surprise_guid,
'fqn8_sales_surprise_guid':fqn8_sales_surprise_guid


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
    mk_rank = transform(output, 'log_mkt_cap', 1)
    vol_rank = transform(output, 'vol', -1)
    
    # Alpha Factors
    # alpha1 = signalize(transform(output, 'f1', 1)*mk_rank)
    # alpha2 = signalize(signalize(transform(output, 'f2', 1)*mk_rank)*vol_rank)
    # alpha3 = transform(output, 'f3', 1)
    # alpha4 = transform(output, 'f4', -1)
    # alpha5 = transform(output, 'f5', -1)
    # alpha6 = transform(output, 'f6', -1)
    # alpha7 = transform(output, 'f7', -1)
    # alpha8 = transform(output, 'f8', -1)
    # alpha9 = signalize(transform(output, 'f9', 1)*vol_rank)
    # alpha10 = transform(output, 'f10', 1)
    
    # alpha11 = transform(output, 'f11', 1)
    # alpha12 = transform(output, 'f12', 1)
    
    # alpha13 = signalize(transform(output, 'f13', 1)*mkt_cap)
    # alpha14 = signalize(transform(output, 'f14', 1)*(1/mkt_cap))
    # alpha15 = signalize(transform(output, 'f15', 1)*(1/mkt_cap))
    
    
    alpha1_c = signalize(1/output['fq1_eps_std']).dropna() 
    alpha2_c = signalize((output['fq1_eps_up'] -output['fq1_eps_dwn'])/output['fq1_eps_num_est']).dropna()
    
    alpha22_c = signalize((output['fq1_sales_up'] -output['fq1_sales_dwn'])/output['fq1_sales_num_est']).dropna()
    
    alpha3_c = signalize(output['fq1_sales_mean']/output['fq1_sales_std']).dropna() 
    # alpha4_c = signalize(((output['rec_buy'] - output['rec_sell'])/(output['rec_buy'] + output['rec_sell']))/output['vol']).dropna()
    # alpha5_c = signalize(output['rec_mark']/output['vol']).dropna()
    alpha6_c = signalize(output['guid_eps_surprise'])
    # alpha7_c = signalize(output['fq1_eps_mm_diff'])
    # alpha8_c = signalize(-output['fq1_eps_lh_dspn'])
    # alpha9_c = signalize(output['fq1_eps_change_pct'])
    # alpha10_c = signalize(-output['eps_gr_std']).dropna()
    # alpha11_c = signalize(output['price_tgt_mean']/output['price_tgt_std']).dropna()    
    # alpha13_c = signalize(-output['guid_eps_std']).dropna()
    
    
    ##############
#     cs_eps_mean_q = output[[
#             'fq0_eps_act', 'fqn1_eps_act', 'fqn2_eps_act', 'fqn3_eps_act', 'fqn4_eps_act', 'fqn5_eps_act','fqn7_eps_act', 'fqn8_eps_act']].mean(axis=1)

#     cs_eps_std_q = output[['fq0_eps_act', 'fqn1_eps_act','fqn2_eps_act',
#         'fqn3_eps_act', 'fqn4_eps_act', 'fqn5_eps_act', 'fqn7_eps_act',
#         'fqn8_eps_act']].std(axis=1)

#     #W
#     alpha22_c = signalize(cs_eps_mean_q/cs_eps_std_q)

#     alpha261_c = signalize((output['fq0_eps_act'] -         output['fqn4_eps_act'])/output['fqn4_eps_act'])
#     alpha262_c = signalize((output['fq0_eps_act'] - output['fqn8_eps_act'])/output['fqn8_eps_act'])
#     alpha263_c = signalize((output['fq0_eps_act'] - output['fqn12_eps_act'])/output['fqn12_eps_act'])

#     alpha26_c = (alpha261_c+alpha262_c+alpha263_c)/3

#     alpha271_c = signalize((output['fq0_eps_act'] -     output['fqn3_eps_act'])/output['fqn3_eps_act'])
#     alpha272_c = signalize((output['fq0_eps_act'] - output['fqn2_eps_act'])/output['fqn2_eps_act'])

#     alpha27_c = (alpha271_c + alpha272_c)/2

#     alpha_c1 = (alpha22_c + alpha26_c + alpha27_c)/3

#     ###BASED ON SALES 

#     cs_sales_mean_q = output[['fq0_sales_act', 
# 'fqn1_sales_act',
# 'fqn2_sales_act',
# 'fqn3_sales_act',
# 'fqn4_sales_act',
# 'fqn5_sales_act',
# 'fqn7_sales_act',
# 'fqn8_sales_act']].mean(axis=1)

#     cs_sales_std_q = output[['fq0_sales_act', 
# 'fqn1_sales_act',
# 'fqn2_sales_act',
# 'fqn3_sales_act',
# 'fqn4_sales_act',
# 'fqn5_sales_act',
# 'fqn7_sales_act',
# 'fqn8_sales_act']].std(axis=1)

#     #W
#     alpha28_c = signalize(cs_sales_mean_q/cs_sales_std_q)


#     alpha291_c = signalize((output['fq0_sales_act'] - output['fqn4_sales_act'])/output['fqn4_sales_act'])
#     alpha292_c = signalize((output['fq0_sales_act'] - output['fqn8_sales_act'])/output['fqn8_sales_act'])

#     alpha29_c = (alpha291_c + alpha292_c)/2

#     alpha301_c = signalize((output['fq0_sales_act'] - output['fqn3_sales_act'])/output['fqn3_sales_act'])
#     alpha302_c = signalize((output['fq0_sales_act'] - output['fqn2_sales_act'])/output['fqn2_sales_act'])

#     alpha30_c = (alpha301_c + alpha302_c)/2

#     alpha_c2 = (alpha28_c + alpha29_c + alpha30_c)/3

    ########
    
    alpha321 = signalize((output['fq0_eps_guid'] - output['fqn4_eps_guid'])/output['fqn4_eps_guid'])
    alpha322 = signalize((output['fq0_eps_guid'] - output['fqn8_eps_guid'])/output['fqn8_eps_guid'])

    alpha32 = (alpha321+alpha322)/2
    
    sales_surprise_std = output[[
    'fq0_sales_surprise',
    'fqn1_sales_surprise', 
    'fqn2_sales_surprise',
    'fqn3_sales_surprise',
    'fqn4_sales_surprise',
    'fqn5_sales_surprise',
    'fqn6_sales_surprise',
    'fqn7_sales_surprise',
    'fqn8_sales_surprise'
]].std(axis=1)

    sales_surprise_mean = output[[
    'fq0_sales_surprise',
    'fqn1_sales_surprise', 
    'fqn2_sales_surprise',
    'fqn3_sales_surprise',
    'fqn4_sales_surprise',
    'fqn5_sales_surprise',
    'fqn6_sales_surprise',
    'fqn7_sales_surprise',
    'fqn8_sales_surprise'
]].mean(axis=1)

    alpha38 = signalize(sales_surprise_mean/sales_surprise_std)

    #####
    
    sales_surprise_guid_std = output[[
    'fq0_sales_surprise_guid',
    'fqn1_sales_surprise_guid', 
    'fqn2_sales_surprise_guid',
    'fqn3_sales_surprise_guid',
    'fqn4_sales_surprise_guid',
    'fqn5_sales_surprise_guid',
    'fqn6_sales_surprise_guid',
    'fqn7_sales_surprise_guid',
    'fqn8_sales_surprise_guid'
]].std(axis=1)

    sales_surprise_guid_mean = output[[
    'fq0_sales_surprise_guid',
    'fqn1_sales_surprise_guid', 
    'fqn2_sales_surprise_guid',
    'fqn3_sales_surprise_guid',
    'fqn4_sales_surprise_guid',
    'fqn5_sales_surprise_guid',
    'fqn6_sales_surprise_guid',
    'fqn7_sales_surprise_guid',
    'fqn8_sales_surprise_guid'
]].mean(axis=1)

    alpha39 = signalize(sales_surprise_guid_mean/sales_surprise_guid_std)

    
    cs_sales_mean_q = output[['fq0_sales_guid', 
'fqn1_sales_guid',
'fqn2_sales_guid',
'fqn3_sales_guid',
'fqn4_sales_guid',
'fqn5_sales_guid',
'fqn7_sales_guid',
'fqn8_sales_guid']].mean(axis=1)

    cs_sales_std_q = output[['fq0_sales_guid', 
'fqn1_sales_guid',
'fqn2_sales_guid',
'fqn3_sales_guid',
'fqn4_sales_guid',
'fqn5_sales_guid',
'fqn7_sales_guid',
'fqn8_sales_guid']].std(axis=1)

#W
    alpha34 = signalize(cs_sales_mean_q/cs_sales_std_q)

    
    
    alpha361 = signalize((output['fq0_sales_guid'] - output['fqn3_sales_guid'])/output['fqn3_sales_guid'])
    alpha362 = signalize((output['fq0_sales_guid'] - output['fqn2_sales_guid'])/output['fqn2_sales_guid'])

    alpha36 = (alpha361+alpha362)/2

    
    ####
    alpha1_c = alpha1_c.sub(alpha1_c.mean())
    alpha2_c = alpha2_c.sub(alpha2_c.mean())
    # alpha22_c = alpha22_c.sub(alpha22_c.mean())
    alpha3_c = alpha3_c.sub(alpha3_c.mean())
    # alpha4_c = alpha4_c.sub(alpha4_c.mean())
    # alpha5_c = alpha5_c.sub(alpha5_c.mean())
    alpha6_c = alpha6_c.sub(alpha6_c.mean())
    # alpha7_c = alpha7_c.sub(alpha7_c.mean())
    # alpha8_c = alpha8_c.sub(alpha8_c.mean())
    # alpha9_c = alpha9_c.sub(alpha9_c.mean())
    # alpha10_c = alpha10_c.sub(alpha10_c.mean())
    # alpha11_c = alpha11_c.sub(alpha11_c.mean())
    # alpha12_c = alpha12_c.sub(alpha12_c.mean())
    # alpha13_c = alpha13_c.sub(alpha13_c.mean())
    alpha38 = alpha38.sub(alpha38.mean())
    alpha39 = alpha39.sub(alpha39.mean())
    alpha32 = alpha32.sub(alpha32.mean())
    alpha34 = alpha34.sub(alpha34.mean())
    alpha36 = alpha36.sub(alpha36.mean())
    
    # alpha_c = (alpha1_c + alpha2_c + alpha3_c + alpha6_c + alpha38 + alpha39 + alpha34 + alpha36)/8 
    
    alpha_cc = signalize(0.000764*alpha1_c + 
    0.000722*alpha2_c + 
    0.002289*alpha3_c + 
    -0.001483*alpha6_c + 
    0.001000*alpha38 + 
    0.001110*alpha39 + 
    -0.001545*alpha32) 
    
    
    
    
    # alpha1 = alpha1.sub(alpha1.mean())
    # alpha2 = alpha2.sub(alpha2.mean())
    # alpha3 = alpha3.sub(alpha3.mean())
    # alpha3[alpha3 < 0] = 0
    # alpha4 = alpha4.sub(alpha4.mean())
    # alpha5 = alpha5.sub(alpha5.mean())
    # alpha6 = alpha6.sub(alpha6.mean())
    # alpha7 = alpha7.sub(alpha7.mean())
    # alpha8 = alpha8.sub(alpha8.mean())
    # alpha9 = alpha9.sub(alpha9.mean())
    # alpha10 = alpha10.sub(alpha10.mean())
    
    # alpha11 = alpha11.sub(alpha11.mean())
    # alpha12 = alpha12.sub(alpha12.mean())
    
    # alpha13 = alpha13.sub(alpha13.mean())
    # alpha14 = alpha14.sub(alpha14.mean())
    # alpha15 = alpha15.sub(alpha15.mean())
      
    # s1 = 1.5*alpha4 + 0.5*alpha5 + alpha6 + alpha7 + alpha8 
    # ra = alpha1 + alpha2 + alpha3 + s1 + alpha9 + alpha10 + alpha13 + alpha14 + alpha15 + alpha_c
    
    # nAlphaFactors = 14
    # context.nAlphaFactors = nAlphaFactors
    
    output['signal_rank'] = alpha_cc #ra/nAlphaFactors
    context.alpha = output['signal_rank']
    
    output['min_weights'] = -context.max_pos_size
    output['max_weights'] = context.max_pos_size
    
    context.min_weights = output['min_weights']
    context.max_weights = output['max_weights']
    
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
        
        (cov_style, cov_sector) = compute_covariance(context, data, validSecurities)
        
        alpha = alpha.sub(alpha.mean())
        # alpha = ((alpha - alpha.mean())/alpha.std()).sort_values()
        alpha = alpha.sort_values()
        alpha[1000:-1000] = 0.0
        try_opt = alpha/alpha.abs().sum()
        try_opt = try_opt.loc[validSecurities]   
       
        
        # Running optimization implementation (testing)         
        my_opt_weights = optimize(context, alpha, factor_loadings, beta, cov_style, cov_sector, variance, adv_wt, try_opt)
        my_opt = pd.Series(my_opt_weights, index=validSecurities) 
        algo.order_optimal_portfolio(
            opt.TargetWeights(my_opt),
            constraints = [])
                
        # min_weights = context.min_weights.copy(deep = True)
        # max_weights = context.max_weights.copy(deep = True)

        # constrain_pos_size = opt.PositionConcentration(
        #     min_weights,
        #     max_weights
        # )

        # max_leverage = opt.MaxGrossExposure(context.max_leverage)

        # # Ensure long and short books
        # # are roughly the same size
        # dollar_neutral = opt.DollarNeutral(context.max_net_exposure)
        
        # # Constrain portfolio turnover
        # max_turnover = opt.MaxTurnover(context.max_turnover)
        
        # factor_risk_constraints = opt.experimental.RiskModelExposure(
        #     context.risk_factor_betas,
            
        #     min_volatility = -0.1,
        #     max_volatility = 0.1,
        #     min_size = -0.1,
        #     max_size = 0.1,
        #     min_short_term_reversal = -0.1,
        #     max_short_term_reversal = 0.1,
        #     min_value = -0.1,
        #     max_value = 0.1,
        #     min_momentum = -0.1,
        #     max_momentum = 0.1,
            
        #     min_energy = -0.05,
        #     max_energy = 0.05,
        #     min_industrials = -0.05,
        #     max_industrials = 0.05,
        #     min_health_care = -0.05,
        #     max_health_care = 0.05,
        #     min_real_estate = -0.05,
        #     max_real_estate = 0.05,
        #     min_consumer_cyclical = -0.05,
        #     max_consumer_cyclical = 0.05,
        #     min_technology = -0.05,
        #     max_technology = 0.05,
        #     min_basic_materials = -0.05,
        #     max_basic_materials = 0.05,
        #     min_consumer_defensive = -0.05,
        #     max_consumer_defensive = 0.05,
        #     min_financial_services = -0.05,
        #     max_financial_services = 0.05,
        #     min_communication_services = -0.05,
        #     max_communication_services = 0.05,
        #     version=opt.Newest
        # )
        
        # beta_neutral = opt.FactorExposure(
        #     beta[['mkt_beta']],
        #     min_exposures={'mkt_beta': -context.max_beta},
        #     max_exposures={'mkt_beta': context.max_beta},
        # )
        
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
        
        
        
def compute_covariance(context, data, securities):
    
    #1. Get factor loadings
    #2. Get 63 days historical returns on stocks
    #3. Get Factor returns by multiplying the factor loadings with st-returns
    factor_loadings = context.risk_factor_betas
    factor_loadings = factor_loadings.loc[securities, :]
    factor_loadings_sector = factor_loadings.copy()        
    
    price_history = data.history(securities, fields="price", bar_count=64, frequency="1d")
    pct_change = price_history.pct_change()
    
    for factor_name in context.styles:
        factor_loadings = factor_loadings.sort_values(by = factor_name)
        factor_loadings.loc[-50:, factor_name] = 1.0
        factor_loadings.loc[:50, factor_name] = -1.0
    
    factor_loadings[np.abs(factor_loadings) != 1.0] = 0        
    
    for factor_name in context.sectors:
        factor_loadings_sector = factor_loadings_sector.sort_values(by = factor_name)
        factor_loadings_sector.loc[-50:, factor_name] = 1.0

    factor_loadings_sector[np.abs(factor_loadings_sector) != 1.0] = 0        
    
    factor_returns_style = pct_change.dot(factor_loadings)
    cov_style = factor_returns_style.cov(min_periods=63)
    
    factor_returns_sector = pct_change.dot(factor_loadings_sector)
    cov_sector = factor_returns_sector.cov(min_periods=63)

    factor_list = context.sectors + context.styles
    cov_style = cov_style[factor_list]
    cov_style = cov_style.reindex(factor_list)
   
    cov_sector = cov_sector[factor_list]
    cov_sector = cov_sector.reindex(factor_list)
   
    return (cov_style, cov_sector) 
       
    
def optimize(context, alpha, factor_loadings, beta, cov_style, cov_sector, var, adv_wt, try_opt):
            
    nstocks = alpha.size
    ini_port = context.initial_portfolio
    
    max_wt = pd.Series(np.ones(nstocks)*context.max_pos_size, index = alpha.index.tolist())
   
    min_wt = pd.Series(-np.ones(nstocks)*context.max_pos_size, index = alpha.index.tolist())
    
    min_turnover = context.min_turnover
    max_turnover = context.max_turnover - min_turnover

    
    # Number of variables =  
    #    nStocks(weights) +  
   
    # Number of Inequality group restrictions =  
    #    nSector +  
    #    nStyles  +  
    #    1 (net exposure restrictions)  
    #    1 (gross exposure restriction) +  
    #    1 (turnover restriction)  
    #    1 (market beta exposure) 

    # Group Constraints - 1
    # min_exposure < Risk Loading transpose * W < ma_exposure
    sector_exp_matrix = None
    nsectors = len(context.sectors)
    sector_exp_bounds = np.full((nsectors), context.max_sector_exposure)
    
    for col_name in context.sectors:
        _loadings = scipysp.csc_matrix(np.matrix(
                factor_loadings[col_name].values.reshape((1, nstocks))))
        
        if sector_exp_matrix is not None:
            sector_exp_matrix = cp.vstack(sector_exp_matrix, _loadings)
            
        else:
            sector_exp_matrix = _loadings
    
    
    style_exp_matrix = None
    nstyles = len(context.styles)
    style_exp_bounds = np.full((nstyles), context.max_style_exposure)

    for col_name in context.styles:
        _loadings = scipysp.csc_matrix(np.matrix(
                factor_loadings[col_name].values.reshape((1, nstocks))))
        
        if style_exp_matrix is not None:
            style_exp_matrix = cp.vstack(style_exp_matrix, _loadings)
        else:
            style_exp_matrix = _loadings
    
    # Group constraints - 3
    # Market beta exposure
    # lBeta < (B1*W1 + B2*W2 + ... + BnWn) < uBeta
    market_beta_exp_matrix = scipysp.csc_matrix(
                np.matrix(beta[['mkt_beta']].values.reshape((nstocks,))))
    
    market_beta_exp_bound = np.full((1), context.max_beta)
    
    # Create optimization variables
    w = cp.Variable(nstocks)
    
    # Systematic risk
    F = cp.vstack(
            sector_exp_matrix, 
            style_exp_matrix)
    
    f = F*w
    
    D = np.diag(var.values)
    chlskyD = linalg.cholesky(D, lower=True)
    fD = chlskyD*w
    
    A = cp.vstack(
            sector_exp_matrix, 
            style_exp_matrix,
            market_beta_exp_matrix
        )
    
    # Group Restrictions Upper Bound
    Ub = np.hstack((
            sector_exp_bounds,
            style_exp_bounds,
            market_beta_exp_bound
        ))
    
    # Group Restrictions Lower Bound
    Lb = np.hstack((
            -1*sector_exp_bounds,
            -1*style_exp_bounds,
            -1*market_beta_exp_bound
        ))
    
     # Optimization Problem Constraints (Group + Variable)
    constraints = [
        A*w <= Ub,
        A*w >= Lb,
        w <= max_wt.values.reshape((nstocks,)),
        w >= min_wt.values.reshape((nstocks,)),
        cp.sum_entries(w) <= context.max_net_exposure,
        cp.sum_entries(w) >= -context.max_net_exposure,
        cp.norm(w,1) <= context.max_leverage,
        cp.norm(w-ini_port.values.reshape((nstocks,)),1) <= context.max_turnover
    ]
        
    #Objective Function - Maximize Alpha
    c = alpha.values
    
    total_alpha = c.T*w
    gamma_sys = cp.Parameter(sign = "positive")
    gamma_sys.value = 5.0 * context.normalizing_constant
    
    gamma_unsys = cp.Parameter(sign = "positive")
    gamma_unsys.value = 100.0 * context.normalizing_constant

    gamma_beta = cp.Parameter(sign = "positive")
    gamma_beta.value = 140.0 * context.normalizing_constant
    
    risk_sys_style = cp.quad_form(f, cov_style.as_matrix()) 
    risk_sys_sector = cp.quad_form(f, cov_sector.as_matrix()) 
    risk_unsys = cp.sum_squares(fD)
    
    mkt_beta_deviation = cp.sum_squares(market_beta_exp_matrix*w - context.target_mkt_beta)
    
    beta_deviation = mkt_beta_deviation
    
    port_deviation = cp.norm(w - try_opt.values, 1)
    gamma_port_deviation = cp.Parameter(sign = "positive")
    gamma_port_deviation.value = 10.0 * context.normalizing_constant
    
    objective = cp.Maximize(total_alpha - gamma_sys*risk_sys_style - gamma_sys*risk_sys_sector - gamma_unsys*risk_unsys - gamma_beta*beta_deviation - gamma_port_deviation*port_deviation)
    
    if context.init:
        prob = cp.Problem(objective, constraints)
        try:
            prob.solve()
            w = np.asarray(w.value).flatten().reshape((nstocks,))
            context.init = False
            
            return w
        except Exception as e:
            print "Error solving optimization"
            print e
            context.init = False
            return ini_port
        
    turnover = np.linspace(0.07, context.max_turnover, num=100)
    
    for max_turnover in turnover:

        # Optimization Problem Constraints (Group + Variable)
        constraints = [
            A*w <= Ub,
            A*w >= Lb,
            w <= max_wt.values.reshape((nstocks,)),
            w >= min_wt.values.reshape((nstocks,)),
            cp.sum_entries(w) <= context.max_net_exposure,
            cp.sum_entries(w) >= -context.max_net_exposure,
            cp.norm(w,1) <= context.max_leverage,
            cp.norm(w-ini_port.values.reshape((nstocks,)),1) <= max_turnover
        ]
    
        prob = cp.Problem(objective, constraints)
        try:
            prob.solve()
            w = np.asarray(w.value).flatten().reshape((nstocks,))
        
            return w
        except Exception as e:
            print "Updating constraints for turnover"

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