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
from quantopian.pipeline.data import factset
from sklearn import preprocessing
from scipy.stats.mstats import winsorize
WIN_LIMIT = 0

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
    context.max_pos_size = 0.005
    context.max_turnover = 0.95
    context.max_beta = 0.05
    context.max_net_exposure = 0.001
    context.max_volatility = 0.05
    context.max_sector_exposure = 0.02
    context.max_style_exposure = 0.40
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

        
def make_pipeline():
    """
    A function to create our dynamic stock selector (pipeline). Documentation
    on pipeline can be found here:
    https://www.quantopian.com/help#pipeline-title
    """
    # Base universe set to the QTradableStocksUS
    base_universe = QTradableStocksUS()
    vol = Volatility(mask = base_universe)

    fq0_sales_act = fe.Actuals.slice('SALES', 'qf', 0).actual_value.latest
    fqn1_sales_act = fe.Actuals.slice('SALES', 'qf', -1).actual_value.latest
    fqn2_sales_act = fe.Actuals.slice('SALES', 'qf', -2).actual_value.latest
    fqn3_sales_act = fe.Actuals.slice('SALES', 'qf', -3).actual_value.latest
    fqn4_sales_act = fe.Actuals.slice('SALES', 'qf', -4).actual_value.latest
    fqn5_sales_act = fe.Actuals.slice('SALES', 'qf', -5).actual_value.latest
    fqn6_sales_act = fe.Actuals.slice('SALES', 'qf', -6).actual_value.latest
    fqn7_sales_act = fe.Actuals.slice('SALES', 'qf', -7).actual_value.latest
    fqn8_sales_act = fe.Actuals.slice('SALES', 'qf', -8).actual_value.latest


    fq1_sales_cons = fe.PeriodicConsensus.slice('SALES', 'qf', 1).mean.latest
    fq0_sales_cons = fe.PeriodicConsensus.slice('SALES', 'qf', 0).mean.latest
    fqn1_sales_cons = fe.PeriodicConsensus.slice('SALES', 'qf', -1).mean.latest
    fqn2_sales_cons = fe.PeriodicConsensus.slice('SALES', 'qf', -2).mean.latest
    fqn3_sales_cons = fe.PeriodicConsensus.slice('SALES', 'qf', -3).mean.latest
    fqn4_sales_cons = fe.PeriodicConsensus.slice('SALES', 'qf', -4).mean.latest
    fqn5_sales_cons = fe.PeriodicConsensus.slice('SALES', 'qf', -5).mean.latest
    fqn6_sales_cons = fe.PeriodicConsensus.slice('SALES', 'qf', -6).mean.latest
    fqn7_sales_cons = fe.PeriodicConsensus.slice('SALES', 'qf', -7).mean.latest
    fqn8_sales_cons = fe.PeriodicConsensus.slice('SALES', 'qf', -8).mean.latest


    #EPS Guidance
    guid_eps_q1_high = fe.Guidance.slice('EPS', 'qf', 1).high.latest
    guid_eps_q0_high = fe.Guidance.slice('EPS', 'qf', 0).high.latest
    guid_eps_qn1_high = fe.Guidance.slice('EPS', 'qf', -1).high.latest
    guid_eps_qn2_high = fe.Guidance.slice('EPS', 'qf', -2).high.latest
    guid_eps_qn3_high = fe.Guidance.slice('EPS', 'qf', -3).high.latest
    guid_eps_qn4_high = fe.Guidance.slice('EPS', 'qf', -4).high.latest
    guid_eps_qn5_high = fe.Guidance.slice('EPS', 'qf', -5).high.latest
    guid_eps_qn6_high = fe.Guidance.slice('EPS', 'qf', -6).high.latest
    guid_eps_qn7_high = fe.Guidance.slice('EPS', 'qf', -7).high.latest
    guid_eps_qn8_high = fe.Guidance.slice('EPS', 'qf', -8).high.latest

    guid_eps_q1_low = fe.Guidance.slice('EPS', 'qf', 1).low.latest
    guid_eps_q0_low = fe.Guidance.slice('EPS', 'qf', 0).low.latest
    guid_eps_qn1_low = fe.Guidance.slice('EPS', 'qf', -1).low.latest
    guid_eps_qn2_low = fe.Guidance.slice('EPS', 'qf', -2).low.latest
    guid_eps_qn3_low = fe.Guidance.slice('EPS', 'qf', -3).low.latest
    guid_eps_qn4_low = fe.Guidance.slice('EPS', 'qf', -4).low.latest
    guid_eps_qn5_low = fe.Guidance.slice('EPS', 'qf', -5).low.latest
    guid_eps_qn6_low = fe.Guidance.slice('EPS', 'qf', -6).low.latest
    guid_eps_qn7_low = fe.Guidance.slice('EPS', 'qf', -7).low.latest
    guid_eps_qn8_low = fe.Guidance.slice('EPS', 'qf', -8).low.latest

    #Sales Guidance
    guid_sales_q1_high = fe.Guidance.slice('SALES', 'qf', 1).high.latest
    guid_sales_q0_high = fe.Guidance.slice('SALES', 'qf', 0).high.latest
    guid_sales_qn1_high = fe.Guidance.slice('SALES', 'qf', -1).high.latest
    guid_sales_qn2_high = fe.Guidance.slice('SALES', 'qf', -2).high.latest
    guid_sales_qn3_high = fe.Guidance.slice('SALES', 'qf', -3).high.latest
    guid_sales_qn4_high = fe.Guidance.slice('SALES', 'qf', -4).high.latest
    guid_sales_qn5_high = fe.Guidance.slice('SALES', 'qf', -5).high.latest
    guid_sales_qn6_high = fe.Guidance.slice('SALES', 'qf', -6).high.latest
    guid_sales_qn7_high = fe.Guidance.slice('SALES', 'qf', -7).high.latest
    guid_sales_qn8_high = fe.Guidance.slice('SALES', 'qf', -8).high.latest

    guid_sales_q1_low = fe.Guidance.slice('SALES', 'qf', 1).low.latest
    guid_sales_q0_low = fe.Guidance.slice('SALES', 'qf', 0).low.latest
    guid_sales_qn1_low = fe.Guidance.slice('SALES', 'qf', -1).low.latest
    guid_sales_qn2_low = fe.Guidance.slice('SALES', 'qf', -2).low.latest
    guid_sales_qn3_low = fe.Guidance.slice('SALES', 'qf', -3).low.latest
    guid_sales_qn4_low = fe.Guidance.slice('SALES', 'qf', -4).low.latest
    guid_sales_qn5_low = fe.Guidance.slice('SALES', 'qf', -5).low.latest
    guid_sales_qn6_low = fe.Guidance.slice('SALES', 'qf', -6).low.latest
    guid_sales_qn7_low = fe.Guidance.slice('SALES', 'qf', -7).low.latest
    guid_sales_qn8_low = fe.Guidance.slice('SALES', 'qf', -8).low.latest


    # Filter stocks out with Mcap < mcap of 100th stock in S&P500
    pipe_screen = base_universe 

    pipe_columns = {
                'vol':vol,
             'fq0_sales_act':fq0_sales_act,
'fqn1_sales_act':fqn1_sales_act,
'fqn2_sales_act':fqn2_sales_act,
'fqn3_sales_act':fqn3_sales_act,
'fqn4_sales_act':fqn4_sales_act,
'fqn5_sales_act':fqn5_sales_act,
'fqn6_sales_act':fqn6_sales_act,
'fqn7_sales_act':fqn7_sales_act,
'fqn8_sales_act':fqn8_sales_act,


#EPS Guidance        
'guid_eps_q1_high':guid_eps_q1_high,
'guid_eps_q0_high':guid_eps_q0_high,
'guid_eps_qn1_high':guid_eps_qn1_high,
'guid_eps_qn2_high':guid_eps_qn2_high,
'guid_eps_qn3_high':guid_eps_qn3_high,
'guid_eps_qn4_high':guid_eps_qn4_high,
'guid_eps_qn5_high':guid_eps_qn5_high,
'guid_eps_qn6_high':guid_eps_qn6_high,
'guid_eps_qn7_high':guid_eps_qn7_high,
'guid_eps_qn8_high':guid_eps_qn8_high,

'guid_eps_q1_low':guid_eps_q1_low,
'guid_eps_q0_low':guid_eps_q0_low,
'guid_eps_qn1_low':guid_eps_qn1_low,
'guid_eps_qn2_low':guid_eps_qn2_low,
'guid_eps_qn3_low':guid_eps_qn3_low,
'guid_eps_qn4_low':guid_eps_qn4_low,
'guid_eps_qn5_low':guid_eps_qn5_low,
'guid_eps_qn6_low':guid_eps_qn6_low,
'guid_eps_qn7_low':guid_eps_qn7_low,
'guid_eps_qn8_low':guid_eps_qn8_low,
        
#Sales Guidance        
'guid_sales_q1_high':guid_sales_q1_high,
'guid_sales_q0_high':guid_sales_q0_high,
'guid_sales_qn1_high':guid_sales_qn1_high,
'guid_sales_qn2_high':guid_sales_qn2_high,
'guid_sales_qn3_high':guid_sales_qn3_high,
'guid_sales_qn4_high':guid_sales_qn4_high,
'guid_sales_qn5_high':guid_sales_qn5_high,
'guid_sales_qn6_high':guid_sales_qn6_high,
'guid_sales_qn7_high':guid_sales_qn7_high,
'guid_sales_qn8_high':guid_sales_qn8_high,

'guid_sales_q1_low':guid_sales_q1_low,
'guid_sales_q0_low':guid_sales_q0_low,
'guid_sales_qn1_low':guid_sales_qn1_low,
'guid_sales_qn2_low':guid_sales_qn2_low,
'guid_sales_qn3_low':guid_sales_qn3_low,
'guid_sales_qn4_low':guid_sales_qn4_low,
'guid_sales_qn5_low':guid_sales_qn5_low,
'guid_sales_qn6_low':guid_sales_qn6_low,
'guid_sales_qn7_low':guid_sales_qn7_low,
'guid_sales_qn8_low':guid_sales_qn8_low,
      
'fq1_sales_cons':fq1_sales_cons,
'fq0_sales_cons':fq0_sales_cons,
'fqn1_sales_cons':fqn1_sales_cons,
'fqn2_sales_cons':fqn2_sales_cons,
'fqn3_sales_cons':fqn3_sales_cons,
'fqn4_sales_cons':fqn4_sales_cons,
'fqn5_sales_cons':fqn5_sales_cons,
'fqn6_sales_cons':fqn6_sales_cons,
'fqn7_sales_cons':fqn7_sales_cons,
'fqn8_sales_cons':fqn8_sales_cons,


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
    
    context.volatility = output['vol']
    
    output['fq1_sales_guid'] = ((output['guid_sales_q1_low'] + output['guid_sales_q1_high'])/2).fillna(output['guid_sales_q1_high']).fillna(output['guid_sales_q1_low'])  
    output['fq0_sales_guid'] = ((output['guid_sales_q0_low'] + output['guid_sales_q0_high'])/2).fillna(output['guid_sales_q0_high']).fillna(output['guid_sales_q0_low'])  
    output['fqn1_sales_guid'] = ((output['guid_sales_qn1_low'] + output['guid_sales_qn1_high'])/2).fillna(output['guid_sales_qn1_high']).fillna(output['guid_sales_qn1_low'])  
    output['fqn2_sales_guid'] = ((output['guid_sales_qn2_low'] + output['guid_sales_qn2_high'])/2).fillna(output['guid_sales_qn2_high']).fillna(output['guid_sales_qn2_low'])  
    output['fqn3_sales_guid'] = ((output['guid_sales_qn3_low'] + output['guid_sales_qn3_high'])/2).fillna(output['guid_sales_qn3_high']).fillna(output['guid_sales_qn3_low'])  
    output['fqn4_sales_guid'] = ((output['guid_sales_qn4_low'] + output['guid_sales_qn4_high'])/2).fillna(output['guid_sales_qn4_high']).fillna(output['guid_sales_qn4_low'])  
    output['fqn5_sales_guid'] = ((output['guid_sales_qn5_low'] + output['guid_sales_qn5_high'])/2).fillna(output['guid_sales_qn5_high']).fillna(output['guid_sales_qn5_low'])  
    output['fqn6_sales_guid'] = ((output['guid_sales_qn6_low'] + output['guid_sales_qn6_high'])/2).fillna(output['guid_sales_qn6_high']).fillna(output['guid_sales_qn6_low'])  
    output['fqn7_sales_guid'] = ((output['guid_sales_qn7_low'] + output['guid_sales_qn7_high'])/2).fillna(output['guid_sales_qn7_high']).fillna(output['guid_sales_qn7_low'])  
    output['fqn8_sales_guid'] = ((output['guid_sales_qn8_low'] + output['guid_sales_qn8_high'])/2).fillna(output['guid_sales_qn8_high']).fillna(output['guid_sales_qn8_low'])  
    
    #EPS Guidance - high/low average (backstopped)
    output['fq1_eps_guid'] = ((output['guid_eps_q1_low'] + output['guid_eps_q1_high'])/2).fillna(output['guid_eps_q1_high']).fillna(output['guid_eps_q1_low'])  
    output['fq0_eps_guid'] = ((output['guid_eps_q0_low'] + output['guid_eps_q0_high'])/2).fillna(output['guid_eps_q0_high']).fillna(output['guid_eps_q0_low'])  
    output['fqn1_eps_guid'] = ((output['guid_eps_qn1_low'] + output['guid_eps_qn1_high'])/2).fillna(output['guid_eps_qn1_high']).fillna(output['guid_eps_qn1_low'])  
    output['fqn2_eps_guid'] = ((output['guid_eps_qn2_low'] + output['guid_eps_qn2_high'])/2).fillna(output['guid_eps_qn2_high']).fillna(output['guid_eps_qn2_low'])  
    output['fqn3_eps_guid'] = ((output['guid_eps_qn3_low'] + output['guid_eps_qn3_high'])/2).fillna(output['guid_eps_qn3_high']).fillna(output['guid_eps_qn3_low'])  
    output['fqn4_eps_guid'] = ((output['guid_eps_qn4_low'] + output['guid_eps_qn4_high'])/2).fillna(output['guid_eps_qn4_high']).fillna(output['guid_eps_qn4_low'])  
    output['fqn5_eps_guid'] = ((output['guid_eps_qn5_low'] + output['guid_eps_qn5_high'])/2).fillna(output['guid_eps_qn5_high']).fillna(output['guid_eps_qn5_low'])  
    output['fqn6_eps_guid'] = ((output['guid_eps_qn6_low'] + output['guid_eps_qn6_high'])/2).fillna(output['guid_eps_qn6_high']).fillna(output['guid_eps_qn6_low'])  
    output['fqn7_eps_guid'] = ((output['guid_eps_qn7_low'] + output['guid_eps_qn7_high'])/2).fillna(output['guid_eps_qn7_high']).fillna(output['guid_eps_qn7_low'])  
    output['fqn8_eps_guid'] = ((output['guid_eps_qn8_low'] + output['guid_eps_qn8_high'])/2).fillna(output['guid_eps_qn8_high']).fillna(output['guid_eps_qn8_low']) 
    
    #Consensus to Guidance Sales Surprise - Works
    output['cons_to_guid_sales_surprise_q1'] = output['fq1_sales_guid'] - output['fq1_sales_cons'] 
    output['cons_to_guid_sales_surprise_q0'] = output['fq0_sales_guid'] - output['fq0_sales_cons'] 
    output['cons_to_guid_sales_surprise_qn1'] = output['fqn1_sales_guid'] - output['fqn1_sales_cons']
    output['cons_to_guid_sales_surprise_qn2'] = output['fqn2_sales_guid'] - output['fqn2_sales_cons']
    output['cons_to_guid_sales_surprise_qn3'] = output['fqn3_sales_guid'] - output['fqn3_sales_cons']
    output['cons_to_guid_sales_surprise_qn4'] = output['fqn4_sales_guid'] - output['fqn4_sales_cons']
    output['cons_to_guid_sales_surprise_qn5'] = output['fqn5_sales_guid'] - output['fqn5_sales_cons']
    output['cons_to_guid_sales_surprise_qn6'] = output['fqn6_sales_guid'] - output['fqn6_sales_cons']
    output['cons_to_guid_sales_surprise_qn7'] = output['fqn7_sales_guid'] - output['fqn7_sales_cons']
    output['cons_to_guid_sales_surprise_qn8'] = output['fqn8_sales_guid'] - output['fqn8_sales_cons']


    #GROWTH
    #Consensus to Guidance Sales Growth - Works
    output['cons_to_guid_sales_growth_q0'] = output['fq1_sales_guid']/output['fq0_sales_cons'] - 1
    output['cons_to_guid_sales_growth_qn1'] = output['fq0_sales_guid']/output['fqn1_sales_cons'] - 1
    output['cons_to_guid_sales_growth_qn2'] = output['fqn1_sales_guid']/output['fqn2_sales_cons'] - 1
    output['cons_to_guid_sales_growth_qn3'] = output['fqn2_sales_guid']/output['fqn3_sales_cons'] - 1
    output['cons_to_guid_sales_growth_qn4'] = output['fqn3_sales_guid']/output['fqn4_sales_cons'] - 1
    output['cons_to_guid_sales_growth_qn5'] = output['fqn4_sales_guid']/output['fqn5_sales_cons'] - 1
    output['cons_to_guid_sales_growth_qn6'] = output['fqn5_sales_guid']/output['fqn6_sales_cons'] - 1
    output['cons_to_guid_sales_growth_qn7'] = output['fqn6_sales_guid']/output['fqn7_sales_cons'] - 1
    output['cons_to_guid_sales_growth_qn8'] = output['fqn7_sales_guid']/output['fqn8_sales_cons'] - 1

    alpha1 = signalize(-output[[
            'fq1_eps_guid',
            'fq0_eps_guid',
            'fqn1_eps_guid',
            'fqn2_eps_guid',
            'fqn3_eps_guid',
            'fqn4_eps_guid',
            'fqn5_eps_guid',
            'fqn6_eps_guid',
            'fqn7_eps_guid',
            'fqn8_eps_guid']].std(axis=1))

    alpha2 = signalize((output['fq1_sales_guid'] - output['fq0_sales_guid'])/output['fq0_sales_guid'])    
    
    cs_cons_to_guid_sales_surprise_std = output[['cons_to_guid_sales_surprise_q1',
                                    'cons_to_guid_sales_surprise_q0',
                                    'cons_to_guid_sales_surprise_qn1',
                                    'cons_to_guid_sales_surprise_qn2',
                                    'cons_to_guid_sales_surprise_qn3',
                                    'cons_to_guid_sales_surprise_qn4',
                                    'cons_to_guid_sales_surprise_qn5',
                                    'cons_to_guid_sales_surprise_qn6',
                                    'cons_to_guid_sales_surprise_qn7',
                                         ]].std(axis=1)


    alpha3 = signalize(-cs_cons_to_guid_sales_surprise_std) 

    output['signal_rank'] = (alpha1+alpha2+alpha3)/3
    context.alpha = output['signal_rank']
    
    output['min_weights'] = -context.max_pos_size
    output['max_weights'] = context.max_pos_size
    
    context.min_weights = output['min_weights']
    context.max_weights = output['max_weights']
    
    context.risk_factor_betas = algo.pipeline_output(
      'risk_pipe'
    ).dropna()
    
    context.beta_pipeline = algo.pipeline_output('beta_pipe')
    
    
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
        # k = alpha.abs().sort_values()
        # print k
        # alpha = ((alpha - alpha.mean())/alpha.std()).sort_values()
        # alpha = alpha.sort_values()
        # alpha[np.abs(alpha) < 0.001] = 0.0
        try_opt = alpha/alpha.abs().sum()
        # print try_opt.sort_values()
        try_opt = try_opt.loc[validSecurities]   
       
        
        # Running optimization implementation (testing)         
        my_opt_weights = optimize(context, alpha, factor_loadings, beta, cov_style, cov_sector, variance, try_opt)
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
       
    
def optimize(context, alpha, factor_loadings, beta, cov_style, cov_sector, var, try_opt):
            
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