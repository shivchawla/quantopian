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
from quantopian.pipeline.factors import SimpleBeta, Returns
from quantopian.pipeline.data.psychsignal import stocktwits
from scipy import stats
from quantopian.pipeline.factors import AverageDollarVolume
import cvxpy as cp
from scipy import sparse as scipysp
from quantopian.pipeline import CustomFilter


def signalize(df):
    return ((df.rank() - 0.5)/df.count()).replace(np.nan, 0.5)

def initialize(context):
    """
    Called once at the start of the algorithm.
    """
    context.max_leverage = 1.075
    context.min_leverage = 0.925
    context.max_sector_pos_size = 0.18
    context.max_pos_size = 0.01
    context.max_turnover = 0.95
    context.max_beta = 0.15
    context.max_net_exposure = 0.075
    context.max_volatility = 0.05
    
    context.assets = symbols('XLB', 'XLY', 'XLF', 'IYR', 'XLP', 'XLV', 'XLU', 'IYZ', 'XLE', 'XLI', 'XLK')
    context.sector_codes = ['101.0', '102.0', '103.0', '104.0', '205.0', '206.0', '207.0', '308.0', '309.0', '310.0', '311.0']
    
    context.min_sector_weights = {}
    context.max_sector_weights = {}
    
    context.sector_names = {
        symbol('XLB') : 'basic_materials', 
        symbol('XLY') : 'consumer_cyclical', 
        symbol('XLF') : 'financial_services',
        symbol('IYR') : 'real_estate',
        symbol('XLP') : 'consumer_defensive',
        symbol('XLV') : 'health_care',
        symbol('XLU') : 'utilities',
        symbol('IYZ') : 'communication_services',
        symbol('XLE') : 'energy',
        symbol('XLI') : 'industrials',
        symbol('XLK') :  'technology'}

    algo.schedule_function(
        rebalance,
        algo.date_rules.every_day(),
        # algo.date_rules.month_start(),
        # algo.date_rules.week_start(),
        algo.time_rules.market_open(hours=1),
    )

    # Create our dynamic stock selector.
    algo.attach_pipeline(make_pipeline(), 'pipeline')
    
    # algo.attach_pipeline(make_sector_pipeline(context.assets), 'sector_pipeline')

    algo.attach_pipeline(
        risk_loading_pipeline(),
        'risk_pipe'
    )
    
    mkt_beta = SimpleBeta(
        target=sid(8554),
        regression_length=260,
    )
    
    # beta_sector_pipe = Pipeline(
    #     columns={
    #         'mkt_beta': mkt_beta
    #     },
    #     screen = StaticAssets(context.assets) & mkt_beta.notnull() 
    # )
    
    beta_pipe = Pipeline(
        columns={
            'mkt_beta': mkt_beta
        },
        screen = QTradableStocksUS() & mkt_beta.notnull() 
    )

    # algo.attach_pipeline(beta_sector_pipe, 'beta_sector_pipe')
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

        
class Sector(CustomFactor):  
    inputs = [Fundamentals.morningstar_sector_code]  
    window_length = 1 
    def compute(self, today, assets, out, code):  
        out[:] = code[-1]

        
class Industry(CustomFactor):  
    inputs = [Fundamentals.morningstar_industry_code]  
    window_length = 1 
    def compute(self, today, assets, out, code):  
        out[:] = code[-1]

        
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

#Contest Entry#3 - 07/05/2019
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

        
class Factor16(CustomFactor):
    inputs = (USEquityPricing.close, Returns(window_length=126))
    window_length = 252
    window_safe = True

    def compute(self, today, assets, out, prices, returns):
        am = np.divide(
            (
                (prices[-21] - prices[-252]) / prices[-252] -
                prices[-1] - prices[-21]
            ) / prices[-21],
            np.nanstd(returns, axis=0)
        )

        out[:] = -am


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
    sector = Sector(mask = base_universe)
    industry = Industry(mask = base_universe)
    
    f1 = Factor1(mask = base_universe)
    f2 = Factor2(mask = base_universe)
    f3 = Factor3(mask = base_universe)
    f4 = Factor4(mask = base_universe)
    f5 = Factor5(mask = base_universe)
    f6 = Factor6(mask = base_universe)
    f7 = Factor7(mask = base_universe)
    f8 = Factor8(mask = base_universe)
    f9 = Factor9(mask = base_universe)
    f10 = Factor10(mask = base_universe)
    f11 = Factor11(mask = base_universe)
    f12 = Factor12(mask = base_universe)
    f13 = Factor13(mask = base_universe)
    f14 = Factor14(mask = base_universe)
    f15 = Factor15(mask = base_universe)
    f16 = Factor16(mask = base_universe)
    
    # Filter stocks out with Mcap < mcap of 100th stock in S&P500
    pipe_screen = base_universe 

    pipe_columns = {
        'mkt_cap': mkt_cap,
        'log_mkt_cap': log_mkt_cap,
        'sector': sector,
        'industry': industry,
        'vol': vol,
        'f1': f1,
        'f2': f2,
        'f3': f3,
        'f4': f4,
        'f5': f5,
        'f6': f6,
        'f7': f7,
        'f8': f8,
        'f9': f9,
        'f10': f10,
        'f11': f11,
        'f12': f12,
        'f13': f13,
        'f14': f14,
        'f15': f15,
        'f16': f16
    }

    return Pipeline(columns=pipe_columns, screen=pipe_screen)

def transform(df, field, multiplier=1):
    return signalize(multiplier*df[field])

def weightedMean(df, weights):
    return df.mean()
    # return (df*weights).mean()/weights.mean()


def before_trading_start(context, data):
    """
    Called every day before market open.
    """
    output = algo.pipeline_output('pipeline')
   
    mkt_cap = output['mkt_cap']
    volatility = output['vol']
    
    mk_rank = transform(output, 'log_mkt_cap', 1)
    # inv_mk_rank = transform(output, 'log_mkt_cap', -1)
    vol_rank = transform(output, 'vol', -1)
      
    # output['mkt_cap'] = inv_mk_rank*output['mkt_cap']
    # output['weightedAlpha'] = output['f1']*output['mkt_cap']
    output['weightedAlpha1'] = output['f1']*output['mkt_cap']
    output['weightedAlpha3'] = output['f3']*output['mkt_cap']
    output['weightedAlpha11'] = output['f11']*output['mkt_cap']
    output['weightedAlpha9'] = -output['f9']*output['mkt_cap']
    output['weightedAlpha6'] = output['f6']*output['mkt_cap']
    output['weightedAlpha12'] = transform(output,'f12',1)*output['mkt_cap']
    output['weightedAlpha10'] = signalize(transform(output,'f10',1)*vol_rank)*output['mkt_cap']
    output['weightedAlpha5'] = -output['f5']*output['mkt_cap']
    output['weightedAlpha16'] = -output['f16']*output['mkt_cap']
    
    gk = output.groupby(['industry'])[['weightedAlpha1','weightedAlpha3', 'weightedAlpha11','weightedAlpha9','weightedAlpha12','weightedAlpha6','weightedAlpha10', 'weightedAlpha5', 'weightedAlpha16', 'mkt_cap']].sum().reset_index()
    
    gk['s_alpha1'] = signalize(gk['weightedAlpha1']/gk['mkt_cap'])
    gk['s_alpha3'] = signalize(gk['weightedAlpha3']/gk['mkt_cap'])
    gk['s_alpha11'] = signalize(gk['weightedAlpha11']/gk['mkt_cap'])
    gk['s_alpha9'] = signalize(gk['weightedAlpha9']/gk['mkt_cap'])
    gk['s_alpha12'] = signalize(gk['weightedAlpha12']/gk['mkt_cap'])
    gk['s_alpha6'] = signalize(gk['weightedAlpha6']/gk['mkt_cap'])
    gk['s_alpha10'] = signalize(gk['weightedAlpha10']/gk['mkt_cap'])
    gk['s_alpha5'] = signalize(gk['weightedAlpha5']/gk['mkt_cap'])
    gk['s_alpha16'] = signalize(gk['weightedAlpha16']/gk['mkt_cap'])
    gk['industry'] = gk['industry'].astype(str)
    
    output['industry'] = output['industry'].astype(str)
   
    gk['alpha_industry'] = gk['s_alpha6'] 
    # + gk['s_alpha3'] + gk['s_alpha11'] + gk['s_alpha12'] + gk['s_alpha10'] + gk['s_alpha16']

    output = output.reset_index()
    output = output.merge(gk, how='outer', on=['industry'])  
    # print output.head(10)

    # # output.set_index(['level_0'], inplace=True)
    # output = output.dropna().set_index('index')
    
    output = output.set_index('index')
    # print output.head(10)
    alpha_industry = output['alpha_industry'].fillna(0)
    
    print alpha_industry.head(10)
    
    # # Alpha Factors
    alpha1 = signalize(transform(output, 'f1', 1)*mk_rank)
    alpha2 = signalize(signalize(transform(output, 'f2', 1)*mk_rank)*vol_rank)
    alpha3 = transform(output, 'f3', 1)
    alpha4 = transform(output, 'f4', -1)
    alpha5 = transform(output, 'f5', -1)
    alpha6 = transform(output, 'f6', -1)
    alpha7 = transform(output, 'f7', -1)
    alpha8 = transform(output, 'f8', -1)
    alpha9 = signalize(transform(output, 'f9', 1)*vol_rank)
    alpha10 = transform(output, 'f10', 1)
    alpha13 = signalize(transform(output, 'f13', 1)*mkt_cap)
    alpha14 = signalize(transform(output, 'f14', 1)*(1/mkt_cap))
    alpha15 = signalize(transform(output, 'f15', 1)*(1/mkt_cap))
    alpha16 = signalize(transform(output, 'f16', 1)*(1/volatility))
    
    alpha1 = alpha1.sub(alpha1.mean()).fillna(0)
    alpha2 = alpha2.sub(alpha2.mean()).fillna(0)
    alpha3 = alpha3.sub(alpha3.mean()).fillna(0)
    alpha3[alpha3 < 0] = 0
    alpha4 = alpha4.sub(alpha4.mean()).fillna(0)
    alpha5 = alpha5.sub(alpha5.mean()).fillna(0)
    alpha6 = alpha6.sub(alpha6.mean()).fillna(0)
    alpha7 = alpha7.sub(alpha7.mean()).fillna(0)
    alpha8 = alpha8.sub(alpha8.mean()).fillna(0)
    alpha9 = alpha9.sub(alpha9.mean()).fillna(0)
    alpha10 = alpha10.sub(alpha10.mean()).fillna(0)
    alpha13 = alpha13.sub(alpha13.mean()).fillna(0)
    alpha14 = alpha14.sub(alpha14.mean()).fillna(0)
    alpha15 = alpha15.sub(alpha15.mean()).fillna(0)
    alpha16 = alpha16.sub(alpha16.mean()).fillna(0)
    
    alpha_stock = alpha1 + alpha2 + alpha3 + alpha4 + alpha5 + alpha6 + alpha7 + alpha8 + alpha9 + alpha10 + alpha13 + alpha14 + alpha15 + alpha16
        
    # print alpha_stock.head(10)   
    
    # Sector Alpha/Beta/Loadings
    context.alpha = alpha_stock + alpha_industry
    # + alpha_stock
   
    alpha = context.alpha
    alpha = alpha[np.isnan(alpha)]
    
    output['min_weights'] = -context.max_pos_size
    output['max_weights'] = context.max_pos_size
    
    context.min_weights = output['min_weights']
    context.max_weights = output['max_weights']
    
    context.beta_pipeline = algo.pipeline_output('beta_pipe')
    context.risk_factor_betas = algo.pipeline_output(
      'risk_pipe'
    ).dropna()
   

def rebalance(context, data):
    
    alpha = context.alpha
    beta = context.beta_pipeline[['mkt_beta']]
    factor_loadings = context.risk_factor_betas
                 
    min_weights = context.min_weights
    max_weights = context.max_weights
    
    if not alpha.empty:
        
        # Create MaximizeAlpha objective
        objective = opt.MaximizeAlpha(alpha)

        constrain_pos_size = opt.PositionConcentration(
            min_weights,
            max_weights
        )

        # Constrain target portfolio's leverage
        max_leverage = opt.MaxGrossExposure(context.max_leverage)
        
        max_turnover = opt.MaxTurnover(context.max_turnover)

        # Ensure long and short books
        # are roughly the same size
        dollar_neutral = opt.DollarNeutral(context.max_net_exposure)
        
        beta_neutral = opt.FactorExposure(
            beta,
            min_exposures={'mkt_beta': -context.max_beta},
            max_exposures={'mkt_beta': context.max_beta},
        )
        
        factor_risk_constraints = opt.experimental.RiskModelExposure(
            factor_loadings,   
            version=opt.Newest)
    
        algo.order_optimal_portfolio(objective, constraints = [
                constrain_pos_size,
                max_leverage,
                dollar_neutral,
                max_turnover,
                factor_risk_constraints,
                beta_neutral
            ])

        
def handle_data(context, data):
    """
    Called every minute.
    """
    pass