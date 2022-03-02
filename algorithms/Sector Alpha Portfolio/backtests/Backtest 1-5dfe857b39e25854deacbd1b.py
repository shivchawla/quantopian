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
    context.max_pos_size = 0.2
    context.max_turnover = 0.95
    context.max_beta = 0.05
    context.max_net_exposure = 0.001
    context.max_volatility = 0.05
    context.max_sector_exposure = 0.02
    context.max_style_exposure = 0.38
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

class fcf(CustomFactor):
    inputs = [Fundamentals.fcf_yield]
    window_length = 1
    window_safe = True
    def compute(self, today, assets, out, fcf_yield):
        out[:] = preprocess(np.nan_to_num(fcf_yield[-1,:]))

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

class CapEx_Vol(CustomFactor):
    inputs=[
            factset.Fundamentals.capex_assets_qf]
    window_length = 2*252
    window_safe = True
    def compute(self, today, assets, out, capex_assets):
        out[:] = preprocess(-np.ptp(capex_assets,axis=0))

class GM_GROWTH(CustomFactor):  
    inputs = [Fundamentals.gross_margin]  
    window_length = 252
    window_safe = True
    def compute(self, today, assets, out, gm):  
        out[:] = preprocess(np.where(gm[-1]>gm[-252],1,0))
   
class GM_STABILITY_2YR(CustomFactor):  
    inputs = [Fundamentals.gross_margin]  
    window_length = 504
    window_safe = True
    def compute(self, today, assets, out, gm):  
        out[:] = -preprocess(np.std([gm[-1]-gm[-252],gm[-252]-gm[-504]],axis=0)) 
    
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

class Sector(CustomFactor):  
    inputs = [Fundamentals.morningstar_sector_code]  
    window_length = 1 
    def compute(self, today, assets, out, code):  
        out[:] = code[-1]

class peg_ratio_std(CustomFactor):
    inputs = [Fundamentals.peg_ratio]
    window_length = 66*8
    window_safe = True
    def compute(self, today, assets, out, var):
        std = np.nanstd([1/var[-1], 
            1/var[-66], 
            1/var[-66*2],
            1/var[-66*3],
            1/var[-66*4],
            1/var[-66*5],
            1/var[-66*6],
            1/var[-66*7]], axis=0)

        out[:] = std
        
        
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
    sector = Sector(mask = base_universe)

    # f1 = Factor1(mask = base_universe)
    # f2 = Factor2(mask = base_universe)
    f3 = Factor3(mask = base_universe)
    # f4 = Factor4(mask = base_universe)
    f5 = Factor5(mask = base_universe)
    # f6 = Factor6(mask = base_universe)
    # f7 = Factor7(mask = base_universe)
    # f8 = Factor8(mask = base_universe)
    f9 = Factor9(mask = base_universe)
    # f10 = Factor10(mask = base_universe)
    # f11 = Factor11(mask = base_universe)
    # f12 = Factor12(mask = base_universe)
    # f13 = Factor13(mask = base_universe)
    # f14 = Factor14(mask = base_universe)
    # f15 = Factor15(mask = base_universe)
    # f16 = fcf(mask = base_universe)
    # f17 = mean_rev(mask = base_universe)
    # f18 = CapEx_Vol(mask = base_universe)
    # f19 = GM_GROWTH(mask = base_universe)
    # f20 = GM_STABILITY_2YR(mask = base_universe)
    
    # f21  = peg_ratio_std(mask = base_universe)
    
    
    # Filter stocks out with Mcap < mcap of 100th stock in S&P500
    pipe_screen = base_universe 

    pipe_columns = {
        'mkt_cap': mkt_cap,
        'log_mkt_cap': log_mkt_cap,
        'vol': vol,
        'adv':adv,
        'sector': sector,
        # 'f1': f1,
        # 'f2': f2,
        'f3': f3,
        # 'f4': f4,
        'f5': f5,
        # 'f6': f6,
        # 'f7': f7,
        # 'f8': f8,
        'f9': f9,
        # 'f10': f10,
        # 'f11': f11,
        # 'f12': f12,
        # 'f13': f13,
        # 'f14': f14,
        # 'f15': f15,
        # 'f21': f21
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
    
    allnames = []
    factors = [3,5,9]
    #Compute Weighted Factor value (within sector)
    for x in factors:
        wfname = 'weightedFactor' + str(x)
        tfname = 'f'+str(x)
        output['flag'] = 0
        output.loc[output[tfname].notnull(), 'flag'] = 1
        output[wfname] = output[tfname]*output['mkt_cap']*output['flag']
        output['mkt_cap_'+tfname] = output['mkt_cap'] #* output['flag']
        allnames.append(wfname)
        allnames.append('mkt_cap_'+tfname)

    #Compute Sectoral Market Cap
    agg_sector_ = output.groupby(['sector'])[allnames].sum().reset_index()
    
    for x in factors:
        wfname = 'weightedFactor' + str(x)
        tfname = 'f'+str(x)
        agg_sector_[wfname] = agg_sector_[wfname]/agg_sector_['mkt_cap_'+tfname]
        agg_sector_.drop(['mkt_cap_'+tfname], axis=1, inplace=True)
    
    # agg_sector_.rename(columns = {'mkt_cap':'sector_mkt_cap'}, inplace=True)
    # agg_sector_.drop(['sector_mkt_cap'], axis=1, inplace=True)
    agg_sector_['sector'] = agg_sector_['sector'].astype(str)
    # print agg_sector_.head()

    sector_code_sid = pd.DataFrame({
        'sector': ['101.0','102.0','103.0','104.0','205.0','206.0','207.0','309.0','310.0','311.0'],
            'sid':symbols('XLB','XLY','XLF','IYR','XLP','XLV','XLU','XLE','XLI','XLK')})

    
    #XLC - 308.0

    # print sector_code_sid
    data = pd.merge(agg_sector_, sector_code_sid, on='sector',     how='outer').dropna()  
    data.set_index(['sid'], inplace=True)

    
    # factor_names = ['alpha1','alpha2','alpha3','alpha4','alpha5','alpha6','alpha7','alpha8','alpha9','alpha11', 'alpha13','alpha14','alpha15']
    
        
    factor_names = ['alpha3', 'alpha5', 'alpha9']

    # data['alpha1'] = signalize(-data['weightedFactor1'])
    # data['alpha2'] = signalize(-data['weightedFactor2'])
    data['alpha3'] = signalize(data['weightedFactor3'])
    # data['alpha4'] = signalize(data['weightedFactor4'])
    data['alpha5'] = signalize(-data['weightedFactor5'])
    # data['alpha6'] = signalize(data['weightedFactor6'])
    # data['alpha7'] = signalize(data['weightedFactor7'])
    # data['alpha8'] = signalize(data['weightedFactor8'])
    data['alpha9'] = signalize(-data['weightedFactor9'])
    # data['alpha11'] = signalize(data['weightedFactor11'])
    # data['alpha13'] = signalize(data['weightedFactor13'])
    # data['alpha14'] = signalize(data['weightedFactor14'])
    # data['alpha15'] = signalize(data['weightedFactor15'])
    # data['alpha16'] = signalize(data['weightedFactor16'])
    # data['alpha21'] = signalize(data['weightedFactor21'])
    

    for name in factor_names:
        data[name] = data[name].sub(data[name].mean())

   
#     net_alpha = (-data['alpha2'] + data['alpha5'] + data['alpha6'] \
# - data['alpha9'] - data['alpha13'] + data['alpha15'])/6

    net_alpha = (data['alpha3'] + data['alpha5'] + data['alpha9'])/3
    
    context.alpha = net_alpha 
    
    data['min_weights'] = -context.max_pos_size
    data['max_weights'] = context.max_pos_size
    
    context.min_weights = data['min_weights']
    context.max_weights = data['max_weights']
    
    context.risk_factor_betas = algo.pipeline_output(
      'risk_pipe'
    ).dropna()
    
    context.beta_pipeline = algo.pipeline_output('beta_pipe')
    context.volatility = vol
    context.adv = adv
    

def rebalance(context, data):
    
    alpha = context.alpha
                            
    if not alpha.empty:
        
        alpha = alpha.sub(alpha.mean())
        try_opt = alpha/alpha.abs().sum()
                
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
            opt.TargetWeights(try_opt),
            constraints = [
                constrain_pos_size,
                max_leverage,
                dollar_neutral,
                # max_turnover
            ])
       
        
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