import quantopian.optimize as opt
import quantopian.algorithm as algo
from quantopian.pipeline import Pipeline
from quantopian.pipeline.factors import CustomFactor, SimpleBeta, Returns
from quantopian.pipeline.data.builtin import USEquityPricing
from quantopian.pipeline.data import Fundamentals
from sklearn import preprocessing
from quantopian.pipeline.experimental import risk_loading_pipeline
from quantopian.pipeline.filters import QTradableStocksUS
from quantopian.pipeline.data.psychsignal import stocktwits
from scipy.stats.mstats import winsorize
from zipline.utils.numpy_utils import (
    repeat_first_axis,
    repeat_last_axis,
)
from quantopian.pipeline.data import factset
from scipy.stats.mstats import gmean
from sklearn.cluster import SpectralClustering

import math
import numpy as np
import pandas as pd
from collections import Counter
from scipy import stats, linalg
import cvxpy as cp
from scipy import sparse as scipysp

WIN_LIMIT = 0
N_FACTOR_WINDOW = 5 # trailing window of alpha factors exported to before_trading_start

def initialize(context):
    context.max_leverage = 1.001
    context.min_leverage = 0.999
    context.max_pos_size = 0.015
    context.max_turnover = 0.95
    context.max_beta = 0.001
    context.max_net_exposure = 0.001
    context.max_sector_exposure = 0.18
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
    
    algo.attach_pipeline(risk_loading_pipeline(), 'risk_loading_pipeline')
    algo.attach_pipeline(beta_pipeline(), 'beta_pipeline')
    algo.attach_pipeline(make_alpha1_pipeline(), 'alpha1_pipeline')
    # algo.attach_pipeline(make_alpha2_pipeline(), 'alpha2_pipeline')
    algo.attach_pipeline(volatility_pipeline(), 'volatility_pipeline')
    
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
    
class Volatility(CustomFactor):  
        inputs = [USEquityPricing.close]  
        window_length = 20  
        def compute(self, today, assets, out, close):  
            daily_returns = np.diff(close, axis = 0) / close[0:-1]  
            out[:] = daily_returns.std(axis = 0) * math.sqrt(252)

            
            
class MarketCap(CustomFactor):
    # Pre-declare inputs and window_length
    inputs = [USEquityPricing.close, Fundamentals.shares_outstanding]
    window_length = 1
    
    # Compute market cap value
    def compute(self, today, assets, out, close, shares):
        out[:] = close[-1] * shares[-1]
        
class LogMarketCap(CustomFactor):
    # Pre-declare inputs and window_length
    inputs = [USEquityPricing.close, Fundamentals.shares_outstanding]
    window_length = 1
    # Compute market cap value
    def compute(self, today, assets, out, close, shares):
        out[:] = np.log(close[-1] * shares[-1])

class Factor_N_Days_Ago(CustomFactor):
    def compute(self, today, assets, out, input_factor):
        out[:] = input_factor[0]

def transform(df, field, multiplier=1):
    return signalize(multiplier*df[field])

def signalize(df):
   return ((df.rank() - 0.5)/df.count()).replace(np.nan,0.5)

def preprocess(a):
    
    a = a.astype(np.float64)
    a[np.isinf(a)] = np.nan
    a = np.nan_to_num(a - np.nanmean(a))
    a = winsorize(a, limits=[WIN_LIMIT,WIN_LIMIT])
 
    return preprocessing.scale(a)

def make_factors_1():
    
    class F1(CustomFactor):
        inputs = [USEquityPricing.high, USEquityPricing.low, USEquityPricing.close, stocktwits.bull_scored_messages, stocktwits.bear_scored_messages, stocktwits.total_scanned_messages]
        window_length = 21
        window_safe = True
        def compute(self, today, assets, out, high, low, close, bull, bear, total):
            v = np.nansum((high-low)/close, axis=0)
            out[:] = preprocess(v*np.nansum(total*(bear-bull), axis=0))
                
    class F2(CustomFactor):
        inputs = [Fundamentals.fcf_yield]
        window_length = 1
        window_safe = True
        def compute(self, today, assets, out, fcf_yield):
            out[:] = preprocess(np.nan_to_num(fcf_yield[-1,:]))
                
    class F3(CustomFactor):
        inputs = [USEquityPricing.open, USEquityPricing.close]
        window_length = 21
        window_safe = True
        def compute(self, today, assets, out, open, close):
            p = (close-open)/close
            out[:] = preprocess(np.nansum(-p,axis=0))
                
    class F4(CustomFactor):   
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
                
    class F5(CustomFactor):
        inputs = [USEquityPricing.high, USEquityPricing.low, USEquityPricing.close, USEquityPricing.volume]
        window_length = 5
        window_safe = True
        def compute(self, today, assets, out, high, low, close, volume):
            vol = np.nansum(volume,axis=0)*np.nansum(np.absolute((high-low)/close),axis=0)
            out[:] = preprocess(-vol)
                
    class F6(CustomFactor):
        inputs = [Fundamentals.growth_score]
        window_length = 1
        window_safe = True
        def compute(self, today, assets, out, growth_score):
            out[:] = preprocess(growth_score[-1,:])
                
    class F7(CustomFactor):
        inputs = [Fundamentals.peg_ratio]
        window_length = 1
        window_safe = True
        def compute(self, today, assets, out, peg_ratio):
            out[:] = preprocess(-1.0/peg_ratio[-1,:])
                
    class F8(CustomFactor):
        inputs = (USEquityPricing.close, USEquityPricing.volume)
 
        # we need one more day to get the direction of the price on the first
        # day of our desired window of 5 days
        window_length = 6
        window_safe = True
            
        def compute(self, today, assets, out, close_extra, volume_extra):
            # slice off the extra row used to get the direction of the close
            # on the first day
            close = close_extra[1:]
            volume = volume_extra[1:]
                
            dollar_volume = close * volume
            denominator = dollar_volume.sum(axis=0)
                
            difference = np.diff(close_extra, axis=0)
            direction = np.where(difference > 0, 1, -1)
            numerator = (direction * dollar_volume).sum(axis=0)
                
            out[:] = preprocess(-np.divide(numerator, denominator))
                
    class F9(CustomFactor):
        inputs = [USEquityPricing.close]
        window_length = 252
        window_safe = True
            
        _x = np.arange(window_length)
        _x_var = np.var(_x)
 
        def compute(self, today, assets, out, close):
            
            x_matrix = repeat_last_axis(
            (self.window_length - 1) / 2 - self._x,
            len(assets),
            )
 
            y_bar = np.nanmean(close, axis=0)
            y_bars = repeat_first_axis(y_bar, self.window_length)
            y_matrix = close - y_bars
 
            out[:] = preprocess(-np.divide(
            (x_matrix * y_matrix).sum(axis=0) / self._x_var,
            self.window_length
            ))
                
    class F10(CustomFactor):
        inputs = [factset.Fundamentals.sales_gr_qf]
        window_length = 2*252
        window_safe = True
        def compute(self, today, assets, out, sales_growth):
            sales_growth = np.nan_to_num(sales_growth)
            sales_growth = preprocessing.scale(sales_growth,axis=0)
            out[:] = preprocess(sales_growth[-1])
 
    class F11(CustomFactor):
        window_length = 2*252
        window_safe = True
        inputs = [factset.Fundamentals.ebit_oper_mgn_qf]
        def compute(self, today, assets, out, ebit_oper_mgn):
            ebit_oper_mgn = np.nan_to_num(ebit_oper_mgn)
            ebit_oper_mgn = preprocessing.scale(ebit_oper_mgn,axis=0)
            out[:] = preprocess(ebit_oper_mgn[-1])
 
    class F12(CustomFactor):
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
 
    class F13(CustomFactor): 
        # the biggest absolute overnight gap in the previous 90 sessions
        inputs = [USEquityPricing.close] ; window_length = 90
        window_safe = True
        def compute(self, today, assets, out, close):
            abs_log_rets = np.abs(np.diff(np.log(close),axis=0))
            max_gap = np.max(abs_log_rets, axis=0)
            out[:] = preprocess(max_gap)
        
    class F14(CustomFactor):
        inputs=[
            factset.Fundamentals.capex_assets_qf]
        window_length = 2*252
        window_safe = True
        def compute(self, today, assets, out, capex_assets):
                 
            out[:] = preprocess(-np.ptp(capex_assets,axis=0))
                
    factors = [F1,F2,F3,F4,F5,F6,F7,F8,F9,F10,F11,F12,F13,F14]
    
    return factors

def make_factors_2():
    class F1(CustomFactor):   
        # Pre-declare inputs and window_length
        inputs = [Fundamentals.pcf_ratio] 
        window_length = 2
        # Compute factor1 value
        def compute(self, today, assets, out, var):
            out[:] = var[-2]
        
    class F2(CustomFactor):   
        # Pre-declare inputs and window_length
        inputs = [Fundamentals.ps_ratio] 
        window_length = 2
        # Compute factor2 value
        def compute(self, today, assets, out, var):
            out[:] = var[-2]
    
    class F3(CustomFactor):
        # Pre-declare inputs and window_length
        inputs = [Fundamentals.enterprise_value, Fundamentals.free_cash_flow, USEquityPricing.close, Fundamentals.shares_outstanding, Fundamentals.total_assets] 
        window_length = 2
        # Compute factor3 value
        def compute(self, today, assets, out, ev, var, close, shares, ta):
            out[:] = var[-2]/(ev[-2]*close[-2]*shares[-2]*ta[-2])**(1./3.)

                
    class F4(CustomFactor):
        # Pre-declare inputs and window_length
        inputs = [Fundamentals.enterprise_value, Fundamentals.free_cash_flow, USEquityPricing.close, Fundamentals.shares_outstanding, Fundamentals.total_assets] 
        window_length = 2
        # Compute factor4 value
        def compute(self, today, assets, out, ev, var, close, shares, ta):
            out[:] = ta[-2]/(ev[-2]*close[-2]*shares[-2])**(1./2.)
        
    class F5(CustomFactor):
        """
        TEM = standard deviation of past 6 quarters' reports
        """
        inputs = [Fundamentals.capital_expenditure, Fundamentals.enterprise_value] 
        window_length = 390
        # Compute factor5 value
        def compute(self, today, assets, out, capex, ev):
            values = capex/ev
            out[:] = values.std(axis = 0)

    class F6(CustomFactor):  
        inputs = [Fundamentals.forward_earning_yield]  
        window_length = 2
        # Compute factor6 value  
        def compute(self, today, assets, out, syield):  
            out[:] =  syield[-2]

    class F7(CustomFactor):  
        inputs = [Fundamentals.earning_yield]  
        window_length = 2
        # Compute factor6 value  
        def compute(self, today, assets, out, syield):  
            out[:] =  syield[-2]

    class F8(CustomFactor):  
        inputs = [Fundamentals.sales_yield]  
        window_length = 2
        # Compute factor6 value  
        def compute(self, today, assets, out, syield):  
            out[:] =  syield[-2]

    class F9(CustomFactor):
        inputs = [USEquityPricing.high, USEquityPricing.low, USEquityPricing.close, stocktwits.bull_scored_messages, stocktwits.bear_scored_messages, stocktwits.total_scanned_messages]
        window_length = 21
        window_safe = True
        def compute(self, today, assets, out, high, low, close, bull, bear, total):
            v = np.nansum((high-low)/close, axis=0)
            out[:] = v*np.nansum(total*(bear-bull), axis=0)


    class F10(CustomFactor):
        inputs = [Fundamentals.capital_expenditure, Fundamentals.cost_of_revenue] 
        window_length = 360

        def compute(self, today, assets, out, capex, cr):
            values = capex/cr
            out[:] = values.mean(axis = 0)

    class F11(CustomFactor):
        inputs = [Fundamentals.revenue_growth] 
        #     inputs = [Fundamentals.operation_revenue_growth3_month_avg] 
        window_length = 360
        def compute(self, today, assets, out, rate):
            out[:] = rate.mean(axis = 0)/rate.std(axis = 0)        

    class F12(CustomFactor):
        inputs = [Fundamentals.gross_margin] 
        window_length = 360
        def compute(self, today, assets, out, rate):
            out[:] = rate.mean(axis = 0)/rate.std(axis = 0)        

    class F13(CustomFactor):
        inputs = [Fundamentals.quick_ratio] 
        window_length = 360
        def compute(self, today, assets, out, rate):
            out[:] = rate.mean(axis = 0)/rate.std(axis = 0)        

    class F14(CustomFactor):
        inputs = [Fundamentals.ebitda_margin] 
        window_length = 360
        def compute(self, today, assets, out, rate):
            out[:] = 1/rate.std(axis = 0)        

    class F15(CustomFactor):
        inputs = [Fundamentals.current_ratio] 
        window_length = 360
        def compute(self, today, assets, out, rate):
            out[:] = rate.mean(axis = 0)/rate.std(axis = 0)        

    factors = [F1,F2,F3,F4,F5,F6,F7,F8,F9,F10,F11,F12,F13,F14,F15]
    
    return factors

def make_alpha1_pipeline():
    factors = make_factors_1()
    
    pipeline_columns = {}
    for k,f in enumerate(factors):
        for days_ago in range(N_FACTOR_WINDOW):
            pipeline_columns['alpha_'+str(k)+'_'+str(days_ago)] = Factor_N_Days_Ago([f(mask=QTradableStocksUS())], window_length=days_ago+1, mask=QTradableStocksUS())
    
    pipe = Pipeline(columns = pipeline_columns,
    screen = QTradableStocksUS())
    
    return pipe

def make_alpha2_pipeline():
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
    
    factors = make_factors_2()
    
    # Filter stocks out with Mcap < mcap of 100th stock in S&P500
    pipe_screen = base_universe 

    pipe_columns = {
        'mkt_cap': mkt_cap,
        'log_mkt_cap': log_mkt_cap,
        'vol': vol
    }

    for k,f in enumerate(factors):
        pipe_columns['f'+str(k+1)] = f(mask = base_universe)

    return Pipeline(columns=pipe_columns, screen=pipe_screen)

def beta_pipeline():
    beta = SimpleBeta(target=sid(8554),regression_length=260,
                      allowed_missing_percentage=1.0
                     )

    pipe = Pipeline(columns = {'mkt_beta': beta},
    screen = QTradableStocksUS())
    return pipe
    
    
    
    
def volatility_pipeline():
    volatility = Volatility(mask = QTradableStocksUS())   
    pipe = Pipeline(columns = {'volatility': volatility},
    screen = QTradableStocksUS())

    return pipe

def generateAlpha1(context):
    alphas = algo.pipeline_output('alpha1_pipeline').dropna()
    
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
    
    combined_alpha = pd.Series(np.zeros_like(alphas_current.iloc[:,1].values),index=alphas_current.index)
    for k in range(n_factors):
        combined_alpha += alphas_current.iloc[:,k]/weights[k]
        
    
    return combined_alpha
    
    
def generateAlpha2(context):
    """
    Called every day before market open.
    """
    output = algo.pipeline_output('alpha2_pipeline')
    mkt_cap = output['mkt_cap']
    mk_rank = transform(output, 'log_mkt_cap', 1)
    vol_rank = transform(output, 'vol', -1)
    
    # Alpha Factors
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
    
    alpha11 = transform(output, 'f11', 1)
    alpha12 = transform(output, 'f12', 1)
    
    alpha13 = signalize(transform(output, 'f13', 1)*mkt_cap)
    alpha14 = signalize(transform(output, 'f14', 1)*(1/mkt_cap))
    alpha15 = signalize(transform(output, 'f15', 1)*(1/mkt_cap))
   
    alpha1 = alpha1.sub(alpha1.mean())
    alpha2 = alpha2.sub(alpha2.mean())
    alpha3 = alpha3.sub(alpha3.mean())
    alpha3[alpha3 < 0] = 0
    alpha4 = alpha4.sub(alpha4.mean())
    alpha5 = alpha5.sub(alpha5.mean())
    alpha6 = alpha6.sub(alpha6.mean())
    alpha7 = alpha7.sub(alpha7.mean())
    alpha8 = alpha8.sub(alpha8.mean())
    alpha9 = alpha9.sub(alpha9.mean())
    alpha10 = alpha10.sub(alpha10.mean())
    
    alpha11 = alpha11.sub(alpha11.mean())
    alpha12 = alpha12.sub(alpha12.mean())
    
    alpha13 = alpha13.sub(alpha13.mean())
    alpha14 = alpha14.sub(alpha14.mean())
    alpha15 = alpha15.sub(alpha15.mean())
      
    s1 = 1.5*alpha4 + 0.5*alpha5 + alpha6 + alpha7 + alpha8 
    ra = alpha1 + alpha2 + alpha3 + s1 + alpha9 + alpha10 + alpha13 + alpha14 + alpha15 + alpha11
    
    nAlphaFactors = 14
    context.nAlphaFactors = nAlphaFactors
    
    return ra/nAlphaFactors
    

def before_trading_start(context, data):
    risk_loadings = algo.pipeline_output('risk_loading_pipeline')
    risk_loadings.fillna(risk_loadings.median(), inplace=True)
    context.risk_loadings = risk_loadings
    context.beta_pipeline = algo.pipeline_output('beta_pipeline')
    context.volatility = algo.pipeline_output('volatility_pipeline')['volatility']
    
    context.alpha1 = generateAlpha1(context)
    # context.alpha2 = generateAlpha2(context)
    
    
def rebalance(context, data):
    
    alpha1 = context.alpha1.dropna()
    # alpha2 = context.alpha2.dropna()
 
    # demean and normalize
    alpha1 = alpha1 - alpha1.mean()
    denom = alpha1.abs().sum()
    alpha1 = alpha1/denom
    alpha1 = alpha1.fillna(0)

    # alpha = (alpha1 + alpha2)/2
    alpha = alpha1
    # alpha = alpha2
    alpha = alpha.fillna(0)
    
    beta = context.beta_pipeline
    factor_loadings = context.risk_loadings
    vol = context.volatility
     
    validSecurities = list(
        set(alpha.index.values.tolist()) & 
        set(beta.index.values.tolist()) &
        set(factor_loadings.index.values.tolist()))
    
    alpha = alpha.loc[validSecurities]
    # alpha2 = alpha2.loc[validSecurities]
    
    factor_loadings = factor_loadings.loc[validSecurities, :]
    beta = beta.loc[validSecurities, :]   
    vol = vol.loc[validSecurities]
    #Variance = Square of volatility
    variance = vol * vol
    
    context.min_turnover = 0.0
    int_port = np.zeros(len(alpha))
    allPos = context.portfolio.positions
    currentSecurities = list(allPos.keys())
    defunctSecurities = list(set(currentSecurities) - set(validSecurities))    
    portfolio_value = context.portfolio.portfolio_value

    for (i,sec) in enumerate(validSecurities):
        if allPos[sec]:
           int_port[i] = (allPos[sec].amount*allPos[sec].last_sale_price)/portfolio_value
     
    for (i,sec) in enumerate(defunctSecurities):
        if allPos[sec]:
           context.min_turnover += allPos[sec].amount/portfolio_value
    
    context.initial_portfolio = pd.Series(int_port, index=validSecurities)

    if not alpha.empty:
        
        # covariance = compute_covariance(context, data, validSecurities)
        (cov_style, cov_sector) = compute_covariance(context, data, validSecurities)

        # Running optimization implementation (testing)         
        # my_opt_weights = optimize(context, alpha, factor_loadings, beta, covariance, variance)
        my_opt_weights = optimize(context, alpha, factor_loadings, beta, cov_style, cov_sector, variance)
        my_opt = pd.Series(my_opt_weights, index=validSecurities) 
        algo.order_optimal_portfolio(
            opt.TargetWeights(my_opt),
            constraints = []
        )
        
        
def compute_covariance(context, data, securities):
    
    #1. Get factor loadings
    #2. Get 63 days historical returns on stocks
    #3. Get Factor returns by multiplying the factor loadings with st-returns
    factor_loadings = context.risk_loadings
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
    
    
def optimize(context, alpha, factor_loadings, beta, cov_style, cov_sector, var):
            
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
            market_beta_exp_bound - 0.05
        ))
    
    # Group Restrictions Lower Bound
    Lb = np.hstack((
            -1*sector_exp_bounds,
            -1*style_exp_bounds,
            -1*market_beta_exp_bound - 0.05
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
    c = alpha.values*100
    
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
    objective = cp.Maximize(total_alpha - gamma_sys*risk_sys_style - gamma_sys*risk_sys_sector - gamma_unsys*risk_unsys - gamma_beta*beta_deviation)
    
    # objective = cp.Maximize(total_alpha)
    
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
        
    turnover = np.linspace(0.15, context.max_turnover, num=100)
    
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

def recording_statements(context, data):
 
    record(num_positions=len(context.portfolio.positions))
    record(leverage=context.account.leverage)