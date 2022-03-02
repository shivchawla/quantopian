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
from quantopian.pipeline.factors import SimpleBeta, Returns
from quantopian.pipeline.data.psychsignal import stocktwits
from scipy import stats
from quantopian.pipeline.factors import AverageDollarVolume
import cvxpy as cp
from scipy import sparse as scipysp

def signalize(df):
   return ((df.rank() - 0.5)/df.count()).replace(np.nan,0.5)
  
def initialize(context):
    """
    Called once at the start of the algorithm.
    """
    context.max_leverage = 1.075
    context.min_leverage = 0.925
    context.max_pos_size = 0.025
    context.max_turnover = 0.95
    context.max_beta = 0.15
    context.max_net_exposure = 0.075
    context.max_volatility = 0.05
    context.max_sector_exposure = 0.18
    context.max_style_exposure = 0.38
    context.target_mkt_beta = 0
    
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

        
class AssetTurnoverConsistency(CustomFactor):
    inputs = [Fundamentals.assets_turnover] 
    window_length = 360
    def compute(self, today, assets, out, rate):
        out[:] = rate.mean(axis = 0)/rate.std(axis = 0)        

        
class TotalYieldConsistency(CustomFactor):
    inputs = [Fundamentals.total_yield] 
    window_length = 360
    def compute(self, today, assets, out, rate):
        out[:] = rate.mean(axis = 0)/rate.std(axis = 0)        

        
class AdvancedMomentum(CustomFactor):
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
    
    
class MeanRev(CustomFactor):   
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
                
            out[:] = b-a        

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
    f16 = AdvancedMomentum(mask = base_universe)
    f17 = MeanRev(mask = base_universe)
    f18 = AssetTurnoverConsistency(mask = base_universe)    
    f19 = TotalYieldConsistency(mask = base_universe)

    # Filter stocks out with Mcap < mcap of 100th stock in S&P500
    pipe_screen = base_universe 

    pipe_columns = {
        'mkt_cap': mkt_cap,
        'log_mkt_cap': log_mkt_cap,
        # 'log_ev':log_ev,
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
        'f16': f16,
        'f17': f17,
        'f18': f18,
        'f19': f19
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
    vol = output['vol']
    # vol_mean = output['vol'].mean()
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
    alpha16 = signalize(transform(output, 'f16', 1)*(1/vol))
    alpha17 = signalize(transform(output, 'f17', 1)*(mkt_cap))
    
    alpha18 = signalize(transform(output, 'f18', 1)*(1/mkt_cap))
    alpha19 = signalize(transform(output, 'f19', 1)*(1/mkt_cap))
        
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
    alpha16 = alpha16.sub(alpha16.mean())    
    alpha17 = alpha17.sub(alpha17.mean())
    alpha18 = alpha18.sub(alpha18.mean())
    alpha19 = alpha19.sub(alpha19.mean())
   
    s1 = 1.5*alpha4 + 0.5*alpha5 + alpha6 + alpha7 + alpha8 
    ra = alpha1 + alpha2 + alpha3 + s1 + alpha9 + alpha10 + alpha13 + alpha14 + alpha15 + alpha11 
    # + alpha19
    
    nAlphaFactors = 14
    context.nAlphaFactors = nAlphaFactors
    
    output['signal_rank'] = ra/nAlphaFactors
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
    
    # print "Variance"
    # print var
    
    beta['sml_beta'] = beta['small_beta'] - beta['large_beta']
    beta['vmg_beta'] = beta['value_beta'] - beta['growth_beta']
    
    # min_weights = context.min_weights.copy(deep = True)
    # max_weights = context.max_weights.copy(deep = True)
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
        
        covariance = compute_covariance(context, data, validSecurities)
        # Running optimization implementation (testing)         
        my_opt_weights = optimize(context, alpha, factor_loadings, beta, covariance, variance)
        my_opt = pd.Series(my_opt_weights, index=validSecurities) 
        algo.order_optimal_portfolio(
            opt.TargetWeights(my_opt),
            constraints = []
        )
           
        
        
def compute_covariance(context, data, securities):
    
    #1. Get factor loadings
    #2. Get 63 days historical returns on stocks
    #3. Get Factor returns by multiplying the factor loadings with st-returns
    factor_loadings = context.risk_factor_betas
    factor_loadings = factor_loadings.loc[securities, :]
        
    price_history = data.history(securities, fields="price", bar_count=64, frequency="1d")
    pct_change = price_history.pct_change()
    
    # for sector_name in context.sectors:
    #     factor_loadings = factor_loadings.sort_values(by = sector_name)
    #     factor_loadings.loc[:100, sector_name] = 1.0
    #     factor_loadings.loc[100:, sector_name] = 0

    
    for factor_name in context.styles:
        factor_loadings = factor_loadings.sort_values(by = factor_name)
        factor_loadings.loc[-50:, factor_name] = 1.0
        factor_loadings.loc[:50, factor_name] = -1.0
        
        
    factor_loadings[np.abs(factor_loadings) != 1.0] = 0        
    factor_returns = pct_change.dot(factor_loadings)
    cov = factor_returns.cov(min_periods=63)
    
    factor_list = context.sectors + context.styles
    cov = cov[factor_list]
    cov = cov.reindex(factor_list)
    
    return cov
    
def optimize(context, alpha, factor_loadings, beta, covar, var):
            
    nstocks = alpha.size
    ini_port = context.initial_portfolio
    
    max_weights = pd.Series(np.ones(nstocks)*context.max_pos_size, index = ini_port.index.tolist())
   
    min_weights = pd.Series(-np.ones(nstocks)*context.max_pos_size, index = ini_port.index.tolist())
   
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
    
    sml_beta_exp_matrix = scipysp.csc_matrix(
                np.matrix(beta[['sml_beta']].values.reshape((nstocks,))))
    vmg_beta_exp_matrix = scipysp.csc_matrix(
                np.matrix(beta[['vmg_beta']].values.reshape((nstocks,))))
    market_beta_exp_bound = np.full((1), context.max_beta)
    
    # Create optimization variables
    w = cp.Variable(nstocks)
    
    F = cp.vstack(
            sector_exp_matrix, 
            style_exp_matrix)
    
    f = F*w
    D = np.diag(var.values)
    # I = np.ones(nstocks).reshape((nstocks,)

    # Combine all Group Restrictons
    A = cp.vstack(
            sector_exp_matrix, 
            style_exp_matrix,
            market_beta_exp_matrix
        )
    
    # Group Restrictions Upper Bound
    Ub = np.hstack((
            sector_exp_bounds,
            style_exp_bounds,
            market_beta_exp_bound,
            # sml_beta_exp_bound,
            # vmg_beta_exp_bound,
        ))
    
    # Group Restrictions Lower Bound
    Lb = np.hstack((
            -1*sector_exp_bounds,
            -1*style_exp_bounds,
            -1*market_beta_exp_bound,
            # -1*sml_beta_exp_bound,
            # -1*vmg_beta_exp_bound,
        ))
    
    # Optimization Problem Constraints (Group + Variable)
    constraints = [
        A*w <= Ub,
        A*w >= Lb,
        w <= max_weights.values.reshape((nstocks,)),
        w >= min_weights.values.reshape((nstocks,)),
        cp.sum_entries(w) <= context.max_net_exposure,
        cp.sum_entries(w) >= -context.max_net_exposure,
        cp.norm(w,1) <= context.max_leverage,
        cp.norm(w-ini_port.values.reshape((nstocks,)),1) <= max_turnover
    ]
        
    #Objective Function - Maximize Alpha
    c = alpha.values
    
    total_alpha = c.T*w
    
    gamma_sys = cp.Parameter(sign = "positive")
    gamma_sys.value = 5.0/context.nAlphaFactors
    
    gamma_unsys = cp.Parameter(sign = "positive")
    gamma_unsys.value = 80.0/context.nAlphaFactors

    risk_sys = cp.quad_form(f, covar.as_matrix()) 
    risk_unsys = cp.quad_form(w, D) 
    
    mkt_beta_deviation = cp.sum_squares(market_beta_exp_matrix*w - context.target_mkt_beta)
    
    beta_deviation = 10*mkt_beta_deviation

    objective = cp.Maximize(total_alpha - gamma_sys*risk_sys - gamma_unsys*risk_unsys - beta_deviation)
    prob = cp.Problem(objective, constraints)
    try:
        prob.solve()
        # print("\nThe optimal value is", prob.value)
        # print("A solution x is")
        w = np.asarray(w.value).flatten().reshape((nstocks,))
        w[abs(w) <= 0.0001] = 0
        
        # print "Mkt Beta"
        context.mkt_beta = market_beta_exp_matrix*w
        # print "SmL Beta"
        context.sml_beta = sml_beta_exp_matrix*w
        # print "VmG Beta"
        context.vmg_beta = vmg_beta_exp_matrix*w
        
        print "Total Alpha"
        print total_alpha.value
        
        print "Total Systematic Risk"
        print gamma_sys.value*risk_sys.value
        
        print "Total Idiosyncratic Risk"
        print gamma_unsys.value*risk_unsys.value
        
        return w
    except Exception as e:
        print "Error solving optimization"
        print e
        return ini_port
        
def record_vars(context, data):
    """
    Plot variables at the end of each day.
    """
    context.last_portfolio_value = context.portfolio.portfolio_value
    record("Num Positions", len(context.portfolio.positions))
    pass

def handle_data(context, data):
    """
    Called every minute.
    """
    pass