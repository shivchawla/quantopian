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
    context.max_pos_size = 0.01
    context.max_turnover = 0.95
    context.max_beta = 0.15
    context.max_net_exposure = 0.075
    context.max_volatility = 0.05
    context.max_sector_exposure = 0.18
    context.max_style_exposure = 0.375
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

    # context.factor_portfolio_values = {}
    
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
    
    algo.schedule_function(
        compute_factor_portfolios,
        algo.date_rules.every_day(),
        algo.time_rules.market_close(),
    )
    
    algo.schedule_function(
        track_factor_portfolio_val,
        algo.date_rules.every_day(),
        algo.time_rules.market_close(hours=1),
    )

    # Create our dynamic stock selector.
    algo.attach_pipeline(make_pipeline(), 'pipeline')

    algo.attach_pipeline(
        risk_loading_pipeline(),
        'risk_pipe'
    )

    mkt_beta = SimpleBeta(
        target=sid(8554),
        regression_length=260,
    )

    value_beta = SimpleBeta(
        target=sid(22010),
        regression_length=260,
    )
    
    growth_beta = SimpleBeta(
        target=sid(22009),
        regression_length=260,
    )
    
    large_beta = SimpleBeta(
        target=sid(22148),
        regression_length=260,
    )
    
    small_beta = SimpleBeta(
        target=sid(21508),
        regression_length=260,
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
    recent_returns = Returns(
        window_length=22, 
        mask=base_universe
    )
    
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

    # Filter stocks out with Mcap < mcap of 100th stock in S&P500
    pipe_screen = base_universe 

    pipe_columns = {
        'mkt_cap': mkt_cap,
        'log_mkt_cap': log_mkt_cap,
        'vol': vol,
        'returns': recent_returns,
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
    # returns = output['returns']
    
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
    alpha13 = signalize(transform(output, 'f13', 1)*mkt_cap)
    alpha14 = signalize(transform(output, 'f14', 1)*(1/mkt_cap))
    alpha15 = signalize(transform(output, 'f15', 1)*(1/mkt_cap))
    
    alpha1 = alpha1.sub(alpha1.mean())
    alpha2 = alpha2.sub(alpha2.mean())
    alpha3 = alpha3.sub(alpha3.mean())
    context.alpha3 = alpha3
    
    alpha3[alpha3 < 0] = 0
    alpha4 = alpha4.sub(alpha4.mean())
    alpha5 = alpha5.sub(alpha5.mean())
    alpha6 = alpha6.sub(alpha6.mean())
    alpha7 = alpha7.sub(alpha7.mean())
    alpha8 = alpha8.sub(alpha8.mean())
    alpha9 = alpha9.sub(alpha9.mean())
    alpha10 = alpha10.sub(alpha10.mean())
    alpha13 = alpha13.sub(alpha13.mean())
    alpha14 = alpha14.sub(alpha14.mean())
    alpha15 = alpha15.sub(alpha15.mean())
    
    context.alpha1 = alpha1
    context.alpha2 = alpha2
    context.alpha4 = alpha4
    context.alpha5 = alpha5
    context.alpha6 = alpha6
    context.alpha7 = alpha7
    context.alpha8 = alpha8
    context.alpha9 = alpha9
    context.alpha10 = alpha10
    context.alpha13 = alpha13
    context.alpha14 = alpha14
    context.alpha15 = alpha15
    
    coef = get_trailing_returns(context, data)

    print "Trailing Returns"
    print coef

    ra = coef['alpha1_tr']*alpha1 + coef['alpha2_tr']*alpha2 + coef['alpha3_tr']*alpha3 + coef['alpha4_tr']*alpha4 + coef['alpha5_tr']*alpha5 + coef['alpha6_tr']*alpha6 + coef['alpha7_tr']*alpha7 + coef['alpha8_tr']*alpha8 + coef['alpha9_tr']*alpha9 + coef['alpha10_tr']*alpha10 + coef['alpha13_tr']*alpha13 + coef['alpha14_tr']*alpha14 + coef['alpha15_tr']*alpha15
    
    # rz = (returns - returns.mean())/returns.std()

    # low_returns = (rz > 0) & (rz < 10)
    # high_returns = (rz > 90) & (rz < 100)

    # ra[low_returns & (ra < ra.mean())] = ra.mean()
    # ra[high_returns & (ra > ra.mean())] = ra.mean()
    
    output['signal_rank'] = ra 
    context.alpha = output['signal_rank']
    
    output['min_weights'] = -context.max_pos_size
    output['max_weights'] = context.max_pos_size
    
    context.min_weights = output['min_weights']
    context.max_weights = output['max_weights']
    
    context.risk_factor_betas = algo.pipeline_output(
      'risk_pipe'
    ).dropna()

    context.beta_pipeline = algo.pipeline_output('beta_pipe')


def optimize(context, alpha, factor_loadings, beta):
            
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
    # sml_beta_exp_bound = np.full((1), 0.15)
    # vmg_beta_exp_bound = np.full((1), 0.15)

    # Create optimization variables
    w = cp.Variable(nstocks)
    
    # Combine all Group Restrictons
    A = cp.vstack(
            sector_exp_matrix, 
            style_exp_matrix,
            market_beta_exp_matrix,
            # sml_beta_exp_matrix,
            # vmg_beta_exp_matrix
        )
    
    # Group Restrictions Upper Bound
    Ub = np.hstack((
            sector_exp_bounds,
            style_exp_bounds,
            market_beta_exp_bound,
            # sml_beta_exp_bound,
            # vmg_beta_exp_bound
        ))
    
    # Group Restrictions Lower Bound
    Lb = np.hstack((
            -1*sector_exp_bounds,
            -1*style_exp_bounds,
            -1*market_beta_exp_bound,
            # -1*sml_beta_exp_bound,
            # -1*vmg_beta_exp_bound
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
    mkt_beta_deviation = cp.sum_squares(market_beta_exp_matrix*w - 0.0)
    # sml_beta_deviation = cp.sum_squares(sml_beta_exp_matrix*w - 0.0)
    # vmg_beta_deviation = cp.sum_squares(vmg_beta_exp_matrix*w - 0.0)
    beta_deviation = 10*mkt_beta_deviation
    objective = cp.Maximize(total_alpha - beta_deviation)
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
        
        return w
    except Exception as e:
        print "Error solving optimization"
        print e
        return ini_port

def get_trailing_returns(context, data):
    
    trailing_returns = {} 
    for i in [1,2,3,4,5,6,7,8,9,10,13,14,15]:
        # print "Tracking Alpha Portfolio " + str(i)
        factor = "alpha"+str(i)
        trailing_returns[factor+'_tr'] = 1.0
        
        if 'factor_portfolio_values' in context:
            port_vals = context.factor_portfolio_values[factor]
            total_returns = (port_vals/port_vals[0]) - 1.0
            # print total_returns
            trailing_return = total_returns[-22:].sum()    
            # print trailing_return
            trailing_returns[factor+'_tr'] = 1 + trailing_return   
            
    return trailing_returns
    
def get_portfolio_change(factor_portfolio, data):
    port_chg = 0
    if factor_portfolio is not None:
        secs = factor_portfolio.index
        
        prices = data.history(secs, "price", 2, "1d")
        pct_change = (prices.ix[-1] - prices.ix[0]) / prices.ix[0]

        # print pct_change
        # print type(pct_change)
        for sec in secs:
            port_chg += factor_portfolio[sec]*pct_change[sec]
    
    return port_chg

def track_factor_portfolio_val(context, data):
    spy = sid(8554)
    spy_val = data.current(spy,'price')
    obj = {'spy':  spy_val}
           
    for i in [1,2,3,4,5,6,7,8,9,10,13,14,15]:
        port_val = 10000000
        # print "Tracking Alpha Portfolio " + str(i)
        factor = "alpha"+str(i)
        factor_portfolio = None

        try:
            factor_portfolio = context[factor+"_portfolio"]
            last_port_val = context.factor_portfolio_values[factor].iloc[-1]
            # print "WTF Worked"
            port_val += last_port_val*get_portfolio_change(factor_portfolio, data)
        except Exception as e:
            x=2
            # print e

            # print "Factor Portfolio is Empty"
        
        # print "Portfolio Value "+factor+" " + str(port_val)
        obj[factor] = port_val
    
    
    if 'factor_portfolio_values' in context:
        context.factor_portfolio_values = context.factor_portfolio_values.append(obj, ignore_index=True)
    else:
        # print obj
        context.factor_portfolio_values = pd.DataFrame([obj])
            

def compute_factor_portfolios(context, data):
    #1. Get the factor portfolio 
    #2. Get the latest portfolio value (at the end of the day)
    #3. Store the factor portfolio val
    
    for i in [1,2,3,4,5,6,7,8,9,10,13,14,15]:
        # print "Tracking Alpha Portfolio " + str(i)
        factor = "alpha"+str(i)
        alpha = context[factor]
        # print alpha
        beta = context.beta_pipeline
        factor_loadings = context.risk_factor_betas
     
        validSecurities = list(
            set(alpha.index.values.tolist()) & 
            set(beta.index.values.tolist()) &
            set(factor_loadings.index.values.tolist()))
    
        alpha = alpha.loc[validSecurities]
        factor_loadings = factor_loadings.loc[validSecurities, :]
        beta = beta.loc[validSecurities, :]
    
        beta['sml_beta'] = beta['small_beta'] - beta['large_beta']
        beta['vmg_beta'] = beta['value_beta'] - beta['growth_beta']
    
        context.min_turnover = 0.0
        int_port = np.zeros(len(alpha))
        
        factor_portfolio = None
        if factor+"_portfolio" in context:
            factor_portfolio = context[factor+"_portfolio"]
        
        # print "dadadads"
        
        if factor_portfolio is not None:
            currentSecurities = list(factor_portfolio.index)            
            defunctSecurities = list(set(currentSecurities)  - set(validSecurities))               
            
            # print factor_portfolio
            
            for (i,sec) in enumerate(validSecurities):
                if sec in currentSecurities:
                    # print  factor_portfolio[sec]
                    # print type(factor_portfolio[sec])
                    int_port[i] = factor_portfolio[sec]
        
            for (i,sec) in enumerate(defunctSecurities):
                if sec in currentSecurities:
                   context.min_turnover += factor_portfolio[sec]
    
        context.initial_portfolio = pd.Series(int_port, index=validSecurities)
                        
        if not alpha.empty:
            # print "Running optimization"
            # Running optimization implementation (testing)         
            factor_port = optimize(context, alpha, factor_loadings, beta)
            factor_port = pd.Series(factor_port, index=validSecurities) 
            # print factor_port
            # print "Saving for " + factor+"_portfolio" 
            context[factor+"_portfolio"] = factor_port
            
               
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
           int_port[i] = allPos[sec].amount/portfolio_value
        
    for (i,sec) in enumerate(defunctSecurities):
        if allPos[sec]:
           context.min_turnover += allPos[sec].amount/portfolio_value
    
    context.initial_portfolio = pd.Series(int_port, index=validSecurities)
                        
    if not alpha.empty:
        
        # Running optimization implementation (testing)         
        my_opt_weights = optimize(context, alpha, factor_loadings, beta)
        my_opt = pd.Series(my_opt_weights, index=validSecurities) 
        
        algo.order_optimal_portfolio(
            opt.TargetWeights(my_opt),
            constraints = []
        )
           

def record_vars(context, data):
    record(VmGBeta = context.vmg_beta, 
           SmLBeta = context.sml_beta, 
           MktBeta = context.mkt_beta)

def handle_data(context, data):
    """
    Called every minute.
    """
    pass