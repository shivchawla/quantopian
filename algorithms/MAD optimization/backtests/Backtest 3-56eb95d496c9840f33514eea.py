
from pandas import Series, DataFrame
import pandas as pd
import statsmodels
import statsmodels.api as sm
import numpy as np
import math
import scipy.stats.mstats as st
from scipy.optimize import minimize


def initialize(context):
    context.benchmarkSecurity = symbol('SPY')
    schedule_function(func=rebalance,
                      date_rule=date_rules.week_start(),
                      time_rule=time_rules.market_open(minutes=1),
                      half_days=True
                      )
    schedule_function(bookkeeping)
    set_slippage(slippage.FixedSlippage(spread=0.00))
    set_commission(commission.PerShare(cost=0, min_trade_cost=None))
    context.numstocks = 100
    context.long_exposure_min = 0.95
    context.short_exposure_min = -0.95
    
    context.beta_low = -0.1
    context.beta_high = 0.1
    context.maxleverage = 2.8
    #context.rank = []

def bookkeeping(context, data):   
    short_count = 0
    long_count = 0
    for sid in context.portfolio.positions:
        if context.portfolio.positions[sid].amount > 0.0:
            long_count = long_count + 1
        if context.portfolio.positions[sid].amount < 0.0:
            short_count = short_count + 1
    record(long_count=long_count)
    record(short_count=short_count)
    # gross leverage should be 2, net leverage should be 0!
    record(leverage=context.account.leverage)
    
def handle_data(context, data):
    record(lever=context.account.leverage, exposure=context.account.net_leverage)
    
def before_trading_start(context, data):         
    df = get_fundamentals(
        query(fundamentals.valuation.market_cap,
              fundamentals.valuation.shares_outstanding,
              fundamentals.share_class_reference.symbol,
              fundamentals.company_reference.standard_name,
              )
        .filter(fundamentals.valuation.market_cap != None)
        .filter(fundamentals.valuation.shares_outstanding != None)  
        .filter(fundamentals.company_reference.primary_exchange_id != "OTCPK") # no pink sheets
        .filter(fundamentals.company_reference.primary_exchange_id != "OTCBB") # no pink sheets
        .filter(fundamentals.asset_classification.morningstar_sector_code != None) # require sector
        .filter(fundamentals.share_class_reference.security_type == 'ST00000001') # common stock only
        .filter(~fundamentals.share_class_reference.symbol.contains('_WI')) # drop when-issued
        .filter(fundamentals.share_class_reference.is_primary_share == True) # remove ancillary classes
        .filter(((fundamentals.valuation.market_cap*1.0) / (fundamentals.valuation.shares_outstanding*1.0)) > 1.0)  # stock price > $1
        .filter(fundamentals.share_class_reference.is_depositary_receipt == False) # !ADR/GDR
        .filter(fundamentals.valuation.market_cap > 100000000) # cap > $100MM
        .filter(~fundamentals.company_reference.standard_name.contains(' LP')) # exclude LPs
        .filter(~fundamentals.company_reference.standard_name.contains(' L P'))
        .filter(~fundamentals.company_reference.standard_name.contains(' L.P'))
        .filter(fundamentals.balance_sheet.limited_partnership == None) # exclude LPs
        .order_by(fundamentals.valuation.market_cap.desc()) 
        .offset(0)
        .limit(500) 
        ).T
    
    context.longs = df.tail(context.numstocks)
    context.shorts = df.head(context.numstocks)
    universe = pd.concat([context.longs, context.shorts], axis=0)
   
    update_universe(universe.index.values)
    context.universe = [stock for stock in universe.index.values]
    
# Computes the weights for the portfolio with the smallest Mean Absolute Deviation  
def minimum_MAD_portfolio(context, opt_data, returns):  
                
    def _leverage(x):  
        return -sum(abs(x)) + context.maxleverage  
    
    def _long_exposure(x):
        return opt_data['longd'].dot(x) - context.long_exposure_min
    
    def _short_exposure(x):
        return context.short_exposure_min - opt_data['shortd'].dot(x)
        
    def _beta_exposure_g(x):
        return -opt_data['beta'].dot(x) + context.beta_high
    
    def _beta_exposure_l(x):
        return -context.beta_low + opt_data['beta'].dot(x)
    
    # Computes the Mean Absolute Deviation for the current iteration of weights  
    def _mad(x, returns):  
        return (returns - returns.mean()).dot(x).abs().mean()
        
    num_assets = len(opt_data)  
    guess = np.zeros(num_assets)  
    
    bnds = []
    limits_l = [0, 0.05]
    limits_s = [-0.05, 0]
       
    for stock in opt_data.index:
        if opt_data['longd'][stock] == 1:
            bnds.append(limits_l)
        else:
            bnds.append(limits_s)    
            
    bnds = tuple(tuple(x) for x in bnds)
    cons = ({'type':'ineq', 'fun': _leverage},
            {'type':'ineq', 'fun': _long_exposure},
            {'type':'ineq', 'fun': _short_exposure},
            {'type':'ineq', 'fun': _beta_exposure_l},
            {'type':'ineq', 'fun': _beta_exposure_g}
           ) 
    min_mad_results = minimize(_mad, guess, args=returns, constraints=cons, bounds=bnds,options =        {'maxiter':1000}, method = 'SLSQP')
    log.info(min_mad_results.success)
    log.info((returns - returns.mean()).dot(min_mad_results.x).abs().mean())
    log.info(opt_data['beta'].dot(min_mad_results.x))
    
    if min_mad_results.success:
        return (True, pd.Series(index=returns.columns, data=min_mad_results.x))
    else:
        return (False, guess)
        
def rebalance(context, data):    
    
    desiredSids = context.universe
    holdingSids = Series(context.portfolio.positions.keys())
    gettingTheBoot = holdingSids[holdingSids.isin(desiredSids) == False]
    for (ix,sid) in gettingTheBoot.iteritems():
        order_target_percent(sid, 0)
   
    log_price = np.log(history(200, "1d", 'price')).dropna(axis=1)
    returns = (log_price - log_price.shift(1)).dropna()
    
    beta = calculate_beta(context, data)                      
    
    longd = Series(index = np.append([context.longs.index], [context.shorts.index]), 
                   data = np.append( [np.ones(len(context.longs))], [np.zeros(len(context.shorts))]), 
                   name = 'longd')
   
    shortd = Series(index = np.append([context.longs.index], [context.shorts.index]), 
                    data = np.append([np.zeros(len(context.longs))],[np.ones(len(context.shorts))]), 
                    name = 'shortd')
     
    opt_data = pd.concat([beta, longd, shortd], axis = 1, join = 'inner')
    
    opt_data = opt_data[opt_data.index.isin(df_intersection_index_col(opt_data, returns))]
    returns = returns[df_intersection_index_col(opt_data, returns)]
    
    (success, weights) = minimum_MAD_portfolio(context, opt_data, returns)
    
    if success: 
        weights = round_portfolio(weights, context)
        for security in weights.keys(): 
                order_target_percent(security, weights[security])

def calculate_beta(context, data):
    returns = history(250,'1d','price').pct_change()
    returns_stocks = returns[context.universe]
    returns_spy = returns[context.benchmarkSecurity]
    
    beta_span = 100
    benchVar = pd.stats.moments.ewmvar(returns_spy, span=beta_span)[beta_span:]
    cov = pd.stats.moments.ewmcov(returns_stocks, returns_spy, span=beta_span)[beta_span:]
    beta = cov.div(benchVar.ix[0]).iloc[-1]
    beta.name = 'beta'
    return beta

def round_portfolio(portfol, context):       
    for security in portfol.keys(): 
        if abs(portfol[security]) < 0.0001 or sid is symbol('HVT_A'):
            portfol[security] = 0
                          
    return portfol
    
def df_intersection_index(df1, df2):
    return df2.index.isin(df1.index) & df1.index.isin(df2.index)

def df_intersection_index_col(df1, df2):
    return df2.keys().intersection(df1.index)

