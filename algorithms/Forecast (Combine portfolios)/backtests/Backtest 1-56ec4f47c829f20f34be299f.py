
from pandas import Series, DataFrame
import pandas as pd
import statsmodels
import statsmodels.api as sm
import datetime as dt
import datetime as datetime
import numpy as np
import math
from brokers.ib import VWAPBestEffort
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
    #set_slippage(slippage.FixedSlippage(spread=0.00))
    #set_commission(commission.PerShare(cost=0, min_trade_cost=None))
    context.numstocks = 100
    context.long_exposure_min = 0.00
    context.short_exposure_min = -0.95
    
    context.beta_low = -1.1
    context.beta_high = 0.1
    context.maxleverage = 1.0
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
    
def add_ebit_ev(df):
    ev = df['enterprise_value']
    ev[ev < 0.0] = 1.0
    df['enterprise_value'] = ev
    df['ebit_ev'] = df['ebit'] / df['enterprise_value']
    return df

def signalize(df):
    return (df.rank() - 0.5)/df.count()

def w_signalize(df, label):
    df.sort([label],inplace=True)
    return  (df['market_cap'].cumsum() - 0.5*df['market_cap'])/df['market_cap'].sum()

def add_ranking_score(df):

    df['rank'] = signalize(
        signalize(df['cash_return']) 
      + signalize(signalize(df['ebit_ev'])*df['market_cap'])
      + signalize(df['buy_back_yield'])
      #+ signalize(df['roe'])
      #+ signalize(signalize(df['roe'])* df['market_cap'])  
    )
    
    return df
    
def scale(df, minimum, maximum) :
    #df.sort(['rank'],ascending=True)
    #log.info(df['rank'].values)
    df['forecast'] = df['rank'].apply(lambda x: x*(maximum - minimum) + minimum )
    #log.info(df['forecast'].values)
    return df
    
#def fractional_ranking(array):
    
    
def before_trading_start(context, data):         
    df = get_fundamentals(
        query(fundamentals.valuation.market_cap,
              fundamentals.valuation.shares_outstanding,
              #fundamentals.income_statement.total_revenue,
              #fundamentals.earnings_report.diluted_eps,
              #fundamentals.earnings_ratios.dps_growth,
              #fundamentals.earnings_ratios.equity_per_share_growth,
              #fundamentals.operation_ratios.roe,
              #fundamentals.operation_ratios.ebit_margin, 
              #fundamentals.valuation_ratios.peg_ratio,
              #fundamentals.valuation_ratios.book_value_yield,
              fundamentals.valuation_ratios.buy_back_yield,
              fundamentals.valuation_ratios.cash_return,
              #fundamentals.valuation_ratios.cf_yield,
              #fundamentals.valuation_ratios.fcf_per_share,
              #fundamentals.valuation_ratios.sales_per_share,
              #fundamentals.earnings_ratios.diluted_eps_growth,
              fundamentals.income_statement.ebit,
              #fundamentals.income_statement.ebit_as_of,
              fundamentals.valuation.enterprise_value,
              #fundamentals.valuation.enterprise_value_as_of,
              fundamentals.share_class_reference.symbol,
              fundamentals.company_reference.standard_name,
              fundamentals.operation_ratios.total_debt_equity_ratio
              )
        #.filter(fundamentals.operation_ratios.total_debt_equity_ratio != None)
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
        .limit(2500) 
        ).T
    
    df = add_ebit_ev(df)
    df = scale(add_ranking_score(df),-0.1, 0.1)
    context.longs = df.sort(['rank'],ascending=False, na_position='last')['forecast'].head(context.numstocks)
    context.shorts = df.sort(['rank'],ascending=True, na_position='last')['forecast'].head(context.numstocks)

    delisted = []
    
    for stock in context.longs.index:
        if get_datetime() > stock.end_date + datetime.timedelta(days=3):
            delisted.append(stock)
    for stock in context.shorts.index:
        if get_datetime() > stock.end_date + datetime.timedelta(days=3):
            delisted.append(stock)
            
    context.longs.drop(delisted)
    context.shorts.drop(delisted)
    
    context.forecast = pd.concat([context.longs, context.shorts], axis=0)
    context.forecast.name = 'forecast'
  
    update_universe(context.forecast.index.values)
    context.universe = [stock for stock in context.forecast.index.values]

def isvalid(sid, data):
    #log.info(data[sid].returns())
    if sid in data:
        return data[sid].close_price > 1.5 and data[sid].returns() < 1 and data[sid].returns() > -1
    else:
        return False


# Computes the weights for the portfolio with the smallest Mean Absolute Deviation  
def minimum_MAD_portfolio(context, opt_data, returns):  

    return (True, (opt_data['longd'] - opt_data['shortd'])/100)
    #log.info(len(opt_data))
    #log.info(opt_data['longd'].dot(np.ones(len(opt_data))))
    #log.info(opt_data['shortd'].dot(np.ones(len(opt_data))))
                     
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
        return -opt_data['forecast'].dot(x) 
        #return (returns - returns.mean()).dot(x).abs().mean()
        #return -opt_data['forecast'].dot(x) + (returns - returns.mean()).dot(x).abs().mean()
        
    num_assets = len(opt_data)  
    guess = np.zeros(num_assets)  
    
    bnds = []
    limits_l = [0, 0.00]
    limits_s = [-0.04, 0]
       
    for stock in opt_data.index:
        if opt_data['longd'][stock] == 1:         
            bnds.append(limits_l)#[0, 0.04/opt_data['beta'][stock]])
        else:
            bnds.append(limits_s)#[-0.04/opt_data['beta'][stock], 0])
            #bnds.append(limits_s)    
    
    #log.info(bnds)
    #for stock in shorts:
     #   bnds.append(limits_s)
          
            
    bnds = tuple(tuple(x) for x in bnds)
    cons = ({'type':'ineq', 'fun': _leverage},
            {'type':'ineq', 'fun': _long_exposure},
            {'type':'ineq', 'fun': _short_exposure},
            {'type':'ineq', 'fun': _beta_exposure_l},
            {'type':'ineq', 'fun': _beta_exposure_g}
           ) 
    #min_mad_results = minimize(_mad, guess, args=returns[returns < 0], constraints=cons)  
    min_mad_results = minimize(_mad, guess, args = returns, constraints=cons, bounds=bnds,options =        {'maxiter':1000}, method = 'SLSQP')
    log.info(min_mad_results.success)
    log.info(opt_data['forecast'].dot(min_mad_results.x))
    log.info((returns - returns.mean()).dot(min_mad_results.x).abs().mean())
    log.info(opt_data['beta'].dot(min_mad_results.x))
    
    
    if min_mad_results.success:
        return (True, pd.Series(index=opt_data.index, data=min_mad_results.x))
    else:
        return (False, guess)
        
def rebalance(context, data):    
    
    desiredSids = context.forecast.index
    holdingSids = Series(context.portfolio.positions.keys())
    gettingTheBoot = holdingSids[holdingSids.isin(desiredSids) == False]
    for (ix,sid) in gettingTheBoot.iteritems():
        place_order(sid, 0)
   
    prices = history(250, "1d", 'price')
    log_price = np.log(prices).dropna(axis=1)
    returns = (log_price - log_price.shift(1)).dropna()
    
    beta = calculate_beta(context, data, prices)                      

    #avg_volume = history(22, "1d", 'volume').dropna(axis=1).mean()
    #avg_volume.name = 'volume'

    
    longd = Series(index = np.append([context.longs.index], [context.shorts.index]), 
                   data = np.append( [np.ones(len(context.longs))], [np.zeros(len(context.shorts))]), 
                   name = 'longd')
   
    shortd = Series(index = np.append([context.longs.index], [context.shorts.index]), 
                    data = np.append([np.zeros(len(context.longs))],[np.ones(len(context.shorts))]), 
                    name = 'shortd')
     
    opt_data = pd.concat([beta, longd, shortd, context.forecast], axis = 1, join = 'inner')
    
    opt_data = opt_data[opt_data.index.isin(df_intersection_index_col(opt_data, returns))]
    returns = returns[df_intersection_index_col(opt_data, returns)]
    
    (success, weights) = minimum_MAD_portfolio(context, opt_data, returns.tail(22))
    
    if success: 
        weights = round_portfolio(weights, context)
        for security in weights.keys(): 
                place_order(security, weights[security])

def calculate_beta(context, data, prices):
    returns = prices.pct_change()
    returns_stocks = returns[context.universe]
    returns_spy = returns[context.benchmarkSecurity]
    
    beta_span = 200
    benchVar = pd.stats.moments.ewmvar(returns_spy, span=beta_span)[beta_span:]
    cov = pd.stats.moments.ewmcov(returns_stocks, returns_spy, span=beta_span)[beta_span:]
    beta = cov.div(benchVar.ix[0]).iloc[-1]
    beta.name = 'beta'
    return beta

def calculate_beta_xyz(context, data):
    returns = history(250, '1d', 'price', ffill=True).pct_change()[1:]  

    index_returns = returns[context.benchmarkSecurity]  
    alphabeta = pd.DataFrame({'beta': np.zeros(len(context.universe)),
                  'alpha': np.zeros(len(context.universe))},
                  index = [context.universe])
    #log.info("Here")
    for asset in context.universe: 
        try:  
            X = returns[asset]  
            (alpha, beta) = linreg(X, index_returns)
            alphabeta['beta'][asset] = beta
            alphabeta['alpha'][asset] = alpha   
        except:  
            log.warn("[Failed Beta Calculation] asset = %s"%asset.symbol)  
            
    return alphabeta

def round_portfolio(portfol, context):   
    
    for security in portfol.keys(): 
        if abs(portfol[security]) < 0.0001 or sid is symbol('HVT_A'):
            portfol[security] = 0
            
                
    return portfol#/sum(portfol)
    
def df_intersection_index(df1, df2):
    return df2.index.isin(df1.index) & df1.index.isin(df2.index)

def df_intersection_index_col(df1, df2):
    return df2.keys().intersection(df1.index)

#def df_intersection_column(df1, df2):
 #   return df2.columns.intersection(df1.columns)
 
#def df_intersection_column_row(df1, df2):
    return df1.index.intersection(df2.keys())

#def df_intersection_row_column(df1, df2):
    return df1.keys().intersection(df2.index)

def linreg(x, y):
    # We add a constant so that we can also fit an intercept (alpha) to the model
    # This just adds a column of 1s to our data
    X = sm.add_constant(x)
    model = sm.OLS(y, X).fit()
    return (model.params[0], model.params[1])  

def place_order(sid, percent):    
    if get_datetime() < sid.end_date + datetime.timedelta(days=3):
        order_target_percent(sid, percent)
         