# Adapted from:
# Li, Bin, and Steven HOI. "On-Line Portfolio Selection with Moving Average Reversion." The 29th International Conference on Machine Learning (ICML2012), 2012.
# http://icml.cc/2012/papers/168.pdf

import numpy as np
import pandas as pd
import datetime
from scipy import optimize


def initialize(context):
    
    
    context.eps = 0.005
    context.turnover = 0.05
    context.pct_index = 1.0 # max percentage of inverse ETF
    context.leverage = 1.0
    #context.stocks = symbols('GLD','OIL')
    context.indices = symbols('SPY','OIL','GLD','TLT')
    context.r1 = 1
    context.r2 = 21
    
    print 'context.eps = ' + str(context.eps)
    print 'context.pct_index = ' + str(context.pct_index)
    print 'context.leverage = ' + str(context.leverage)
    
    schedule_function(trade, date_rules.every_day(), time_rules.market_open(minutes=60))
    set_benchmark(symbol('QQQ'))
    
    context.data = []
 
def before_trading_start(context,data): 
    
    fundamental_df = get_fundamentals(
        query(
            fundamentals.valuation.market_cap,
        )
        .filter(fundamentals.company_reference.primary_exchange_id == 'NAS')
        .filter(fundamentals.valuation.market_cap != None)
        .order_by(fundamentals.valuation.market_cap.desc()).limit(50)) 
    update_universe(fundamental_df.columns.values)
    context.stocks = [stock for stock in fundamental_df]

    #context.stocks.append(symbols('SH')[0]) # add inverse ETF to universe
    #context.stocks.append(symbol('SPY'))
               
    
    # check if data exists
    #for stock in context.stocks:
     #   if stock not in context.data:
      #      context.stocks.remove(stock)
    
def handle_data(context, data):
    
    leverage = context.account.leverage
    
    #if leverage >= 3.0:
     #   print "Leverage >= 3.0"
    
    record(leverage = leverage)
            
    for stock in context.stocks:
        if stock.security_end_date < get_datetime() + datetime.timedelta(days=5):  # de-listed ?
            context.stocks.remove(stock)
        if stock in security_lists.leveraged_etf_list: # leveraged ETF?
            context.stocks.remove(stock)
        if (stock is sid(36929) or stock is sid(27358)) and stock in context.stocks:
            context.stocks.remove(stock)
            
    # check if data exists
    for stock in context.stocks:
        if stock not in data:
            context.stocks.remove(stock)

def trade(context,data):
    
    prices = history(20*390,'1m','price')[context.stocks].dropna(axis=1)
    context.stocks = list(prices.columns.values)
    
    # skip bar if any orders are open
    for stock in context.stocks:
        if bool(get_open_orders(stock)):
            return
    
    sum_weighted_port = np.zeros(len(context.stocks))
    sum_weights = 0
    
    for n in range(context.r1,context.r2):
        (weight,weighted_port) = get_allocation(context,data,prices.tail(n*390))
        sum_weighted_port += weight * weighted_port
        sum_weights += weight
        #log.info(weight)
        
    allocation_optimum = sum_weighted_port/sum_weights
    #log.info(sum(abs(allocation_optimum)))
        
    rebalance_portfolio(data, context, allocation_optimum)

    
def get_allocation(context,data,prices):
    
    prices = pd.ewma(prices,span=390).as_matrix(context.stocks)
    
    b_t = np.zeros(len(context.stocks))
    
    # update portfolio
    for i, stock in enumerate(context.stocks):
        b_t[i] = context.portfolio.positions[stock].amount*data[stock].price
         
    denom = np.sum(abs(b_t))
    # test for divide-by-zero case
    if denom > 0:
        b_t = np.divide(b_t,denom)
    else:
        b_t = np.zeros(len(context.stocks))
        #b_t = np.ones(len(context.stocks)) / len(context.stocks)

    x_tilde = np.zeros(len(context.stocks))
    
    b_0 = np.zeros(len(context.stocks)) / len(context.stocks)
    
    context.ls = {}
    for stock in context.stocks:
        context.ls[stock] = 0
    
    # find relative moving volume weighted average price for each secuirty
    for i,stock in enumerate(context.stocks):
        mean_price = np.mean(prices[:,i])
        x_tilde[i] = (1.0*mean_price/prices[-1,i]) - 1 + (0.0*prices[-1,i]/mean_price)
    
    for i,stock in enumerate(context.stocks):
        price_rel = x_tilde[i]
        context.ls[stock] = 1
        if price_rel < 0 and price_rel > -0.003:
            price_rel = 0 #1.0/price_rel
            context.ls[stock] = 1
        elif price_rel > 0 and price_rel < 0.003:
            price_rel = 0
            context.ls[stock] = 1
        log.info(price_rel)    
        x_tilde[i] = price_rel
        
    x_bar = x_tilde.mean()
    #x_tilde = x_tilde - x_bar + 1
        
    bnds = []
    limits = [-1, 1]
    
    for stock in context.stocks:
            bnds.append(limits)
        
    bnds = tuple(tuple(x) for x in bnds)
     
    cons = ({'type': 'ineq', 'fun': lambda x:  np.sum(abs(x)) - 2.0},{'type': 'ineq', 'fun': lambda x:  2.5 - np.sum(abs(x))},{'type': 'ineq', 'fun': lambda x:  np.sum(x)+0.95}, {'type': 'ineq', 'fun': lambda x:  -np.sum(x)+0.95},{'type': 'ineq', 'fun': lambda x: np.dot(x,x_tilde) - context.eps})# {'type': 'ineq','fun': lambda x: context.turnover - np.sum(abs(x-b_t))})
    
    
   
   # res = optimize.minimize(totalreturn, b_t, args=x_tilde, method='SLSQP', constraints=cons, bounds=bnds)
    
    res = optimize.minimize(norm_squared, b_0, args=(b_t,x_tilde),method='SLSQP',constraints=cons,bounds=bnds, options={'disp': False,  'maxiter': 1000, 'iprint': 1, 'ftol': 1e-6})
    
    
    allocation = res.x
    log.info(np.sum(abs(allocation)))
    log.info(allocation)
    #allocation[allocation<0] = 0 
    #allocation = allocation/np.sum(abs(allocation))
    
    if res.success and (np.dot(allocation,x_tilde) - context.eps > 0.0):
       #log.info("Success")
       return (np.dot(allocation,x_tilde), allocation)
    else:
       #log.info("Failure") 
       return (1,b_t)
    
    

def rebalance_portfolio(data, context, desired_port):
    
    record(long = sum(desired_port))
    #record(inverse = desired_port[-1])
    
    for i, stock in enumerate(context.stocks):
        order_target_percent(stock, context.leverage*context.ls[stock]*desired_port[i])
    
    for stock in data:
        if stock not in context.stocks and stock not in context.indices:
            order_target_percent(stock,0)     
       
        
def norm_squared(b,*args):
    
    b_t = np.asarray(args[0])
        
    delta_b = b - b_t
     
    return 0.5*np.dot(delta_b,delta_b.T)

def norm_squared_deriv(b,*args):
    
    b_t = np.asarray(args)
    delta_b = b - b_t
        
    return delta_b

def totalreturn(b, *args): 
    u = np.array(args[0]) #np.asarray(args)
   # log.info(u.shape)
    u = u.T
   # log.info(u.shape)
    return -np.dot(b, u)
                              
def totalreturn_grad(*args):     return -np.array(args[0])
                                                                                                                                     
                                                                                                                                            
                             
                                                                                                                                            
                                                                                                                                           
