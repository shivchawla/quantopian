# Adapted from:
# Li, Bin, and Steven HOI. "On-Line Portfolio Selection with Moving Average Reversion." The 29th International Conference on Machine Learning (ICML2012), 2012.
# http://icml.cc/2012/papers/168.pdf

import numpy as np
from scipy import optimize
import pandas as pd



def initialize(context):
    
    
    context.eps = 1.005
    context.turnover = 0.05
    context.pct_index = 1.0 # max percentage of inverse ETF
    context.leverage = 1.0
    #context.stocks = symbols('GLD','OIL')
    context.indices = symbols('SPY','OIL','GLD','TLT')
    
    print 'context.eps = ' + str(context.eps)
    print 'context.pct_index = ' + str(context.pct_index)
    print 'context.leverage = ' + str(context.leverage)
    
    schedule_function(trade, date_rules.every_day(), time_rules.market_open(minutes=2))
    
    
    context.data = []
 
def before_trading_start(context): 
    
    fundamental_df = get_fundamentals(
        query(
            fundamentals.valuation.market_cap,
        )
        #.filter(fundamentals.company_reference.primary_exchange_id == 'NAS')
        .filter(fundamentals.valuation.market_cap != None)
        .order_by(fundamentals.valuation.market_cap.desc()).limit(100)) 
    update_universe(fundamental_df.columns.values)
    context.stocks = [stock for stock in fundamental_df][-20:]

    #context.stocks.append(symbols('SH')[0]) # add inverse ETF to universe
    #context.stocks.append(symbol('SPY'))
               
    
    # check if data exists
    for stock in context.stocks:
        if stock not in context.data:
            context.stocks.remove(stock)
    
    # Update context.fundamental_df with the securities (and pe_ratio) that we need
    context.fundamental_df = fundamental_df[context.stocks]

    #context.stocks.append(symbol('SH'))
    #context.more = symbols('GLD','TLT','OIL')
    #for stock in context.more:
        #context.stocks.append(stock)
    #context.stocks=symbols('XLE','XLP','XLY','XLF','XLV','XLI','XLB','XLK','XLU')
    
    
def handle_data(context, data):
    
    record(leverage = context.account.leverage)
    
    context.data = data
    for stock in context.stocks:
        if stock not in data:
            context.stocks.remove(stock)

def get_inefficiency(context):
    
    ineff = {}
    #ineff[symbol('SH')] = 1
   # ineff[symbol('SPY')] = 1
   
    #context.fundamental_df.sort(columns = 'market_cap', axis=0, ascending=False, inplace = True)
    
    #context.fundamental_df.transpose()
    
   
    
   
    
    #log.info(tsum)
    csum = 0        
    for stock in context.stocks:
        
            #csum = csum + pow(context.fundamental_df[stock]['market_cap'],(1/3.0))
        ineff[stock] = 1.0
            #log.info(stock)
            #log.info(float(csum)/tsum)
           
    
    return ineff    
    
def get_allocation(context,data,n,prices):
    
    prices = pd.ewma(prices,span=390).as_matrix(context.stocks)
    
    b_t = []
    
    for stock in context.stocks:
        if stock not in context.indices:
           b_t.append(context.portfolio.positions[stock].amount*data[stock].price)
         
    m = len(b_t)
   
    #mcap weighted portfolio
    b_0 = np.zeros(m)
     #sum the market cap
    summktcap = context.fundamental_df.sum(axis=1)['market_cap']
    #log.info(summktcap)
    for i, stock in enumerate(context.stocks):
        if stock is not symbol('SH'):
           mcap = context.fundamental_df[stock]['market_cap']
           b_0[i] = float(mcap)/summktcap
     
   # b_0 = np.ones(m) / m  # equal-weight portfolio
    denom = np.sum(b_t)
    #log.info(denom)
    
    Inefficiency = get_inefficiency(context)

    if denom == 0.0:
        b_t = np.copy(b_0)
    else:     
        b_t = np.divide(b_t,denom)
    
    x_tilde = []
    
    for i, stock in enumerate(context.stocks):
        #log.info(np.mean(prices[:,i]))
        mean_price = np.mean(prices[:,i])
        #log.info(mean_price/prices[-1,i])
        #log.info(np.mean(prices[:,i])/prices[-1,i])
        x_tilde.append(((mean_price/prices[-1,i])-1)*Inefficiency[stock] + 1)
        
    bnds = []
    limits = [0.0,0.2]
    
    for stock in context.stocks:
        if stock is symbol('SH'):
            bnds.append([0, 0.5])
        elif stock not in context.indices:
            bnds.append(limits)
        else :
            bnds.append([0,context.pct_index])
    
    #bnds[-1] = [0,context.pct_index] # limit exposure to index
        
    bnds = tuple(tuple(x) for x in bnds)
     
    cons = ({'type': 'eq', 'fun': lambda x:  np.sum(x)-1.0},{'type': 'ineq', 'fun': lambda x: np.dot(x,x_tilde) - context.eps})# {'type': 'ineq','fun': lambda x: context.turnover - np.sum(abs(x-b_t))})
    
    
   
   # res = optimize.minimize(totalreturn, b_t, args=x_tilde, method='SLSQP', constraints=cons, bounds=bnds)
    
    res = optimize.minimize(norm_squared, b_0, args=(b_t,x_tilde),method='SLSQP',constraints=cons,bounds=bnds, options={'disp': False,  'maxiter': 100, 'iprint': 1, 'ftol': 1e-6})
    
    
    allocation = res.x
    allocation[allocation<0] = 0 
    allocation = allocation/np.sum(allocation)
    
    if res.success and (np.dot(allocation,x_tilde)-context.eps > 0.0):
       #log.info("success")
       return (allocation,np.dot(allocation,x_tilde))
    else:
       return (b_t,1)
    
     
def trade(context,data):
    
    # check if data exists
    for stock in context.stocks:
        if stock not in data:
            context.stocks.remove(stock)
        
    # check for de-listed stocks & leveraged ETFs
    for stock in context.stocks:  
        if stock.security_end_date < get_datetime():  # de-listed ?  
            context.stocks.remove(stock)
        if stock in security_lists.leveraged_etf_list: # leveraged ETF?
            context.stocks.remove(stock)
    
    # check for open orders      
    if get_open_orders():
        return
    
    # find average weighted allocation over range of trailing window lengths
    a = np.zeros(len(context.stocks))
    prices=history(8*390,'1m','price')
    w=0
    for n in range(3,9):
        (a,w) = get_allocation(context,data,n,prices.tail(n*390))
        a += w*a
        w += w
    
    allocation = a/w
    if np.sum(allocation)!= 0:
        allocation = allocation/np.sum(allocation)
    
    allocate(context,data,allocation)

def allocate(context, data, desired_port):
    
    record(long = sum(desired_port[0:-1]))
    #record(inverse = desired_port[-1])
    
    for i, stock in enumerate(context.stocks):
        order_target_percent(stock, context.leverage*desired_port[i])
    
    for stock in data:
        if stock not in context.stocks and stock not in context.indices:
            order_target_percent(stock,0)
    
    #order_target_percent(symbol('SPY'), -context.leverage*0.64)
    
   # order_target_percent(symbol('TLT'), 0.08)
    
    #order_target_percent(symbol('GLD'),-0.15)
    #order_target_percent(symbol('OIL'),-0.36)
   #order_target_percent(symbol('IYR'),-0.25)
        
       
        
def norm_squared(b,*args):
    
    b_t = np.asarray(args[0])
    u=np.array(args[1])
        
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
                                                                                                                                     
                                                                                                                                            
                             
                                                                                                                                            
                                                                                                                                           