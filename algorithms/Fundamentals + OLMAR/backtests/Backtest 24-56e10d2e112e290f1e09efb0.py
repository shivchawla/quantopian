
import numpy as np
import pandas as pd
import datetime
import random
from scipy import optimize


def initialize(context):
    
    
    context.eps = 1.005
    context.pct_index = 1.0 # max percentage of inverse ETF
    context.leverage = 1.0
    context.indices = symbols('SPY','OIL','GLD','TLT')
    context.r1 = 15
    context.r2 = 20
    context.btret = 0.1
    
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
            fundamentals.asset_classification.morningstar_sector_code
        )
        .filter(fundamentals.company_reference.primary_exchange_id == 'NAS')
        .filter(fundamentals.company_reference.primary_exchange_id != "OTCPK") # no pink sheets
        .filter(fundamentals.company_reference.primary_exchange_id != "OTCBB") # no pink sheets
        .filter(fundamentals.share_class_reference.is_depositary_receipt == False) # !ADR/GDR
        .filter(fundamentals.share_class_reference.is_primary_share == True) # remove ancillary classes
        .filter(fundamentals.valuation.market_cap != None)
        .filter(fundamentals.valuation.shares_outstanding != None)
        .filter(fundamentals.valuation.market_cap > 100000000)
        #.filter(fundamentals.asset_classification.morningstar_sector_code != 206)
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
        #print "Leverage >= 3.0"
    
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
    
    if len(context.stocks) == 0 :
        return
    
    # skip bar if any orders are open
    for stock in context.stocks:
        if bool(get_open_orders(stock)):
            return
    
    sum_weighted_port_l = np.zeros(len(context.stocks))
    sum_weights_l = 0
    
    sum_weighted_port_s = np.zeros(len(context.stocks))
    sum_weights_s = 0
    iteration_l = 0
    iteration_s = 0
    
    
    for n in range(context.r1,context.r2):
        (weight_l,weighted_port_l, weight_s,weighted_port_s, x_l, x_s) = get_allocation(context,data,prices.tail(n*390))
        
        #log.info(weight_l)
        sum_weighted_port_l += weighted_port_l
        sum_weights_l += weight_l
        sum_weighted_port_s += weighted_port_s
        sum_weights_s += weight_s
        iteration_l += x_l
        iteration_s += x_s
        
    allocation_optimum_l = sum_weighted_port_l/sum_weights_l 
    allocation_optimum_s = sum_weighted_port_s/sum_weights_s
        
    allocation_optimum = allocation_optimum_l - allocation_optimum_s
    #allocation_optimum = - allocation_optimum_s 
        
    rebalance_portfolio(data, context, allocation_optimum)

    
def get_allocation(context,data,prices):
    
    prices = pd.ewma(prices,span=390).as_matrix(context.stocks)
    
    b_t_long = np.zeros(len(context.stocks))
    b_t_short = np.zeros(len(context.stocks))
    
    # update portfolio
    for i, stock in enumerate(context.stocks):
        if context.portfolio.positions[stock].amount >0:
            b_t_long[i] = context.portfolio.positions[stock].amount*data[stock].price
        else:
            b_t_short[i] = context.portfolio.positions[stock].amount*data[stock].price
         
    denom_long = np.sum(b_t_long)
    denom_short = np.sum(b_t_short)
    
    # test for divide-by-zero case
    if denom_long > 0:
        b_t_long = np.divide(b_t_long,denom_long)
    else:     
        b_t_long = np.ones(len(context.stocks)) / len(context.stocks)

    if denom_short < 0:
        b_t_short = np.divide(b_t_short,denom_short)
    else:     
        b_t_short = np.ones(len(context.stocks)) / len(context.stocks)
        
        
    x_tilde_long = np.zeros(len(context.stocks))
    x_tilde_short = np.zeros(len(context.stocks))
    
    b_0 = np.ones(len(context.stocks)) / len(context.stocks)
    
    context.ls = {}
    for stock in context.stocks:
        context.ls[stock] = 0
    
    # find relative moving volume weighted average price for each secuirty
    for i,stock in enumerate(context.stocks):
        mean_price = np.mean(prices[:,i])
        x_tilde_long[i] = (1.0*mean_price/prices[-1,i])
        x_tilde_short[i] = (1.0*prices[-1,i]/mean_price)
    
    for i,stock in enumerate(context.stocks):
        price_rel_long = x_tilde_long[i]
        price_rel_short = x_tilde_short[i]
                      
        if price_rel_long < 1.0:
            price_rel_long = 0
        if price_rel_short < 1.0:
            price_rel_short = 0
        x_tilde_long[i] = price_rel_long
        x_tilde_short[i] = price_rel_short
                
        
    bnds_l = []
    bnds_s = []
    limits_l = [0.0, 1]
    limits_s = [0.0, 1]
    
    for stock in context.stocks:
            bnds_l.append(limits_l)
            bnds_s.append(limits_s)
            
        
    bnds_l = tuple(tuple(x) for x in bnds_l)
    bnds_s = tuple(tuple(x) for x in bnds_s)
     
    cons_long = ({'type': 'eq', 'fun': lambda x:  np.sum(x)-1.0},{'type': 'ineq', 'fun': lambda x: np.dot(x,x_tilde_long) - context.eps})# {'type': 'ineq','fun': lambda x: context.turnover - np.sum(abs(x-b_t))})
    cons_short = ({'type': 'eq', 'fun': lambda x:  np.sum(x)-1.0},{'type': 'ineq', 'fun': lambda x: np.dot(x,x_tilde_short) - context.eps})# {'type': 'ineq','fun': lambda x: context.turnover - np.sum(abs(x-b_t))})
    
    
   
   # res = optimize.minimize(totalreturn, b_t, args=x_tilde, method='SLSQP', constraints=cons, bounds=bnds)
    
    res_long = optimize.minimize(norm_squared, b_0, args=(b_t_long,x_tilde_long),method='SLSQP',constraints=cons_long,bounds=bnds_l, options={'disp': False,  'maxiter': 100, 'iprint': 1, 'ftol': 1e-6})
   
    res_short = optimize.minimize(norm_squared, b_0, args=(b_t_short,x_tilde_short),method='SLSQP',constraints=cons_short,bounds=bnds_s, options={'disp': False,  'maxiter': 100, 'iprint': 1, 'ftol': 1e-6})
        
    
    allocation_l = res_long.x
    allocation_l[allocation_l<0] = 0 
    allocation_l = allocation_l/np.sum(allocation_l)
    
    allocation_s = res_short.x
    allocation_s[allocation_s<0] = 0 
    allocation_s = allocation_s/np.sum(allocation_s)
    #b_t_short = np.zeros(len(context.stocks))                  
    #allocation_s = np.zeros(len(context.stocks))
    
    if res_long.success and (np.dot(allocation_l,x_tilde_long) - context.eps > 0.0) and res_short.success and (np.dot(allocation_s,x_tilde_short) - context.eps) > 0.0:
        return (np.dot(allocation_l,x_tilde_long), np.dot(allocation_l,x_tilde_long)*allocation_l, np.dot(allocation_s,x_tilde_short), np.dot(allocation_s,x_tilde_short)*allocation_s, res_long.nit, res_short.nit)
    elif res_long.success and (np.dot(allocation_l,x_tilde_long) - context.eps > 0.0):
       return (np.dot(allocation_l,x_tilde_long), np.dot(allocation_l,x_tilde_long)*allocation_l, 0.1, 0.1* b_t_short, res_long.nit, res_short.nit)
    elif res_short.success and (np.dot(allocation_s,x_tilde_short) - context.eps > 0.0):
       return (0.1, 0.1*b_t_long, np.dot(allocation_s,x_tilde_short), np.dot(allocation_s,x_tilde_short)*allocation_s, res_long.nit, res_short.nit)
    else:            
        return (0.1, 0.1*b_t_long, 0.1, 0.1* b_t_short, res_long.nit, res_short.nit)
    
def rebalance_portfolio(data, context, desired_port):
    
    record(long = sum(desired_port))
    #record(inverse = desired_port[-1])
    
    for i, stock in enumerate(context.stocks):
        order_target_percent(stock, context.leverage*desired_port[i])
    
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