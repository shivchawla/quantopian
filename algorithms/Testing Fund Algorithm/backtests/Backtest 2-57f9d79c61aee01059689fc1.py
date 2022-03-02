import numpy as np
import pandas as pd
import datetime
import random
from scipy import optimize


def initialize(context):
       
    context.eps = 1.005
    context.leverage = 1.0
    
    context.indices = symbols('SPY','OIL','GLD','TLT')
    context.r1 = 1
    context.r2 = 21
    
    schedule_function(trade, date_rules.every_day(), time_rules.market_open(minutes=60))
    context.data = []
 
def before_trading_start(context,data): 
    
    fundamental_df = get_fundamentals(
        query(
            fundamentals.valuation.market_cap,
        )
        .filter(fundamentals.company_reference.primary_exchange_id == 'NAS')
        .filter(fundamentals.valuation.market_cap != None)
        .filter(fundamentals.valuation.shares_outstanding != None)  
        .filter(fundamentals.valuation.market_cap > 100000000) # cap > $100MM 
        .filter(fundamentals.share_class_reference.security_type == 'ST00000001') # common stock only
        .filter(~fundamentals.company_reference.standard_name.contains(' LP')) # exclude LPs
        .filter(~fundamentals.company_reference.standard_name.contains(' L P'))
        .filter(~fundamentals.company_reference.standard_name.contains(' L.P'))
        .filter(fundamentals.company_reference.primary_exchange_id != "OTCPK") # no pink sheets
        .filter(fundamentals.company_reference.primary_exchange_id != "OTCBB") # no pink sheets
        .filter(~fundamentals.share_class_reference.symbol.contains('_WI')) # drop when-issued
        .filter(fundamentals.share_class_reference.is_primary_share == True) # remove ancillary classes
        .filter(fundamentals.share_class_reference.is_depositary_receipt == False) # !ADR/GDR
        .order_by(fundamentals.valuation.market_cap.desc()).limit(40))
    context.stocks = [stock for stock in fundamental_df]
    
def handle_data(context, data):
    return
               
        
def basictest(context, data):
    for stock in context.stocks:
        if stock.security_end_date < get_datetime() + datetime.timedelta(days=5):  # de-listed ?
            context.stocks.remove(stock)
        if stock in security_lists.leveraged_etf_list: # leveraged ETF?
            context.stocks.remove(stock)
        if (stock is sid(36929) or stock is sid(27358)) and stock in context.stocks:
            context.stocks.remove(stock)
            
    # check if data exists
    for stock in context.stocks:
        if data.can_trade(stock)==False:
            context.stocks.remove(stock)

def trade(context, data):
    
    basictest(context, data)
            
    prices = data.history(context.stocks, 'price', 20*390, '1m').dropna(axis=1)
    context.stocks = list(prices.columns.values)
    
    if len(context.stocks) == 0: 
        return
    
    # skip bar if any orders are open
    for stock in context.stocks:
        if bool(get_open_orders(stock)):
            return
    
    allocation_optimum = getportfolio(context, data, prices)
     
    rebalance_portfolio(data, context, allocation_optimum)
      
    record(leverage = context.account.leverage)

def getportfolio(context, data, prices):

    sum_weighted_port_l = np.zeros(len(context.stocks))
    sum_weights_l = 0
    
    sum_weighted_port_s = np.zeros(len(context.stocks))
    sum_weights_s = 0
    iteration_l = 0
    iteration_s = 0
    
    context.cs = np.zeros(len(context.stocks), dtype=bool)
    context.cl = np.zeros(len(context.stocks), dtype=bool)
     
    for n in range(context.r1,context.r2):
        (weight_l,weighted_port_l, weight_s,weighted_port_s, x_l, x_s) = get_allocation(context,data,prices.tail(n*390))
        
        sum_weighted_port_l += weighted_port_l
        sum_weights_l += weight_l
        sum_weighted_port_s += weighted_port_s
        sum_weights_s += weight_s
        
        
    allocation_optimum_l = sum_weighted_port_l/sum_weights_l 
    allocation_optimum_s = sum_weighted_port_s/sum_weights_s
    
    y = allocation_optimum_l - allocation_optimum_s
    x = np.sum(abs(y))
    
    if x == 0.0:
        return y
    else:
        return (1.0/x) * y
    

def get_allocation(context,data, prices):
    
    prices = pd.ewma(prices,span=390).as_matrix(context.stocks)
    
    b_t_long = np.zeros(len(context.stocks))
    b_t_short = np.zeros(len(context.stocks))
    
    # update portfolio
    for i, stock in enumerate(context.stocks):
        if context.portfolio.positions[stock].amount > 0:
            b_t_long[i] = context.portfolio.positions[stock].amount*data.current(stock,'price')
        else:
            b_t_short[i] = context.portfolio.positions[stock].amount*data.current(stock,'price')
         
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
                      
        if price_rel_long < 1.0 or price_rel_long > 1.003:
            price_rel_long = 0
        if price_rel_short < 1.0 or price_rel_short > 1.003:
            price_rel_short = 0
        x_tilde_long[i] = price_rel_long
        x_tilde_short[i] = price_rel_short
                
        
    bnds_l = []
    bnds_s = []
    limits_l = [0.0, 0.2]
    limits_s = [0.0, 0.2]
    
    for i, stock in enumerate(context.stocks):
            bnds_l.append(limits_l)
            bnds_s.append(limits_s)
               
     
    cons_long = ({'type': 'eq', 'fun': lambda x:  np.sum(x)-1.0},{'type': 'ineq', 'fun': lambda x: np.dot(x,x_tilde_long) - context.eps})
    
    cons_short = ({'type': 'eq', 'fun': lambda x:  np.sum(x)-1.0},{'type': 'ineq', 'fun': lambda x: np.dot(x,x_tilde_short) - context.eps})
    
       
    res_long = optimize.minimize(norm_squared, b_0, args=(b_t_long,x_tilde_long),method='SLSQP',constraints=cons_long,bounds=bnds_l, options={'disp': False,  'maxiter': 100, 'iprint': 1, 'ftol': 1e-6})
   
    res_short = optimize.minimize(norm_squared, b_0, args=(b_t_short,x_tilde_short),method='SLSQP',constraints=cons_short,bounds=bnds_s, options={'disp': False,  'maxiter': 100, 'iprint': 1, 'ftol': 1e-6})
        
    
    allocation_l = res_long.x
    allocation_l[allocation_l<0] = 0 
    allocation_l = allocation_l/np.sum(allocation_l)
    
    allocation_s = res_short.x
    allocation_s[allocation_s<0] = 0 
    allocation_s = allocation_s/np.sum(allocation_s)
    
    
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
    
    trades = np.zeros(len(context.stocks))
    
    #message= '{symbol} {amount}'
    for i, stock in enumerate(context.stocks):
        trades[i] = desired_port[i]*context.portfolio.portfolio_value - context.portfolio.positions[stock].amount*data.current(stock,'price')
        
        
    log.info("LONG Targets")     
    for i, stock in enumerate(context.stocks):
   
         if trades[i] > 0.0:
           message = '{symbol} {amount}' 
           message = message.format(symbol=stock.symbol, amount= trades[i])  
           log.info(message)
    
    log.info("SHORT Targets")       
    for i, stock in enumerate(context.stocks):
   
         if trades[i] < 0.0:
           message = '{symbol} {amount}' 
           message = message.format(symbol=stock.symbol, amount=trades[i])  
           log.info(message)        
             
    
    for i, stock in enumerate(context.stocks):
           order_target_percent(stock, context.leverage*desired_port[i])
         
    for stock in context.portfolio.positions:
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
    u = np.array(args[0]) 
    u = u.T
    return -np.dot(b, u)
                              
def totalreturn_grad(*args):     
    return -np.array(args[0])
