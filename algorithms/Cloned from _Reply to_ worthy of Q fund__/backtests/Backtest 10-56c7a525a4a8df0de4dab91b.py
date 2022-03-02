import numpy as np
import pandas as pd
import datetime

def initialize(context):
    
    context.eps = 1.1
    context.leverage = 1.0
    
    schedule_function(trade, date_rules.every_day(), time_rules.market_open(minutes=60))
    
    set_benchmark(symbol('QQQ'))
    
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
    
def handle_data(context, data):
    
    leverage = context.account.leverage
    
    if leverage >= 3.0:
        print "Leverage >= 3.0"
    
    record(leverage = leverage)
            
    for stock in context.stocks:
        if stock.security_end_date < get_datetime() + datetime.timedelta(days=5):  # de-listed ?
            context.stocks.remove(stock)
        if stock in security_lists.leveraged_etf_list: # leveraged ETF?
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
    
    for n in range(1,21):
        (weight,weighted_port) = get_weighted_port(data,context,prices.tail(n*390))
        sum_weighted_port += weighted_port
        sum_weights += weight
        
    allocation_optimum = sum_weighted_port/sum_weights
    record(ret = sum_weights / 20.0)
        
    rebalance_portfolio(data, context, allocation_optimum)
        
def get_weighted_port(data,context,prices):
    
    prices = pd.ewma(prices,span=390).as_matrix(context.stocks)
    
    b_t = np.zeros(len(context.stocks))
    
    # update portfolio
    for i, stock in enumerate(context.stocks):
        b_t[i] = abs(context.portfolio.positions[stock].amount*data[stock].price)
    
    denom = np.sum(b_t)
    # test for divide-by-zero case
    if denom > 0:
        b_t = np.divide(b_t,denom)
    else:     
        b_t = np.ones(len(context.stocks)) / len(context.stocks)

    x_tilde = np.zeros(len(context.stocks))

    b = np.zeros(len(context.stocks))
    
    context.ls = {}
    for stock in context.stocks:
        context.ls[stock] = 0
    
    # find relative moving volume weighted average price for each secuirty
    for i,stock in enumerate(context.stocks):
        mean_price = np.mean(prices[:,i])
        x_tilde[i] = (0.5*mean_price/prices[-1,i]) + (0.5*prices[-1,i]/mean_price)
        
    x_bar = x_tilde.mean()
    #x_tilde = x_tilde - x_bar + 1    
    
    for i,stock in enumerate(context.stocks):
        price_rel = x_tilde[i]
        if price_rel < 1.0:
            #price_rel = 1.0/price_rel
            context.ls[stock] = 1
        elif price_rel > 1.0:
            context.ls[stock] = 1
        else:
            context.ls[stock] += 0
            price_rel = 1.0
        #x_tilde[i] = price_rel
    
    ###########################
    # Inside of OLMAR (algo 2)

    
    
    record(x_bar = x_bar)

    # Calculate terms for lambda (lam)
    dot_prod = np.dot(b_t, x_tilde)
    num = context.eps - dot_prod
    denom = (np.linalg.norm((x_tilde-x_bar)))**2

    # test for divide-by-zero case
    if denom == 0.0:
        lam = 0 # no portolio update
    else:     
        lam = max(0, num/denom)
    
    b = b_t + lam*(x_tilde-x_bar)

    b_norm = simplex_projection(b)
    
    weight = np.dot(b_norm,x_tilde)
    
    #record(ret = weight)
    return (weight,weight*b_norm)

def rebalance_portfolio(data, context, desired_port):
    
    # check for open orders      
    for stock in context.stocks:
        if get_open_orders(stock):
            return
    
    pct_ls = 0
        
    for i, stock in enumerate(context.stocks):
        pct_ls += context.ls[stock]*desired_port[i]
        order_target_percent(stock, context.leverage*context.ls[stock]*desired_port[i])
        
    #order_target_percent(sid(19920), -context.leverage*pct_ls)
    
    record(pct_ls = pct_ls)
    
    for stock in data:
        if stock not in context.stocks + [sid(19920)]:
            order_target_percent(stock,0)

def simplex_projection(v, b=1):
    """Projection vectors to the simplex domain

Implemented according to the paper: Efficient projections onto the
l1-ball for learning in high dimensions, John Duchi, et al. ICML 2008.
Implementation Time: 2011 June 17 by Bin@libin AT pmail.ntu.edu.sg
Optimization Problem: min_{w}\| w - v \|_{2}^{2}
s.t. sum_{i=1}^{m}=z, w_{i}\geq 0

Input: A vector v \in R^{m}, and a scalar z > 0 (default=1)
Output: Projection vector w

:Example:
>>> proj = simplex_projection([.4 ,.3, -.4, .5])
>>> print proj
array([ 0.33333333, 0.23333333, 0. , 0.43333333])
>>> print proj.sum()
1.0

Original matlab implementation: John Duchi (jduchi@cs.berkeley.edu)
Python-port: Copyright 2012 by Thomas Wiecki (thomas.wiecki@gmail.com).
"""

    v = np.asarray(v)
    p = len(v)

    # Sort v into u in descending order
    v = (v > 0) * v
    u = np.sort(v)[::-1]
    sv = np.cumsum(u)

    rho = np.where(u > (sv - b) / np.arange(1, p+1))[0][-1]
    theta = np.max([0, (sv[rho] - b) / (rho+1)])
    w = (v - theta)
    w[w<0] = 0
    return w
