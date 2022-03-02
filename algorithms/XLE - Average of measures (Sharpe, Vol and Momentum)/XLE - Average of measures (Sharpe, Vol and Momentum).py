import numpy as np
import datetime
import scipy.stats as st   



def initialize(context):
    set_symbol_lookup_date('2015-05-01')
    context.index = sid(19655)
    set_benchmark(sid(19655))
    context.S=symbols('XOM', 'CVX', 'SLB', 'EOG', 'COP', 'OXY', 'HAL', 'WMB', 'VLO', 'TSO', 'SE', 'COG','NOV')
    
    schedule_function(myfunc,
                  date_rule=date_rules.week_start(),
                  time_rule=time_rules.market_open(hours=2))
    
            
    
def handle_data(context, data):
    record(l=context.account.leverage)
    pass

def cancelOpenOrders():
    oo = get_open_orders()
    if oo:
        for stock in oo:
            for oid in oo[stock]:
                cancel_order(oid)
  
def removedelistedstocks(context):
    for stock in context.S:  
        if (stock.security_end_date < (get_datetime()  + datetime.timedelta(days=5))):  
           log.info(1)
           context.S.remove(stock)
  
def removePositions(context):
    for stock in context.portfolio.positions:
        if stock not in context.S and stock is not context.index:
            order_target_percent(stock, 0)

def myfunc(context, data):
    
    removedelistedstocks(context)
    cancelOpenOrders()
    removePositions(context)
                     
    prices = history(90, "1d", "price",ffill = True).dropna(axis = 1)
        
    context.S = [asset for asset in prices
                       if asset != context.index]
     
    ret = np.log1p(prices).diff().dropna().values
     
    cumret = np.cumsum(ret, axis=0)
    
    lastret = cumret[-1,:]
    meanret = np.mean(lastret)
    stdret = np.std(lastret)
    
    i = 0
    score = []
    for sid in context.S:
        score.append((lastret[i] - meanret)/np.std(cumret[:,i]))
        i += 1
    
        
    netscore = np.sum(np.abs(score))
    
    val1 = np.abs(score)/netscore 

    score2 = score - np.mean(score)
    val2 = np.abs(score2) / np.sum(np.abs(score2))
    
    score3 = score - np.min(score)
    val3 = np.abs(score3) / np.sum(np.abs(score3))
    
    score4 = score - np.max(score)
    val4 = np.abs(score4) / np.sum(np.abs(score4))
    
    
    i = 0
    wsum = 0
    for sid in context.S:
        try:
            val = (val1[i] + val2[i] + val3[i] + val4[i])/4.0
            if val > 0:
                order_target_percent(sid, val)
                wsum += val
            elif val < 0:
                order_target_percent(sid, -val)
                wsum -= val
            else:
                order_target_percent(sid, 0)
                
        except:
            log.info("exception")
            continue
            
        i += 1
        
    try:
        order_target_percent(context.index, -wsum)
    except:
        log.info("exception xle")