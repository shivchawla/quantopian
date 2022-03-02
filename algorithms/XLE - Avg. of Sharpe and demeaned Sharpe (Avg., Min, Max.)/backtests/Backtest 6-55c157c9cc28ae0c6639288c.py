import numpy as np
import datetime
import scipy.stats as st   



def initialize(context):
    set_symbol_lookup_date('2015-05-01')
    context.XLE = sid(19655)
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
           order_target_percent(stock, 0)
           context.S.remove(stock)
  
    
def myfunc(context, data):
    
    removedelistedstocks(context)
    cancelOpenOrders()
    
                     
    prices = history(90, "1d", "price")
    prices = prices.drop([context.XLE], axis=1)
    #ret = prices.pct_change(5).dropna().values
    #ret = np.log1p(ret)
    ret = np.log1p(prices).diff().dropna().values
   
    cumret = np.cumsum(ret, axis=0)
    lastret = cumret[-1,:]
    meanret = np.mean(lastret)
    stdret = np.std(lastret)
    
    i = 0
    score = []
    for sid in prices:
        score.append((lastret[i] - meanret)/ np.std(cumret[:,i])) #
        i += 1
        
    record(Mean = np.mean(score), Min = min(score), Max = max(score), Skew = st.skew(score))
   
    #score = score - np.min(score)  
    #if st.skew(lastret) > 0:
     #   score = score - np.min(score)
    #else:
     #   score = score - np.max(score)
    
        
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
    for sid in prices:
        try:
            val = (val1[i] + val2[i] + val3[i] + val4[i])/4.0
            if val > 0:
                order_target_percent(sid, val)
                wsum += val
            elif val < 0:
                x=1
                order_target_percent(sid, -val)
                wsum -= val
            else:
                order_target_percent(sid, 0)
                
        except:
            log.info("exception")
            continue
            
        i += 1
        
    try:
        x=1
        order_target_percent(context.XLE, -wsum)
    except:
        log.info("exception xle")