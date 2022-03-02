import numpy as np
import datetime
import scipy.stats as st        

def initialize(context):
    context.index = sid(19658)
    
    context.S = [sid(24), sid(5061), sid(21839), sid(3766), sid(35920), sid(1900), sid(5692), sid(3951), sid(6295), sid(32146), sid(25555), sid(2518)]
     
    set_benchmark(sid(19658))

    schedule_function(myfunc,
                  date_rule=date_rules.week_start(),
                  time_rule=time_rules.market_open(hours=2))
    
def before_trading_start(context):
    # Number of stocks to find
    num_stocks = 50
    
    fundamental_df = get_fundamentals(
        query(
            fundamentals.valuation.market_cap,
        )
        .filter(fundamentals.valuation.market_cap > 1e9)
        #.filter(fundamentals.income_statement.ebit < 0)
        .filter(fundamentals.asset_classification.morningstar_sector_code == 311)
        .order_by(fundamentals.valuation.market_cap.desc())
    )
    
    context.S = [stock for stock in fundamental_df] 
    
    
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
    
        
    score = score - np.max(score)    
    netscore = np.sum(np.abs(score))
    
    i = 0
    wsum = 0
    for sid in context.S:
        try:
            val = score[i]/netscore
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