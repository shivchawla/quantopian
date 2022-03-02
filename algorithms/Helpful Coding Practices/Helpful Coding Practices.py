    # Important python modules
    from scipy import stats
    import numpy as np
    from pytz import timezone
    import pandas as pd

    # Calclating rolling stats using a pandas and dataframes
    high = history(60, "1d", "high")  
    low = history(60, "1d", "low")
    #below gives 10 day rolling max and min
    rolling_max = pandas.stats.moments.rolling_max(high,10)
    rolling_min = pandas.stats.moments.rolling_min(low,10)
    p = pd.rolling_mean(context.prices,window_roll)[window_roll-1:]
    v = pd.rolling_sum(context.volumes,window_roll)[window_roll-1:]
    
    
    #Define a dictionary
    context.LastStop={}

    # Define a list 
    context.MyList=[]

    # Apending to a list
    context.MyList.append(x) 
    
    # Rolling (shifting values in a list)
    context.prices = np.roll(context.prices,-1,axis=0)
    
    # Sort a list  
    BuyList=sorted(StdList.items(),key=operator.itemgetter(1),reverse=True)

    # Runing a OLS regression
    # Is stock trending up ?
    Y = dfHistD[S].values/dfHistD[S].mean()
    X = range(len(dfHistD))
    # Add column of ones so we get intercept
    A=sm.add_constant(X)
    results = sm.OLS(Y,A).fit()
    InterB,SlopeM=results.params


    # Canceling any open orders
    orders = get_open_orders()
    for sid in orders:
        for oid in orders[sid]:
            cancel_order(oid)

    # Fetching data from external URL
    fetch_csv(vixUrl, 
              symbol='VIX', 
              skiprows=1,
              date_column='Date', 
              pre_func=addFieldsVIX) 
    
    # Get current datetime 
    current_datetime = get_datetime().astimezone(timezone('US/Eastern'))
    
    # Check if stock did not trade
    data[stock].datetime < get_datetime()
    
    
    
    #### Book-keeping and calculating Alpha/beta/other stats ##
    def initialize(context):  
    context.equity = Series()  
    context.benchmark = Series()  
    context.benchmarkSecurity = symbol('SPY')  
    set_benchmark(context.benchmarkSecurity)  

    schedule_function(func=T1559_closing_bookkeeping,  
                      date_rule=date_rules.every_day(),  
                      time_rule=time_rules.market_close(minutes=1),  
                      half_days=True  
                      )  
. . .
def T1559_closing_bookkeeping(context, data):  
    try:  
        record_equity(context, data)  
    except:  
        log.error("Exception while recording equity-based stats")  
. . .
def record_equity(context, data):  
    context.equity = context.equity.append(Series({get_datetime(): context.portfolio.portfolio_value}))  
    context.benchmark = context.benchmark.append(Series({get_datetime(): data[context.benchmarkSecurity].close_price}))  
    recordCorr(context)  
. . .
def recordCorr(context):  
    record( leverage=context.account.leverage )  
    span = 60.0  
    if (context.equity.size > span):  
        equityReturns = context.equity.pct_change()  
        benchmarkReturns = context.benchmark.pct_change()  
        benchVar = pd.stats.moments.ewmvar(benchmarkReturns, span=span)  
        cov = pd.stats.moments.ewmcov(equityReturns, benchmarkReturns, span=span)  
        record(emwv_3m_beta = cov.tail(1).item() / benchVar.tail(1).item())  
. . .    

def get_alphas_and_betas(context, data):  
    """  
    returns a dataframe of 'alpha' and 'beta' exposures  
    for each asset in the current universe.  
    """  
    prices = history(context.lookback, '1d', 'price', ffill=True)  
    returns = prices.pct_change()[1:]  
    index_returns = returns[context.index]  
    factors = {}  
    for asset in context.portfolio.positions:  
        try:  
            X = returns[asset]  
            factors[asset] = linreg(X, index_returns)  
        except:  
            log.warn("[Failed Beta Calculation] asset = %s"%asset.symbol)  
    return pd.DataFrame(factors, index=['alpha', 'beta'])  


   