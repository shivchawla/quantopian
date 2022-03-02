
from quantopian.algorithm import attach_pipeline, pipeline_output
from quantopian.pipeline import Pipeline
from quantopian.pipeline import CustomFactor
from quantopian.pipeline.factors import SimpleMovingAverage, AverageDollarVolume 
from quantopian.pipeline.data.builtin import USEquityPricing

from quantopian.pipeline.data import morningstar

from pandas import Series, DataFrame
import pandas as pd
import statsmodels
import statsmodels.api 
import datetime as dt
import datetime as datetime
import numpy as np
import math
from brokers.ib import VWAPBestEffort
import scipy.stats as st
from sklearn.decomposition import PCA
from scipy.optimize import minimize


class UniverseFilter(CustomFactor):  
    """  
    Return 1.0 for the following class of assets, otherwise 0.0:  
      * No Financials (103), Real Estate (104), Basic Materials (101) and ADR  
        (Basic Materials are too much sensitive to exogenous macroeconomical shocks.)  
      * Only primary common stocks  
      * Exclude When Distributed(WD), When Issued(WI) and VJ - usuallly companies in bankruptcy  
      * Exclude Halted stocks (_V, _H)  
      * Only NYSE, AMEX and Nasdaq  
    """  
    window_length = 1  
    inputs = [USEquityPricing.close,
              morningstar.share_class_reference.is_primary_share,#.latest,  
              morningstar.share_class_reference.is_depositary_receipt,  
              morningstar.asset_classification.morningstar_sector_code,
              morningstar.valuation.market_cap,
              morningstar.valuation.shares_outstanding,
              morningstar.balance_sheet.limited_partnership,
              #morningstar.company_reference.primary_exchange_id
              #morningstar.share_class_reference.security_type
              ]  
    
    def compute(self, today, assets, out,close, is_primary_share, is_depositary_receipt, sector_code, market_cap, shares_outstanding, limited_partnership):#,primary_exchange_id):#, secuirty_type):  
        criteria = close[-1] > 1.0
        criteria = criteria & is_primary_share[-1] # Only primary Common Stock  
        criteria = criteria & (~is_depositary_receipt[-1]) # No ADR  
        #criteria = criteria & (sector_code[-1] != 101) # No Basic Materials  
        #criteria = criteria & (sector_code[-1] != 103) # No Financials  
        #criteria = criteria & (sector_code[-1] != 104) # No Real Estate 
        criteria = criteria & (market_cap[-1] > 100000000)
        criteria = criteria #& (~limited_partnership[-1]) # not an LP
        #criteria = criteria & (primary_exchange_id != "OTCPK" & primary_exchange_id != "OTCBB")
        
        def accept_symbol(equity):  
            symbol = equity.symbol  
            #if symbol.endswith("_PR") or symbol.endswith("_WI") or symbol.endswith("_WD") or                                symbol.endswith("_VJ") or symbol.endswith("_V") or symbol.endswith("_H"):  
            if symbol.endswith("_WI"):
                return False  
            else:  
                return True  
        def accept_exchange(equity):  
            exchange = equity.exchange  
            if exchange == "NEW YORK STOCK EXCHANGE":  
                return True  
            elif exchange == "AMERICAN STOCK EXCHANGE":  
                return True  
            elif exchange.startswith("NASDAQ"):  
                return True  
            else:  
                return False  
        vsid = np.vectorize(sid)  
        equities = vsid(assets)  
        # Exclude When Distributed(WD), When Issued(WI) and VJ (bankruptcy) and Halted stocks (V, H)  
        vaccept_symbol = np.vectorize(accept_symbol)  
        accept_symbol = vaccept_symbol(equities)  
        criteria = criteria & (accept_symbol)  
        # Only NYSE, AMEX and Nasdaq  
        vaccept_exchange = np.vectorize(accept_exchange)  
        accept_exchange = vaccept_exchange(equities)  
        criteria = criteria & (accept_exchange)  
        out[:] = criteria.astype(float)  
        

def signalize(df):
   return ((df.rank() - 0.5)/df.count()).replace(np.nan,0.5)
    

def _signalize(a):
    x = (st.rankdata(a) - 0.5)/len(a)
    x[x==np.isnan] = 0.5
    return x
    
    
def _beta(ts, benchmark, benchmark_var):
    return np.cov(ts, benchmark)[0, 1] / benchmark_var 

class Beta(CustomFactor):
    
    inputs = [USEquityPricing.close]
    window_length = 60
    
    def compute(self, today, assets, out, close):
        returns = pd.DataFrame(close, columns=assets).pct_change()[1:]
        spy_returns = returns[sid(8554)]
        spy_returns_var = np.var(spy_returns)
        out[:] = returns.apply(_beta, args=(spy_returns, spy_returns_var))

class CashReturn(CustomFactor):   
    
    # Pre-declare inputs and window_length
    inputs = [morningstar.valuation_ratios.cash_return] 
    window_length = 1
    
    # Compute factor1 value
    def compute(self, today, assets, out, cash_return):       
        out[:] = cash_return[-1]
        
        
class MarketCap_O(CustomFactor):   
    
    # Pre-declare inputs and window_length
    inputs = [morningstar.valuation.market_cap] 
    window_length = 1
    
    # Compute factor2 value
    def compute(self, today, assets, out, market_cap):       
        out[:] = market_cap[-1]
        
        
class MarketCap(CustomFactor):   
    
    # Pre-declare inputs and window_length
    inputs = [USEquityPricing.close, morningstar.valuation.shares_outstanding] 
    window_length = 1
    
    # Compute market cap value
    def compute(self, today, assets, out, close, shares):       
        out[:] = close[-1] * shares[-1]
        
class BuybackYield(CustomFactor):
    # Pre-declare inputs and window_length
    inputs = [morningstar.valuation_ratios.buy_back_yield] 
    window_length = 1
    
    # Compute market cap value
    def compute(self, today, assets, out, buy_back_yield):       
        out[:] = buy_back_yield[-1]
        
        
class Ebit_EV(CustomFactor):
     # Pre-declare inputs and window_length
    inputs = [morningstar.income_statement.ebit, morningstar.valuation.enterprise_value] 
    window_length = 1
    
    # Compute 
    def compute(self, today, assets, out, ebit, enterprise_value):       
        ev = enterprise_value[-1].copy()
        ev[ev < 0.0] = 1.0
        out[:] = ebit[-1]/(ev)
        
class Ebitda_EV(CustomFactor):
     # Pre-declare inputs and window_length
    inputs = [morningstar.income_statement.ebitda, morningstar.valuation.enterprise_value] 
    window_length = 1
    
    # Compute 
    def compute(self, today, assets, out, ebitda, enterprise_value):       
        ev = enterprise_value[-1].copy()
        ev[ev < 0.0] = 1.0
        out[:] = ebitda[-1]/(ev)        
       
class Momentum(CustomFactor):
    # Pre-declare inputs and window_length
    inputs = [USEquityPricing.close] 
    window_length = 252
    
    def compute(self, today, assets, out, close): 
        out[:] = close.mean()

class Returns(CustomFactor):
    inputs = [USEquityPricing.close]
    window_length = 300

    def compute(self, today, assets, out, prices):
        # Getting the range of indexes so we can reindex later
        index = range(0, len(prices[1]))

        # Calculated a shifted return
        prices = pd.DataFrame(np.log(prices)).dropna(axis=1)
        #prices = prices[:-22]
        #monthly = prices[::-22]
        
        R = (prices/prices.shift(50))
        R = R[np.isfinite(R[R.columns])].fillna(0)
        #R=R.std()
       # R = R.tail(1).reindex(index, fill_value=np.nan)
        #R = R[:-1]#.mean()*100
        #R = R - np.maximum(0.0,(R-5))**2 + np.minimum(0.0,(R+5))**2 
        #R = (prices/prices.shift(50))
        #R = R[np.isfinite(R[R.columns])].fillna(0).mean()

        # Subtracts the cross-sectional average out of each data point on each day. 
        R = (R.T - R.T.mean()).T.mean()
        
        # Fill in nan values so we can drop them later
        #ranks = ranks.reindex(index, fill_value=np.nan)
        #out[:] = np.array(ranks)
        out[:] = np.array(R.reindex(index, fill_value=np.nan))
        
class VolumeTurnover(CustomFactor):
    inputs = [USEquityPricing.volume]
    window_length = 70
    
    def compute(self, today, assets, out, volume):
        # Getting the range of indexes so we can reindex later
        index = range(0, len(volume[1]))

        # Calculated a shifted return
        V = pd.DataFrame(volume)

        V1 = V.shift(1)
        V = (V - V1)/(V + V1)
        V = V[np.isfinite(V[V.columns])].fillna(0)
        
        # Subtracts the cross-sectional average out of each data point on each day. 
        out[:] = (V.T - V.T.mean()).T.mean()

class NetIncome(CustomFactor):
    inputs = [morningstar.cash_flow_statement.net_income]
    window_length = 1
    
    def compute(self, today, assets, out, net_income):
        out[:] = net_income[-1]
        
class Roic(CustomFactor):
    inputs = [morningstar.operation_ratios.roic, morningstar.cash_flow_statement.net_income, morningstar.cash_flow_statement.free_cash_flow]
    window_length = 1
    def compute(self, today, assets, out, roic, net_income, free_cash_flow):
        out[:] = (roic[-1]*free_cash_flow[-1])/net_income[-1]

        
class AssetTurnover(CustomFactor):
    inputs = [morningstar.operation_ratios.assets_turnover]
    window_length = 1
    def compute(self, today, assets, out, assets_turnover):
        out[:] = assets_turnover[-1]
        
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
    context.long_exposure_min = 0.95
    context.short_exposure_min = -0.95
    
    context.beta_low = -1.0
    context.beta_high = 1.0
    context.maxleverage = 2.0
    context.maxposition = 0.04
    context.bnchmarkHedge = False
    #context.rank = []
    
    universe = UniverseFilter()
    
    market_cap = MarketCap()
    
    #dollar_volume = AverageDollarVolume(window_length=70)
    
    # with universe flag and top 2500 names by market cap
    universefilter = (universe*market_cap).top(2500)
    
    pipe = Pipeline(screen = universefilter)
    
    attach_pipeline(pipe, 'ranked')
       
    # Add the two factors defined to the pipeline
 
    pipe.add(market_cap,'mcap')
    
    ebit_ev = Ebit_EV()
    pipe.add(ebit_ev, 'ebit_ev')
    
    #ebitda_ev = Ebitda_EV()
    #pipe.add(ebitda_ev, 'ebitda_ev')
        
    bbyield = BuybackYield()
    pipe.add(bbyield, 'bbyield')
    
    cashreturn = CashReturn()
    pipe.add(cashreturn, 'cashreturn')
        
    momentum = Returns() #SimpleMovingAverage([USEquityPricing.close], window_length=252)
    pipe.add(momentum,'momentum')
    
    #assetTurnover = AssetTurnover()
    #pipe.add(assetTurnover,'assetTurnover')
    
    #roic = Roic()
    #pipe.add(roic,'roic')
    #volumeturnover = VolumeTurnover()
    #pipe.add(volumeturnover, 'vturnover')
     
    #net_income = NetIncome()
    #pipe.add(net_income/dollar_volume,'qi')
    #beta = Beta()
    #pipe.add(beta,'beta')
   
    context.benchmarkSecurity = sid(21519)
    
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
  
       
def transform(context):
    df = context.output
    df['ebit_ev_rank'] = signalize(signalize(df['ebit_ev']*df['mcap']))
    df['bbyield_rank'] = signalize(df['bbyield'])
    df['creturn_rank'] = signalize(df['cashreturn'])
    df['momentum_rank'] = signalize(df['momentum'])
    #df['qi_rank'] = signalize(df['qi'])
    #df['vturnover_rank'] = signalize(-df['vturnover'])
    #df['roic_rank'] = signalize(df['roic'])
    #df['assetTurnover_rank'] = signalize(df['assetTurnover'])
    df['combo_rank'] = signalize(
                                df['ebit_ev_rank']  
                               + df['bbyield_rank']  
                               + df['creturn_rank'] 
                               + 0.1*df['momentum_rank'])
                               #+ df['assetTurnover_rank']) 
                               #+ df['roic_rank'])
    minimum = -10
    maximum = 10
    df['forecast'] = df['combo_rank'].apply(lambda x: x*(maximum - minimum) + minimum )
            
    return df

def before_trading_start(context, data):
    # Call pipelive_output to get the output
    context.output = pipeline_output('ranked')
    

    context.output = transform(context)
    
    #log.info(len(context.output))
    # Narrow down the securities to only the top 200 & update my universe
    #context.longs = context.output.sort(['combo_rank'], ascending=False)[['combo_rank','ebit_ev_rank', 'bbyield_rank', 'bbyield_rank', 'creturn_rank', 'momentum_rank']].iloc[:100]
    #context.shorts = context.output.sort(['combo_rank'], ascending=True)[['combo_rank','ebit_ev_rank', 'bbyield_rank', 'creturn_rank', 'momentum_rank']].iloc[:100]   
    context.longs = context.output.sort(['combo_rank'], ascending=False).iloc[:200]
    context.shorts = context.output.sort(['combo_rank'], ascending=True).iloc[:200]   
    
    delisted = []
    
    for stock in context.longs.index:
        if get_datetime() > stock.end_date + datetime.timedelta(days=3):
            delisted.append(stock)
    context.longs.drop(delisted)    
    
    delisted = []
    for stock in context.shorts.index:
        if get_datetime() > stock.end_date + datetime.timedelta(days=3):
            delisted.append(stock)
            
    context.shorts.drop(delisted)
    
    context.forecast = pd.concat([context.longs, context.shorts], axis=0)
    context.forecast.name = 'forecast'
  
    update_universe(context.forecast.index.values)
    context.universe = [stock for stock in context.forecast.index.values]

def handle_data(context, data):  
    
     # Record and plot the leverage of our portfolio over time. 
    record(leverage = context.account.leverage)
    log.info((sum(context.longs['mcap']) + sum(context.shorts['mcap']))*0.000000001)
    #print "Long List"
    #log.info("\n" + str(context.longs.sort(['combo_rank'], ascending=False).head(10)))
    
    #print "Short List" 
    #log.info("\n" + str(context.shorts.sort(['combo_rank'], ascending=True).head(10)))

# Computes the weights for the portfolio with the smallest Mean Absolute Deviation  
def minimum_MAD_portfolio(context, opt_data, returns):  
                     
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
    limits_l = [0, context.maxposition]
    limits_s = [-context.maxposition, 0]
       
    for stock in opt_data.index:
        if opt_data['longd'][stock] == 1:         
            bnds.append(limits_l)
        else:
            bnds.append(limits_s)          
            
    bnds = tuple(tuple(x) for x in bnds)
    cons = ({'type':'ineq', 'fun': _leverage}
            #,{'type':'ineq', 'fun': _long_exposure}
            #,{'type':'ineq', 'fun': _short_exposure}
            #,{'type':'ineq', 'fun': _beta_exposure_l}
            #,{'type':'ineq', 'fun': _beta_exposure_g}
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
        if sid is not context.benchmarkSecurity:
            place_order(data, sid, 0)
   
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
            place_order(data, security, weights[security]) 
    
    if context.bnchmarkHedge:
        order_target_percent(context.benchmarkSecurity, -1.0)

def place_order(data, sid, percent):
    
    if isvalid(sid, data):
        order_target_percent(sid, percent)
        
        
def isvalid(sid, data):

    if sid in data:
        return data[sid].close_price > 1.5 and data[sid].returns() < 1 and data[sid].returns() > -1 and get_datetime() < sid.end_date + datetime.timedelta(days=3)
    else:
        return False        
        
def calculate_beta(context, data, prices):
    returns = prices.pct_change()
    returns_stocks = returns[context.universe]
    returns_spy = returns[context.benchmarkSecurity]
    
    beta_span = 200
    #benchVar = pd.stats.moments.ewmvar(returns_spy, span=beta_span)[beta_span:]
    #cov = pd.stats.moments.ewmcov(returns_stocks, returns_spy, span=beta_span)[beta_span:]
    #beta = cov.div(benchVar.ix[0]).iloc[-1]
    std = returns_stocks.std()
    beta = std / np.mean(std)
    beta.name = 'beta'
    return beta

def df_intersection_index(df1, df2):
    return df2.index.isin(df1.index) & df1.index.isin(df2.index)

def df_intersection_index_col(df1, df2):
    return df2.keys().intersection(df1.index)

def round_portfolio(portfol, context):   
    
    for security in portfol.keys(): 
        if abs(portfol[security]) < 0.0001 or sid is symbol('HVT_A') or sid is symbol('BH'):
            portfol[security] = 0
            
                
    return portfol#/sum(portfol)