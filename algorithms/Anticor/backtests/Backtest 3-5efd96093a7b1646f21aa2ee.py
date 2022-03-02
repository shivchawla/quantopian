import numpy as np
import math
import pandas as pd
import quantopian.optimize as opt
import quantopian.algorithm as algo
from quantopian.pipeline.experimental import risk_loading_pipeline
from quantopian.pipeline import CustomFactor, Pipeline
from quantopian.pipeline.data import Fundamentals 
from quantopian.pipeline.data.builtin import USEquityPricing
from quantopian.pipeline.filters import QTradableStocksUS
from quantopian.pipeline.factors import SimpleBeta, AverageDollarVolume
from quantopian.pipeline.data import morningstar
from quantopian.pipeline.factors.morningstar import MarketCap

import cvxpy as cp
from scipy import sparse as scipysp
from scipy import stats, linalg

def initialize(context):

    context.max_leverage = 1.075
    context.min_leverage = 0.925
    context.max_pos_size = 0.01
    context.max_turnover = 0.95
    context.max_beta = 0.15
    context.max_net_exposure = 0.075
    context.max_volatility = 0.05
    context.max_sector_exposure = 0.18
    context.max_style_exposure = 0.375
    context.target_mkt_beta = 0
    context.beta_calc_days = 126
    context.normalizing_constant = 0.0714 
    context.window = 22
    set_slippage(slippage.FixedSlippage(spread=0.0))
    set_commission(commission.PerShare(cost=0, min_trade_cost=0))

    
    # context.etf_symbols = ['XLB','XLY','XLF','XLP','XLV','XLU','XLE','XLI','XLK']
    
    # context.etfs = [sid(19654),sid(19662),sid(19656), sid(19659),sid(19661), sid(19660), sid(19655), sid(19657), sid(19658)]
    
    # context.names = []
    # array = context.etfs.copy()
    
    # for s1 in context.etfs:
    #     if array is not None:
    #         array.remove(s1)
    #         for s2 in array:
    #             if s1 != s2:
    #                 n = (s1.symbol+'_'+s2.symbol)
    #                 context.names.append(n)
    
         
    # combs = len(context.names)            
    # context.weights = np.ones(combs)/combs
    context.weights = None
    
    context.sectors = [
        'basic_materials', 
        'consumer_cyclical', 
        'financial_services',
        'real_estate',
        'consumer_defensive',
        'health_care',
        'utilities',
        'communication_services',
        'energy',
        'industrials',
        'technology']
    
    context.styles = [
        'momentum', 
        'size', 
        'value', 
        'short_term_reversal', 
        'volatility']

    # Rebalance every day, 1 hour after market open.
    algo.schedule_function(
        rebalance,
        algo.date_rules.every_day(),
        algo.time_rules.market_open(hours=1),
    )

    algo.attach_pipeline(top_50_pipeline(), 'top_50')
    # Create our dynamic stock selector.
    # algo.attach_pipeline(make_pipeline(), 'pipeline')

    # algo.attach_pipeline(
    #     risk_loading_pipeline(),
    #     'risk_pipe'
    # )
    
    # algo.attach_pipeline(volatility_pipeline(), 'volatility_pipe');

    mkt_beta = SimpleBeta(
        target=sid(8554),
        regression_length=context.beta_calc_days,
    )
    
    value_beta = SimpleBeta(
        target=sid(22010),
        regression_length=context.beta_calc_days,
    )
    
    growth_beta = SimpleBeta(
        target=sid(22009),
        regression_length=context.beta_calc_days,
    )
    
    large_beta = SimpleBeta(
        target=sid(22148),
        regression_length=context.beta_calc_days,
    )
    
    small_beta = SimpleBeta(
        target=sid(21508),
        regression_length=context.beta_calc_days,
    )
    
    beta_pipe = Pipeline(
        columns={
            'mkt_beta': mkt_beta,
            'value_beta': value_beta,
            'growth_beta': growth_beta,
            'small_beta': small_beta,
            'large_beta': large_beta,
        },
        screen = QTradableStocksUS() & 
        mkt_beta.notnull() & 
        value_beta.notnull() & 
        growth_beta.notnull() & 
        small_beta.notnull() & 
        large_beta.notnull()
    )
    
    # algo.attach_pipeline(beta_pipe, 'beta_pipe')


class Volatility(CustomFactor):  
    inputs = [USEquityPricing.close]  
    window_length = 20  
    def compute(self, today, assets, out, close):  
        daily_returns = np.diff(close, axis = 0) / close[0:-1]  
        out[:] = daily_returns.std(axis = 0) * math.sqrt(252)
        
class AlphaFactor(CustomFactor):
    # Pre-declare inputs and window_length
    inputs = [USEquityPricing.close, Fundamentals.shares_outstanding]
    window_length = 1

    # Compute market cap value
    def compute(self, today, assets, out, close, shares):
        out[:] = np.log(close[-1] * shares[-1])
                
  
def top_50_pipeline():
    top_50 = (morningstar.valuation.market_cap.latest.top(50) & AverageDollarVolume(window_length=22).top(50))
    
    return Pipeline(
        columns={
            "market_cap": MarketCap()
        },
        screen=top_50
    )


def make_pipeline():
   
    # Base universe set to the QTradableStocksUS
    base_universe = QTradableStocksUS()

    f1 = AlphaFactor(mask = base_universe).zscore()
 
    # Filter stocks out with Mcap < mcap of 100th stock in S&P500
    pipe_screen = base_universe

    pipe_columns = {
        'f1': f1
    }

    return Pipeline(columns=pipe_columns, screen=pipe_screen)

def volatility_pipeline():
    
    volatility = Volatility(mask = QTradableStocksUS())   
  
    pipe = Pipeline(columns = {'volatility': volatility},
    screen = QTradableStocksUS())

    return pipe


def before_trading_start(context, data):
    """
    Called every day before market open.
    """
    if context.weights is None:
   
        output = algo.pipeline_output('top_50')
    
        context.stocks = output.index.tolist()
        context.stock_tickers = [sid.symbol for sid in context.stocks]
        print(context.stocks)
    
        context.names = []
        array = context.stocks.copy()
    
        for s1 in context.stocks:
            if array is not None:
                array.remove(s1)
                for s2 in array:
                    if s1 != s2:
                        n = (s1.symbol+'*'+s2.symbol)
                        context.names.append(n)
    
        combs = len(context.names)            
        context.weights = np.ones(combs)/combs

        
    # context.alpha = output['f1']
    
    # output['min_weights'] = -context.max_pos_size
    # output['max_weights'] = context.max_pos_size
    
    # context.min_weights = output['min_weights']
    # context.max_weights = output['max_weights']
    
    # context.risk_factor_betas = algo.pipeline_output(
    #   'risk_pipe'
    # ).dropna()
    
    # context.beta_pipeline = algo.pipeline_output('beta_pipe')
    # context.volatility = algo.pipeline_output('volatility_pipe')['volatility']
    

def anticor(context, data, b):
              
    w = context.window;
    
    price_history = data.history(context.stocks, fields="price", bar_count=2*w+1, frequency="1d")
    
    pct_change = price_history.pct_change().iloc[1:]
       
    pct_change.columns = [sid.symbol for sid in pct_change.columns]    
    df = pd.DataFrame()
        
    for n in context.names:
        sids = n.split('*')
        s1 = sids[0]
        s2 = sids[1]
        df[n] = pct_change[s1] - pct_change[s2]

    # now that we differnce return array, run actual olmar
   
    n = len(df.index)
    LX1 = df.head(w)
    LX2 = df.tail(n-w)
    
    
    mu1 = np.average(LX1, axis=0)
    sig1 = np.std(LX1, axis=0, ddof=1)
    mu2 = np.average(LX2, axis=0)
    sig2 = np.std(LX2, axis=0, ddof=1)
    sigma = np.outer(np.transpose(sig1),sig2)
       
    mu_matrix = np.ones((mu2.shape[0],mu2.shape[0]), dtype = bool)
    for i in range(0, mu2.shape[0]):
        for j in range(0, mu2.shape[0]):
            if mu2[i] > mu2[j]:
                mu_matrix[i,j] = True
            else:
                mu_matrix[i,j] = False           
    
    # print (mu_matrix)
    #Covariance matrix is dot product of x - mu of window 1 and window 2 (mxm)
            
    mCov = (1.0/np.float64(w-1)) * np.dot(np.transpose(np.subtract(LX1,mu1)),np.subtract(LX2,mu2)) 
    
    # print(mCov)
    #Correlation matrix is mCov divided element wise by sigma (mxm), 0 if sig1, sig2 = 0

    mCorr = np.where(sigma != 0, np.divide(mCov,sigma), 0)
    
    # print(mCorr)
    #Multiply the correlation matrix by the boolean matrix comparing mu2[i] to mu2[j] and by the boolean matrix where correlation matrix is greater than zero
            
    claim = np.multiply(mCorr,np.multiply(mCorr > 0,mu_matrix))             

    #The boolean claim matrix will be used to obtain only the entries that meet the criteria that mu2[i] > mu2[j] and mCorr is > 0 for the i_corr and j_corr matrices
            
    bool_claim = claim > 0
            
    #If stock i is negatively correlated with itself we want to add that correlation to all instances of i.  To do this, we multiply a matrix of ones by the diagonal of the correlation matrix row wise.
            
    i_corr = np.multiply(np.ones((mu1.shape[0],mu2.shape[0])),np.diagonal(mCorr)[:,np.newaxis])  
            
    #Since our condition is when the correlation is negative, we zero out any positive values, also we want to multiply by the bool_claim matrix to obtain valid entries only
            
    i_corr = np.where(i_corr > 0,0,i_corr)
    i_corr = np.multiply(i_corr,bool_claim)
            
    #Subtracting out these negative correlations is essentially the same as adding them to the claims matrix
            
    claim -= i_corr
            
    #We repeat the same process for stock j except this time we will multiply the diagonal of the correlation matrix column wise
            
    j_corr = np.multiply(np.ones((mu1.shape[0],mu2.shape[0])),np.diagonal(mCorr)[np.newaxis,:]) 

#Since our condition is when the correlation is negative, we zero out any positive values again multiplying by the bool_claim matrix
           
    j_corr = np.where(j_corr > 0,0,j_corr)
    j_corr = np.multiply(j_corr,bool_claim)            

#Once again subtract these to obtain our final claims matrix            

    claim -= j_corr   
                                
#Create the wealth transfer matrix first by summing the claims matrix along the rows
                  
    sum_claim = np.sum(claim, axis=1)
            
    #Then divide each element of the claims matrix by the sum of it's corresponding row
            
    transfer = np.divide(claim,sum_claim[:,np.newaxis])
            
    #Multiply the original weights to get the transfer matrix row wise

    transfer = np.multiply(transfer,b[:,np.newaxis])
            
    #Replace the nan with zeros in case the divide encountered any
            
    transfer = np.where(np.isnan(transfer),0,transfer)                        
    #We don't transfer any stock to itself, so we zero out the diagonals     
      
    np.fill_diagonal(transfer,0)
                        
    #Create the new portfolio weight adjustments, by adding the j direction weights or the sum by columns and subtracting the i direction weights or the sum by rows
                        
    adjustment = np.subtract(np.sum(transfer, axis=0),np.sum(transfer,axis=1))
    
    b += adjustment
    
    print(b)
    print(adjustment)
    
    return b

def rebalance(context, data):
    
    context.weights = anticor(context, data, context.weights)
    
    wt = context.weights
    wtdict = {} 
    
    for (i, name) in enumerate(context.names):
        sids = name.split('*')
        # print(sids)
        sid1 = sids[0]
        sid2 = sids[1]
        
        print(sid1)
        print(sid2)
        
        if sid1 in wtdict:
            wtdict[sid1] += wt[i]
        else: 
            wtdict[sid1] = wt[i]
         
        if sid2 in wtdict:
            wtdict[sid2] -= wt[i]
        else: 
            wtdict[sid2] = -wt[i]
    
        
    print("Weights")
    print(wtdict)
    
    for (k,v) in wtdict.items():
        try:
            idx = context.stock_tickers.index(k)
            order_target_percent(context.stocks[idx], -v)
        except:
            print("Security not found")
            
    
           
def record_vars(context, data):
    """
    Plot variables at the end of each day.
    """
    pass


def handle_data(context, data):
    """
    Called every minute.
    """
    pass