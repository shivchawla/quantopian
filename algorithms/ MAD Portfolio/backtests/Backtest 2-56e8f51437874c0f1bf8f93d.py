from scipy.optimize import minimize
import numpy as np
import pandas as pd

# Computes the weights for the portfolio with the smallest Mean Absolute Deviation  
def minimum_MAD_portfolio(returns):  

    def _sum_check(x):  
        return sum(abs(x)) - 1  

    # Computes the Mean Absolute Deviation for the current iteration of weights  
    def _mad(x, returns):  
        return (returns - returns.mean()).dot(x).abs().mean()

    num_assets = len(returns.columns)  
    guess = np.ones(num_assets)  
    cons = {'type':'eq', 'fun': _sum_check}  
    #min_mad_results = minimize(_mad, guess, args=returns[returns < 0], constraints=cons)  
    min_mad_results = minimize(_mad, guess, args=returns, constraints=cons)  
    return pd.Series(index=returns.columns, data=min_mad_results.x)  

# Put any initialization logic here.  The context object will be passed to
# the other methods in your algorithm.
def initialize(context):
    set_commission(commission.PerTrade(cost=0))
    set_slippage(slippage.FixedSlippage(spread=0))
    context.stocks = [
        sid(19662),  # XLY Consumer Discrectionary SPDR Fund   
        sid(19656),  # XLF Financial SPDR Fund  
        sid(19658),  # XLK Technology SPDR Fund  
        sid(19655),  # XLE Energy SPDR Fund  
        sid(19661),  # XLV Health Care SPRD Fund  
        sid(19657),  # XLI Industrial SPDR Fund  
        sid(19659),  # XLP Consumer Staples SPDR Fund   
        sid(19654),  # XLB Materials SPDR Fund  
        sid(19660),  # XLU Utilities SPDR Fund
    ]
    schedule_function(rebalance, date_rules.every_day())

# Will be called on every trade event for the securities you specify. 
def handle_data(context, data):
    record(lever=context.account.leverage, exposure=context.account.net_leverage)

def rebalance(context, data):
    
    returns = history(200, "1d", "price").pct_change().dropna()
    
    #return_rank = (returns.rank(axis=1) - 0.5)/len(returns.columns)
    #log.info(return_rank)
    
    weights = minimum_MAD_portfolio(returns)
    for security in returns:
        # print(weights[security])
        order_target_percent(security, weights[security])

