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
from quantopian.pipeline.factors import SimpleBeta
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

    # Create our dynamic stock selector.
    algo.attach_pipeline(make_pipeline(), 'pipeline')

    algo.attach_pipeline(
        risk_loading_pipeline(),
        'risk_pipe'
    )
    
    algo.attach_pipeline(volatility_pipeline(), 'volatility_pipe');

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
    
    algo.attach_pipeline(beta_pipe, 'beta_pipe')


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
    output = algo.pipeline_output('pipeline')
        
    context.alpha = output['f1']
    
    output['min_weights'] = -context.max_pos_size
    output['max_weights'] = context.max_pos_size
    
    context.min_weights = output['min_weights']
    context.max_weights = output['max_weights']
    
    context.risk_factor_betas = algo.pipeline_output(
      'risk_pipe'
    ).dropna()
    
    context.beta_pipeline = algo.pipeline_output('beta_pipe')
    context.volatility = algo.pipeline_output('volatility_pipe')['volatility']
    

def compute_covariance(context, data, securities):
    
    #1. Get factor loadings
    #2. Get 63 days historical returns on stocks
    #3. Get Factor returns by multiplying the factor loadings with st-returns
    factor_loadings = context.risk_factor_betas
    factor_loadings = factor_loadings.loc[securities, :]
    factor_loadings_sector = factor_loadings.copy()        
    
    price_history = data.history(securities, fields="price", bar_count=64, frequency="1d")
    pct_change = price_history.pct_change()
    
    for factor_name in context.styles:
        factor_loadings = factor_loadings.sort_values(by = factor_name)
        factor_loadings.loc[-50:, factor_name] = 1.0
        factor_loadings.loc[:50, factor_name] = -1.0
    
    factor_loadings[np.abs(factor_loadings) != 1.0] = 0        
    
    for factor_name in context.sectors:
        factor_loadings_sector = factor_loadings_sector.sort_values(by = factor_name)
        factor_loadings_sector.loc[-50:, factor_name] = 1.0

    factor_loadings_sector[np.abs(factor_loadings_sector) != 1.0] = 0        
    
    factor_returns_style = pct_change.dot(factor_loadings)
    cov_style = factor_returns_style.cov(min_periods=63)
    
    factor_returns_sector = pct_change.dot(factor_loadings_sector)
    cov_sector = factor_returns_sector.cov(min_periods=63)

    factor_list = context.sectors + context.styles
    cov_style = cov_style[factor_list]
    cov_style = cov_style.reindex(factor_list)
   
    cov_sector = cov_sector[factor_list]
    cov_sector = cov_sector.reindex(factor_list)
   
    return (cov_style, cov_sector) 
    
    
def optimize(context, alpha, factor_loadings, beta, cov_style, cov_sector, var):
            
    nstocks = alpha.size
    
    initial_portfolio = context.initial_portfolio.reshape((nstocks,))
    min_turnover = context.min_turnover
    max_turnover = context.max_turnover - min_turnover
    
    # Number of variables =  
    #    nStocks(weights) +  
   
    # Number of Inequality group restrictions =  
    #    nSector +  
    #    nStyles  +  
    #    1 (net exposure restrictions)  
    #    1 (gross exposure restriction) +  
    #    1 (turnover restriction)  
    #    1 (market beta exposure) 

    # Group Constraints - 1
    # min_exposure < Risk Loading transpose * W < ma_exposure
    sector_exp_matrix = None
    nsectors = len(context.sectors)
    sector_exp_bounds = np.full((nsectors), context.max_sector_exposure)
    
    for col_name in context.sectors:
        _loadings = scipysp.csc_matrix(np.matrix(
                factor_loadings[col_name].values.reshape((1, nstocks))))
        
        if sector_exp_matrix is not None:
            sector_exp_matrix = cp.vstack(sector_exp_matrix, _loadings)
            
        else:
            sector_exp_matrix = _loadings
    
    
    style_exp_matrix = None
    nstyles = len(context.styles)
    style_exp_bounds = np.full((nstyles), context.max_style_exposure)

    for col_name in context.styles:
        _loadings = scipysp.csc_matrix(np.matrix(
                factor_loadings[col_name].values.reshape((1, nstocks))))
        
        if style_exp_matrix is not None:
            style_exp_matrix = cp.vstack(style_exp_matrix, _loadings)
        else:
            style_exp_matrix = _loadings
    
    
    # Group constraints - 3
    # Market beta exposure
    # lBeta < (B1*W1 + B2*W2 + ... + BnWn) < uBeta
    
    #Market beta
    market_beta_exp_matrix = scipysp.csc_matrix(
                np.matrix(beta[['mkt_beta']].values.reshape((nstocks,))))
    
    #SML BEta
    sml_beta_exp_matrix = scipysp.csc_matrix(
                np.matrix(beta[['sml_beta']].values.reshape((nstocks,))))
    
    #VMG Beta
    vmg_beta_exp_matrix = scipysp.csc_matrix(
                np.matrix(beta[['vmg_beta']].values.reshape((nstocks,))))
    
    
    market_beta_exp_bound = np.full((1), context.max_beta)
   
    # Create optimization variables
    w = cp.Variable(nstocks)
    
    F = cp.vstack(
            sector_exp_matrix, 
            style_exp_matrix)
    
    f = F*w
    
    #Idiiosyncratic risk
    D = np.diag(var.values)
    chlskyD = linalg.cholesky(D, lower=True)
    fD = chlskyD*w

    # Combine all Group Restrictons
    A = cp.vstack(
            sector_exp_matrix, 
            style_exp_matrix,
            market_beta_exp_matrix
        )
    
    # Group Restrictions Upper Bound
    Ub = np.hstack((
            sector_exp_bounds,
            style_exp_bounds,
            market_beta_exp_bound
        ))
    
    # Group Restrictions Lower Bound
    Lb = np.hstack((
            -1*sector_exp_bounds,
            -1*style_exp_bounds,
            -1*market_beta_exp_bound
        ))
    
    # Optimization Problem Constraints (Group + Variable)
    constraints = [
        A*w <= Ub,
        A*w >= Lb,
        w <= context.max_pos_size,
        w >= -context.max_pos_size,
        cp.sum_entries(w) <= context.max_net_exposure,
        cp.sum_entries(w) >= -context.max_net_exposure,
        cp.norm(w,1) <= context.max_leverage,
        cp.norm(w-initial_portfolio,1) <= max_turnover
    ]
        
    #Objective Function - Maximize Alpha
    c = alpha.values
    
    total_alpha = c.T*w
    gamma_sys = cp.Parameter(sign = "positive")
    gamma_sys.value = 5.0 * context.normalizing_constant
    
    gamma_unsys = cp.Parameter(sign = "positive")
    gamma_unsys.value = 100.0 * context.normalizing_constant

    gamma_beta = cp.Parameter(sign = "positive")
    gamma_beta.value = 140.0 * context.normalizing_constant
    
    
    risk_sys_style = cp.quad_form(f, cov_style.as_matrix()) 
    risk_sys_sector = cp.quad_form(f, cov_sector.as_matrix()) 
    risk_unsys = cp.sum_squares(fD)
    
    mkt_beta_deviation = cp.sum_squares(market_beta_exp_matrix*w - context.target_mkt_beta)
    
    # sml_beta_deviation = cp.sum_squares(sml_beta_exp_matrix*w - 0.0)       
    # vmg_beta_deviation = cp.sum_squares(vmg_beta_exp_matrix*w - 0.0)
    
    beta_deviation = mkt_beta_deviation
    
    #Penalty on portfolio deviation from initial (used in Objective 4)
    port_deviation = cp.norm(w - initial_portfolio, 1)
    gamma_port_deviation = cp.Parameter(sign = "positive")
    gamma_port_deviation.value = 10.0 * context.normalizing_constant

    #Objective 1 - Maximize Alpha
    #objective = cp.Maximize(total_alpha)
    
    #Objective 2: Maximize alpha with beta penalty
    #objective = cp.Maximize(total_alpha - beta_deviation)
    
    #Objective 3: Maximize alpha With risk control and beta penalty 
    objective = cp.Maximize(total_alpha - gamma_sys*risk_sys_style - gamma_sys*risk_sys_sector - gamma_unsys*risk_unsys - gamma_beta*beta_deviation)
      
    #Objective 4: Maximize alpha With risk control, beta penalty and turnover penalty  
    # objective = cp.Maximize(total_alpha - gamma_sys*risk_sys_style - gamma_sys*risk_sys_sector - gamma_unsys*risk_unsys - gamma_beta*beta_deviation - gamma_port_deviation*port_deviation)
   
    
    prob = cp.Problem(objective, constraints)
    try:
        prob.solve()
        # print("\nThe optimal value is", prob.value)
        # print("A solution x is")
        w = np.asarray(w.value).flatten().reshape((nstocks,))
        w[abs(w) <= 0.0001] = 0
        
        # print w
        return w
    except Exception as e:
        print(e)
        return initial_portfolio
        

def rebalance(context, data):
    
    alpha = context.alpha
    beta = context.beta_pipeline
    factor_loadings = context.risk_factor_betas
    vol = context.volatility
     
    validSecurities = list(
        set(alpha.index.values.tolist()) & 
        set(beta.index.values.tolist()) &
        set(factor_loadings.index.values.tolist()))
    
    alpha = alpha.loc[validSecurities]
    factor_loadings = factor_loadings.loc[validSecurities, :]
    beta = beta.loc[validSecurities, :]   
    
    beta['sml_beta'] = beta['small_beta'] - beta['large_beta']
    beta['vmg_beta'] = beta['value_beta'] - beta['growth_beta']
   
    vol = vol.loc[validSecurities]
    variance = vol * vol
    
    (cov_style, cov_sector) = compute_covariance(context, data, validSecurities)

    min_weights = context.min_weights.copy(deep = True)
    max_weights = context.max_weights.copy(deep = True)
    
    context.min_turnover = 0.0

    int_port = np.zeros(len(alpha))
                        
    portfolio_value = context.portfolio.portfolio_value
    allPos = context.portfolio.positions
    currentSecurities = list(allPos.keys())
    defunctSecurities = list(set(currentSecurities) - set(validSecurities))               
    for (i,sec) in enumerate(validSecurities):
        if allPos[sec]:
           int_port[i] = allPos[sec].amount/portfolio_value
        
    for (i,sec) in enumerate(defunctSecurities):
        if allPos[sec]:
           context.min_turnover += allPos[sec].amount/portfolio_value
    
    context.initial_portfolio = int_port
                        
    if not alpha.empty:
        
        # Running optimization implementation (testing)         
        my_opt_weights = optimize(context, alpha, factor_loadings, beta, cov_style, cov_sector, variance)
        my_opt = pd.Series(my_opt_weights, index=validSecurities) 
        
        algo.order_optimal_portfolio(
            opt.TargetWeights(my_opt),
            constraints = []
        )
        

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