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
from cvxopt import matrix, solvers, spdiag, spmatrix, sparse
import cvxpy as cp
from scipy import sparse as scipysp

def initialize(context):

    context.max_leverage = 1.075
    context.min_leverage = 0.925
    context.max_pos_size = 0.01
    context.max_turnover = 0.95
    context.max_beta = 0.15
    context.max_net_exposure = 0.075
    context.max_volatility = 0.05
    context.max_sector_exposure = 0.175
    context.max_style_exposure = 0.375
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

    beta = SimpleBeta(
        target=sid(8554),
        regression_length=260,
    )
    
    beta_pipe = Pipeline(
        columns={
            'beta': beta
        },
        screen = QTradableStocksUS() & beta.notnull(),
    )
    algo.attach_pipeline(beta_pipe, 'beta_pipe')

        
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
        'f1': f1,
    }

    return Pipeline(columns=pipe_columns, screen=pipe_screen)

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

def getDuplicateColumns(df):
    '''
    Get a list of duplicate columns.
    It will iterate over all the columns in dataframe and find the columns         whose contents are duplicate.
    :param df: Dataframe object
    :return: List of columns whose contents are duplicates.
    '''
    duplicateColumnNames = set()
    # Iterate over all the columns in dataframe
    for x in range(df.shape[1]):
        # Select column at xth index.
        col = df.iloc[:, x]
        # Iterate over all the columns in DataFrame from (x+1)th index till end
        for y in range(x + 1, df.shape[1]):
            # Select column at yth index.
            otherCol = df.iloc[:, y]
            # Check if two columns at x 7 y index are equal
            if col.equals(otherCol):
                duplicateColumnNames.add(df.columns.values[y])
                
    return list(duplicateColumnNames)    

def identity_mat(size):
    return scipysp.csc_matrix(np.identity(size))

def optimize(context, alpha, factor_loadings, beta):
            
    nstocks = alpha.size
    
    initial_portfolio = context.initial_portfolio.reshape((nstocks,))
    min_turnover = context.min_turnover
    max_turnover = context.max_turnover - min_turnover
    
    # w = wl + ws  
    # Number of variables =  
    #    2*nStocks(weights) +  
    #    nStocks(turnover link vars) 
    
    #Number of Inequality group restrictions =  
    #    nSector +  
    #    nStyles  +  
    #    1 (net exposure restrictions)  
    #    1 (gross exposure restriction) +  
    #    1 (turnover restriction)  
    #    2*nStocks(turnover link vars) +  
    #    1 (market beta exposure) +  

    zeros_array = np.zeros(nstocks)
    I = np.identity(nstocks)
    
    ONE_ROW_MAT = np.asmatrix(np.full((1, nstocks), 1.0))
    ZERO_ROW_MAT = np.asmatrix(np.full((1, nstocks), 0.0))
    
    # Group Constraints - 1
    # min_exposure < Risk Loading transpose * W < ma_exposure
    sector_exp_matrix = None
    nsectors = len(context.sectors)
    sector_exp_bounds = np.full((nsectors), context.max_sector_exposure)
    
    
    for col_name in context.sectors:
        _loadings = np.matrix(np.hstack((
                factor_loadings[col_name].values,
                factor_loadings[col_name].values,
                zeros_array)).reshape((1, 3*nstocks)))
        
        if sector_exp_matrix is not None:
            sector_exp_matrix = np.concatenate((sector_exp_matrix, _loadings))
            
        else:
            sector_exp_matrix = _loadings
    
    sector_exp_matrix = scipysp.csc_matrix(sector_exp_matrix)
    
    style_exp_matrix = None
    nstyles = len(context.styles)
    style_exp_bounds = np.full((nstyles), context.max_style_exposure)

    for col_name in context.styles:
       
        _loadings = np.matrix(np.hstack((
                factor_loadings[col_name].values,
                factor_loadings[col_name].values,
                zeros_array)).reshape((1, 3*nstocks)))
        
        if style_exp_matrix is not None:
            style_exp_matrix = np.concatenate((style_exp_matrix, _loadings))
        else:
            style_exp_matrix = _loadings
    
    style_exp_matrix = scipysp.csc_matrix(style_exp_matrix)
    
    # Group Constraints - 2
    # Dollar Neutral or min_net_exp < sum of weights < max_net_exp
    net_exp_matrix = scipysp.csc_matrix(
        np.concatenate((ONE_ROW_MAT, ONE_ROW_MAT, ZERO_ROW_MAT), axis=1))
    
    net_exp_bounds = np.full((1), context.max_net_exposure)
    
    # Group Constraints - 3
    # Gross Exposure or min_gross_exp < sum of abs weights < max_gross_exp
    # aw = abs(w)
    # w - aw <= 0 ; aw > 0
    # min_gross_exp < aw1 + aw2 + ..... + awn < max_gross_exp
    gross_exp_matrix = scipysp.csc_matrix(
        np.concatenate((ONE_ROW_MAT, -ONE_ROW_MAT, ZERO_ROW_MAT), axis=1))
    
    gross_exp_bounds_u = np.full((1), context.max_leverage)
    gross_exp_bounds_l = np.full((1), context.min_leverage)
    
    # Group Constraints - 4
    # Turnover < max_turnover
    # Sum of abs(w-w0) < ma_turnover
    # t = abs(w-w0) #new variable      
    turnover_matrix = scipysp.csc_matrix(
        np.concatenate((ZERO_ROW_MAT, ZERO_ROW_MAT, ONE_ROW_MAT), axis=1))
    turnover_bound = np.full((1), max_turnover)
    
    # t = |wl + ws - w0|
    # t > wl + ws - w0  =>  -np.inf < wl + ws - t < w0
    # t > -wl -ws + w0 =>  w0 < wl + ws + t < np.inf
    # t >= 0
    # Group Constraints - 5 (turnover link vars)
    turnover_lnk_matrix_1 = scipysp.csc_matrix(
        np.concatenate((I, I, -1*I), axis=1))
    
    turnover_lnk_matrix_2 = scipysp.csc_matrix(
        np.concatenate((I, I, I), axis=1))
    
    turnover_lnk_bounds_1l = -np.full((nstocks), np.inf)   
    turnover_lnk_bounds_1u = initial_portfolio
   
    turnover_lnk_bounds_2l = initial_portfolio
    turnover_lnk_bounds_2u = np.full((nstocks), np.inf)

    # Group constraints - 6
    # Market beta exposure
    # lBeta < (B1*W1 + B2*W2 + ... + BnWn) < uBeta
    market_beta_exp_matrix = scipysp.csc_matrix(np.matrix(np.concatenate((
            np.reshape(beta.values, (nstocks,)),
            np.reshape(beta.values, (nstocks,)),
            zeros_array)).reshape((1, 3*nstocks))))
    
    market_beta_exp_bound = np.full((1), context.max_beta)

    # Create optimization variables
    wL = cp.Variable(nstocks)
    wS = cp.Variable(nstocks)
    tr = cp.Variable(nstocks)

    # Combine all Group Restrictons
    A = cp.vstack(
            sector_exp_matrix, 
            style_exp_matrix,
            net_exp_matrix,
            gross_exp_matrix,
            turnover_matrix,
            # turnover_lnk_matrix_1,
            # turnover_lnk_matrix_2,
            market_beta_exp_matrix,
        )
    
    # Group Restrictions Upper Bound
    Ub = np.hstack((
            sector_exp_bounds,
            style_exp_bounds,
            net_exp_bounds,
            gross_exp_bounds_u,
            turnover_bound,
            # turnover_lnk_bounds_1u,
            # turnover_lnk_bounds_2u,
            market_beta_exp_bound
        ))
    
    # Group Restrictions Lower Bound
    Lb = np.hstack((
            -1*sector_exp_bounds,
            -1*style_exp_bounds,
            -1*net_exp_bounds,
            gross_exp_bounds_l,
            np.full((1,), 0.0),
            # turnover_lnk_bounds_1l,
            # turnover_lnk_bounds_2l,
            -1*market_beta_exp_bound
        ))
    
    # Stack Variables
    x = cp.vstack(wL,wS,tr)
    
    # Optimization Problem Constraints (Group + Variable)
    constraints = [
        A*x <= Ub,
        A*x >= Lb,
        wL >= 0,
        wL <= context.max_pos_size,
        wS <= 0,
        wS >= -context.max_pos_size,
        tr >= 0]
        
    #Objective Function - Maximize Alpha
    c = np.hstack((
        alpha.values,
        alpha.values,
        zeros_array))
    
    objective = cp.Maximize(c.T*x)
    prob = cp.Problem(objective, constraints)
    prob.solve()
    
    # print("\nThe optimal value is", prob.value)
    print("A solution x is")
    x = np.asarray(x.value).flatten().reshape((3*nstocks,))
    w = x[0:nstocks] + x[nstocks:2*nstocks]
    tr = x[2*nstocks:3*nstocks]
    actualtr = np.abs(w - initial_portfolio)
    print w[6:10]
    print actualtr[6:10]
    print tr[6:10]


def rebalance(context, data):
    
    alpha = context.alpha
    beta = context.beta_pipeline[['beta']]
    factor_loadings = context.risk_factor_betas
     
    validSecurities = list(
        set(alpha.index.values.tolist()) & 
        set(beta.index.values.tolist()) &
        set(factor_loadings.index.values.tolist()))
    
    alpha = alpha.loc[validSecurities]
    factor_loadings = factor_loadings.loc[validSecurities, :]
    beta = beta.loc[validSecurities, :]

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
        optimize(context, alpha, factor_loadings, beta)
        
        # Create MaximizeAlpha objective
        objective = opt.MaximizeAlpha(alpha)

        constrain_pos_size = opt.PositionConcentration(
            min_weights,
            max_weights
        )

        # Constrain target portfolio's leverage
        max_leverage = opt.MaxGrossExposure(context.max_leverage)

        # Ensure long and short books
        # are roughly the same size
        dollar_neutral = opt.DollarNeutral(context.max_net_exposure)
        
        # Constrain portfolio turnover
        max_turnover = opt.MaxTurnover(context.max_turnover)
        
        factor_risk_constraints = opt.experimental.RiskModelExposure(
            context.risk_factor_betas,
            # max_volatility = context.max_volatility,
            version=opt.Newest
        )
        
        beta_neutral = opt.FactorExposure(
            beta,
            min_exposures={'beta': -context.max_beta},
            max_exposures={'beta': context.max_beta},
        )

        algo.order_optimal_portfolio(
            objective = objective,
            constraints = [
                constrain_pos_size,
                max_leverage,
                dollar_neutral,
                max_turnover,
                factor_risk_constraints,
                beta_neutral
            ]
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