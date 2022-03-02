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

    
def optimize(context):
    alpha = context.alpha
    factor_loadings = context.risk_factor_betas.loc[alpha.index.tolist(), :]
    beta = context.beta_pipeline[['beta']].loc[alpha.index.tolist(), :]
    
    nstocks = alpha.size
    initial_portfolio = context.initial_portfolio
    min_turnover = context.min_turnover
    max_turnover = context.max_turnover - min_turnover
    
    # w = wl + ws  
    # Number of variables =  
    #    2*nStocks(weights) +  
    #    nStocks(turnover link vars) 
    
    #Number of Inequality group restrictions =  
    #    2*nSector +  
    #    2*nStyles  +  
    #    2*1 (net exposure restrictions)  
    #    2*1 (gross exposure restriction) +  
    #    1 (turnover restriction)  
    #    2*nStocks(turnover link vars) +  
    #    2*1 (market beta exposure) +  
    #    2*nStocks (Weight long bounds)  
    #    2*nStocks (Weight short bounds)  
    #    nStocks (turnover link vars lower bound)
     
    ones_matrix = matrix(1.0, (nstocks, 1), 'd')
    zeros_matrix = matrix(0.0, (nstocks, 1), 'd')
    
    # Group Constraints - 1/2
    # min_exposure < Risk Loading transpose * W < ma_exposure
    sector_exp_matrix = matrix()
    nsectors = len(context.sectors)
    sector_exp_bounds = matrix(context.max_sector_exposure, (nsectors,1), 'd')

    for col_name in context.sectors:
        _loadings = sparse([
                matrix(factor_loadings[col_name].values, (nstocks, 1), 'd'),
                matrix(factor_loadings[col_name].values, (nstocks, 1), 'd'),
                zeros_matrix]).T
        
        if sector_exp_matrix:
            sector_exp_matrix = sparse([sector_exp_matrix, _loadings])
        else:
            sector_exp_matrix = _loadings
    
    style_exp_matrix = matrix()
    nstyles = len(context.styles)
    style_exp_bounds = matrix(context.max_style_exposure, (nstyles,1), 'd')

    for col_name in context.styles:
        _loadings = sparse([
                matrix(factor_loadings[col_name].values, (nstocks, 1), 'd'),
                matrix(factor_loadings[col_name].values, (nstocks, 1), 'd'),
                zeros_matrix]).T
        
        if style_exp_matrix:
            style_exp_matrix = sparse([style_exp_matrix, _loadings])
        else:
            style_exp_matrix = _loadings
    
    # Group Constraints - 3/4
    # Dollar Neutral or min_net_exp < sum of weights < max_net_exp
    net_exp_matrix = sparse([
            ones_matrix,
            ones_matrix,
            zeros_matrix]).T
    
    net_exp_bounds = matrix(context.max_net_exposure, (1,1), 'd')
    
    # Group Constraints - 5/6
    # Gross Exposure or min_gross_exp < sum of abs weights < max_gross_exp
    # aw = abs(w)
    # w - aw <= 0 ; aw > 0
    # min_gross_exp < aw1 + aw2 + ..... + awn < max_gross_exp
    gross_exp_matrix = sparse([
            ones_matrix,
            -1*ones_matrix,
            zeros_matrix]).T
    
    gross_exp_bounds_u = matrix(context.max_leverage, (1,1), 'd')
    gross_exp_bounds_l = -1*matrix(context.min_leverage, (1,1), 'd')
    
    # Group Constraints - 8
    # Turnover < max_turnover
    # Sum of abs(w-w0) < ma_turnover
    # dw = abs(w-w0) #new variable      
    # dw >= w - wo 
    # dw >= 0 variable constraint (Number of variables = nstocks)
    # Total number of variables = 2*nstocks 
    # dw1 + dw2 + ... + dwn < max_turnover (Number of Group Restriction = 1)
    # Total Number of Group Restriction = nFactors + nStocks
    turnover_matrix = sparse([
            zeros_matrix,
            zeros_matrix,
            ones_matrix]).T
    
    turnover_bound = matrix(max_turnover, (1,1), 'd')
    turnover_lnk_bounds = matrix(initial_portfolio, (nstocks,1), 'd') 
    
    # d = |wl + ws - w0|
    # d > wl + ws - w0  => wl + ws - d < w0
    # d > -wl -ws + w0 => -wl -ws -d < -w0
    # Group Constraints - 9 (turnover link vars)
    turnover_lnk_matrix_1 = matrix()
    for i in range(nstocks):
        var_indicator = np.zeros(nstocks)
        var_indicator[i] = 1.0
        _var_link_matrix =  sparse([
                matrix(var_indicator, (nstocks,1), 'd'),
                matrix(var_indicator, (nstocks,1), 'd'),
                matrix(-var_indicator, (nstocks,1), 'd')]).T
        
        if turnover_lnk_matrix_1:
            turnover_lnk_matrix_1 = sparse([turnover_lnk_matrix_1, _var_link_matrix])
        else:
            turnover_lnk_matrix_1 = _var_link_matrix
    
  
    # Group Constraints - 10 (turnover link vars)
    turnover_lnk_matrix_2 = matrix()
    for i in range(nstocks):
        var_indicator = np.zeros(nstocks)
        var_indicator[i] = 1.0
        _var_link_matrix =  sparse([
                matrix(-var_indicator, (nstocks,1), 'd'),
                matrix(-var_indicator, (nstocks,1), 'd'),
                matrix(-var_indicator, (nstocks,1), 'd')]).T
        
        if turnover_lnk_matrix_2:
            turnover_lnk_matrix_2 = sparse([turnover_lnk_matrix_2, _var_link_matrix])
        else:
            turnover_lnk_matrix_2 = _var_link_matrix
     
    # Group constraints - 10/11
    # Market beta exposure
    # lBeta < (B1*W1 + B2*W2 + ... + BnWn) < uBeta
    market_beta_exp_matrix = sparse([
            matrix(beta.values, (nstocks,1), 'd'),
            matrix(beta.values, (nstocks,1), 'd'),
            zeros_matrix]).T
    
    market_beta_exp_bound = matrix(context.max_beta, (1,1), 'd')
    
    # Group Restrictions - 12/13
    # Portfolio Weight Upper Bound
    weight_var_matrix_long = matrix()
    for i in range(nstocks):
        var_indicator = np.zeros(nstocks)
        var_indicator[i] = 1.0
        _var_matrix =  sparse([
                matrix(var_indicator, (nstocks,1), 'd'),
                zeros_matrix,
                zeros_matrix]).T
        if weight_var_matrix_long:
            weight_var_matrix_long = sparse([weight_var_matrix_long, _var_matrix])
        else:
            weight_var_matrix_long = _var_matrix
    
    
    weight_var_matrix_short = matrix()
    for i in range(nstocks):
        var_indicator = np.zeros(nstocks)
        var_indicator[i] = 1.0
        _var_matrix =  sparse([
                zeros_matrix,
                matrix(var_indicator, (nstocks,1), 'd'),
                zeros_matrix]).T
        if weight_var_matrix_short:
            weight_var_matrix_short = sparse([weight_var_matrix_short, _var_matrix])
        else:
            weight_var_matrix_short = _var_matrix
    
    
    turnover_var_matrix = matrix()
    for i in range(nstocks):
        var_indicator = np.zeros(nstocks)
        var_indicator[i] = -1.0
        _var_matrix =  sparse([
                zeros_matrix,
                zeros_matrix,
                matrix(var_indicator, (nstocks,1), 'd')]).T
        if turnover_var_matrix:
            turnover_var_matrix = sparse([turnover_var_matrix, _var_matrix])
        else:
            turnover_var_matrix = _var_matrix
    
    
    # Combine all Group Restrictons
    G = sparse([
            sector_exp_matrix, 
            -1*sector_exp_matrix,
            
            style_exp_matrix,
            -1*style_exp_matrix,
            
            net_exp_matrix,
            -1*net_exp_matrix,
            
            gross_exp_matrix,
            -1*gross_exp_matrix,
            
            turnover_matrix,
            turnover_lnk_matrix_1,
            turnover_lnk_matrix_2,
            
            market_beta_exp_matrix,
            -1*market_beta_exp_matrix,
            
            -1*weight_var_matrix_long,
            weight_var_matrix_long,
            
            -1*weight_var_matrix_short,
            weight_var_matrix_short,
            
            turnover_var_matrix
        ])
      
    h = matrix([
            sector_exp_bounds,
            sector_exp_bounds,
            
            style_exp_bounds,
            style_exp_bounds,
            
            net_exp_bounds,
            net_exp_bounds,
            
            gross_exp_bounds_u,
            gross_exp_bounds_l,

            turnover_bound,
            turnover_lnk_bounds,
            -1*turnover_lnk_bounds,
            
            market_beta_exp_bound,
            market_beta_exp_bound,

            zeros_matrix, # xL > 0
            matrix(context.max_pos_size, (nstocks, 1), 'd'), #xL < U
            
            matrix(context.max_pos_size, (nstocks, 1), 'd'), #-xS < U
            zeros_matrix, #xS < 0
            
            zeros_matrix
        ])
    
          
    #Objective Function - Maximize Alpha
    c = matrix([
        matrix(alpha.values, (nstocks,1)),
        matrix(alpha.values, (nstocks,1)),
        zeros_matrix])
 
    sol = solvers.lp(-c, G, h)
    print sol['x']
    
def rebalance(context, data):
    
    alpha = context.alpha
    beta_pipeline = context.beta_pipeline
    
    min_weights = context.min_weights.copy(deep = True)
    max_weights = context.max_weights.copy(deep = True)
    
    context.min_turnover = 0.0

    int_port = np.zeros(len(alpha))
    allSecurities = alpha.index.tolist()
                        
    portfolio_value = context.portfolio.portfolio_value
    allPos = context.portfolio.positions
    currentSecurities = list(allPos.keys())
    defunctSecurities = list(set(currentSecurities) - set(allSecurities))                   
                        
    for (i,sec) in enumerate(allSecurities):
        if allPos[sec]:
           int_port[i] = allPos[sec].amount/portfolio_value
        
    for (i,sec) in enumerate(defunctSecurities):
        if allPos[sec]:
           context.min_turnover += allPos[sec].amount/portfolio_value
    
    context.initial_portfolio = int_port
                        
    if not alpha.empty:
        
        # Running optimization implementation (testing)         
        # optimize(context)
        
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
            beta_pipeline[['beta']],
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