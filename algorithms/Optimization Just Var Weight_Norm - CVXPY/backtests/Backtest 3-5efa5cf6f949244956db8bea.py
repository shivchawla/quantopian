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


def optimize(context, alpha, factor_loadings, beta):
            
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
    market_beta_exp_matrix = scipysp.csc_matrix(
                np.matrix(beta.values.reshape((nstocks,))))
            
    market_beta_exp_bound = np.full((1), context.max_beta)

    # Create optimization variables
    w = cp.Variable(nstocks)
    
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
    beta_deviation = cp.sum_squares(market_beta_exp_matrix*w - 0.0)
    # objective = cp.Maximize(total_alpha - beta_deviation)
    objective = cp.Maximize(total_alpha)
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
        my_opt_weights = optimize(context, alpha, factor_loadings, beta)
        my_opt = pd.Series(my_opt_weights, index=validSecurities) 
        
        #print(my_opt.head(10))

        # Create MaximizeAlpha objective
        # objective = opt.MaximizeAlpha(alpha)

        # constrain_pos_size = opt.PositionConcentration(
        #     min_weights,
        #     max_weights
        # )

        # # Constrain target portfolio's leverage
        # max_leverage = opt.MaxGrossExposure(context.max_leverage)

        # # Ensure long and short books
        # # are roughly the same size
        # dollar_neutral = opt.DollarNeutral(context.max_net_exposure)
        
        # # Constrain portfolio turnover
        # max_turnover = opt.MaxTurnover(context.max_turnover)
        
        # factor_risk_constraints = opt.experimental.RiskModelExposure(
        #     context.risk_factor_betas,
        #     version=opt.Newest
        # )
        
        # beta_neutral = opt.FactorExposure(
        #     beta,
        #     min_exposures={'beta': -context.max_beta},
        #     max_exposures={'beta': context.max_beta},
        # )

        # q_opt = opt.calculate_optimal_portfolio(objective,[
        #         constrain_pos_size,
        #         max_leverage,
        #         dollar_neutral,
        #         max_turnover,
        #         factor_risk_constraints,
        #         beta_neutral
        #     ])
        
        #print(q_opt.head(10))   

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