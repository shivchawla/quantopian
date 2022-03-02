"""
This is a template algorithm on Quantopian for you to adapt and fill in.
"""
import quantopian.algorithm as algo
import quantopian.optimize as opt
from quantopian.pipeline import CustomFactor, Pipeline
from quantopian.pipeline.experimental import risk_loading_pipeline
from quantopian.pipeline.data.builtin import USEquityPricing
from quantopian.pipeline.factors import Returns, DailyReturns
from quantopian.pipeline.filters import QTradableStocksUS, Q500US
import numpy as np
from statsmodels import regression
import statsmodels.api as sm


index_returns = DailyReturns()[symbol('SPY')]
returns = DailyReturns()

class PositiveBeta(CustomFactor):
    window_safe = True
    inputs=[returns, index_returns]
    window_length = 252
    
    def compute(self, today, assets, out, returns, index):
        idxr = np.copy(index)
        idxr[idxr<0] = np.nan
        
        X = idxr
        X = sm.add_constant(X) 

        model = regression.linear_model.OLS(returns, X, missing='drop').fit()
        intercept, slope = model.params
        out[:] = slope
        
class NegativeBeta(CustomFactor):
    window_safe = True
    inputs=[returns, index_returns]
    window_length = 252
    
    def compute(self, today, assets, out, returns, index):
        # index[index>0] = np.nan 
        idxr  = np.copy(index)
        idxr[idxr>0] = np.nan
        
        X = idxr
        X = sm.add_constant(X) 
        
        model = regression.linear_model.OLS(returns, X, missing='drop').fit()
        intercept, slope = model.params
        out[:] = slope


def initialize(context):
    """
    Called once at the start of the algorithm.
    """
    
    context.max_pos_size = 0.05
    # Rebalance every day, 1 hour after market open.
    algo.schedule_function(
        rebalance,
        algo.date_rules.every_day(),
        algo.time_rules.market_open(hours=1),
    )

    # Record tracking variables at the end of each day.
    algo.schedule_function(
        record_vars,
        algo.date_rules.every_day(),
        algo.time_rules.market_close(),
    )
    
    set_slippage(slippage.FixedSlippage(spread=0.0))
    set_commission(commission.PerShare(cost=0, min_trade_cost=0))
    
    # Create our dynamic stock selector.
    algo.attach_pipeline(make_pipeline(), 'pipeline')
    
    algo.attach_pipeline(
        risk_loading_pipeline(),
        'risk_pipe'
    )
        
        

def make_pipeline():
    """
    A function to create our dynamic stock selector (pipeline). Documentation
    on pipeline can be found here:
    https://www.quantopian.com/help#pipeline-title
    """

    # Base universe set to the QTradableStocksUS
    # base_universe = QTradableStocksUS()
    base_universe = Q500US()
    
    # returns = DailyReturns(inputs=[USEquityPricing.close])
    # index = returns[symbol("SPY")]
    
    # index_positive = index.clip(min_bound=0, max_bound=np.inf)
    # index_negative = index.clip(min_bound=-np.inf, max_bound=0)
      
    regressions_positive = PositiveBeta(mask=base_universe)
    
    regressions_negative = NegativeBeta(mask=base_universe)
    
    pipe =  Pipeline(
            columns = {"beta_postive":regressions_positive,
                       "beta_negative":regressions_negative
                      },
            screen=base_universe
            )
    return pipe

    

def before_trading_start(context, data):
    """
    Called every day before market open.
    """
    output = algo.pipeline_output('pipeline')   
    
    context.alpha = (output['beta_postive'] - output['beta_negative'])/(output['beta_postive'] + output['beta_negative'])
    # .sort_values()
    # .zscore()
    
    # context.alpha[100:-100] = 0
    # context.alpha.drop(context.alpha[10:-10].index)

    context.alpha = context.alpha/context.alpha.abs().sum()
     
    output['min_weights'] = 0
    output['max_weights'] = context.max_pos_size
    
    context.min_weights = output['min_weights']
    context.max_weights = output['max_weights']
    
    
    # These are the securities that we are interested in trading each day.
    context.security_list = output.index
    
    context.risk_factor_betas = algo.pipeline_output(
      'risk_pipe'
    ).dropna()



def rebalance(context, data):
    """
    Execute orders according to our schedule_function() timing.
    """
    alpha = context.alpha
    
    if not alpha.empty:
        # Create MaximizeAlpha objective
        objective = opt.MaximizeAlpha(alpha)

        constrain_pos_size = opt.PositionConcentration(
            context.min_weights,
            context.max_weights
        )

        # Constrain target portfolio's leverage
        max_leverage = opt.MaxGrossExposure(1.0)

        # Ensure long and short books
        # are roughly the same size
        # dollar_neutral = opt.DollarNeutral(0.0)
        
        # Constrain portfolio turnover
        # max_turnover = opt.MaxTurnover(context.max_turnover)
        
        factor_risk_constraints = opt.experimental.RiskModelExposure(
            context.risk_factor_betas,
            version=opt.Newest
        )
        
        # beta_neutral = opt.FactorExposure(
        #     beta,
        #     min_exposures={'beta': context.min_beta},
        #     max_exposures={'beta': context.max_beta},
        # )
        
        # if chg <= 0:
        algo.order_optimal_portfolio(
                objective = objective,
                constraints = [
                    constrain_pos_size,
                    max_leverage,
                    # dollar_neutral,
                    # max_turnover,
                    factor_risk_constraints,
                    # beta_neutral
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