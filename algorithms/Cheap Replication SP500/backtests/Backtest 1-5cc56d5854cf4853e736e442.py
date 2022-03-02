import numpy as np
import math
import quantopian.optimize as opt
import quantopian.algorithm as algo
from quantopian.pipeline.experimental import risk_loading_pipeline
from quantopian.pipeline import CustomFactor, Pipeline
from quantopian.pipeline.data import Fundamentals
from quantopian.pipeline.data.builtin import USEquityPricing
from quantopian.pipeline.filters import QTradableStocksUS
from quantopian.pipeline.factors import SimpleBeta
from pandas import Series, DataFrame
import pandas as pd

def signalize(df):
   return ((df.rank() - 0.5)/df.count()).replace(np.nan,0.5)


nStocks = 200

def initialize(context):
    """
    Called once at the start of the algorithm.
    """

    context.max_leverage = 1.0
    context.max_pos_size = 0.01
    context.max_turnover = 0.95

    # Rebalance every day, 1 hour after market open.
    algo.schedule_function(
        rebalance,
        algo.date_rules.every_day(),
        algo.time_rules.market_close(hours=1),
    )

    # Record tracking variables at the end of each day.
    algo.schedule_function(
        record_vars,
        algo.date_rules.every_day(),
        algo.time_rules.market_close(),
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

    pipe = Pipeline(
        columns={
            'beta': beta,
        },
        screen=beta.notnull(),
    )
    algo.attach_pipeline(pipe, 'beta_pipe')

    
class Volatility(CustomFactor):  
    inputs = [USEquityPricing.close]  
    window_length = 20  
    def compute(self, today, assets, out, close):  
        # [0:-1] is needed to remove last close since diff is one element shorter  
        daily_returns = np.diff(close, axis = 0) / close[0:-1]  
        out[:] = daily_returns.std(axis = 0) * math.sqrt(252)

# Create custom factor subclass to calculate a market cap based on yesterday's
# close
class MarketCap(CustomFactor):
    # Pre-declare inputs and window_length
    inputs = [USEquityPricing.close, Fundamentals.shares_outstanding]
    window_length = 1

    # Compute market cap value
    def compute(self, today, assets, out, close, shares):
        out[:] = close[-1] * shares[-1]


def make_pipeline():
    """
    A function to create our dynamic stock selector (pipeline). Documentation
    on pipeline can be found here:
    https://www.quantopian.com/help#pipeline-title
    """

    # Base universe set to the QTradableStocksUS
    base_universe = QTradableStocksUS()

    mkt_cap = MarketCap(mask = base_universe)
    
    mkt_cap_top = mkt_cap.top(nStocks)
    
    # Filter stocks out with Mcap < mcap of 100th stock in S&P500
    pipe_screen = base_universe & mkt_cap_top

    pipe_columns = {
        'mkt_cap': mkt_cap,
    }

    return Pipeline(columns=pipe_columns, screen=pipe_screen)

def transform(df, field, multiplier=1):
    return signalize(multiplier*df[field])
        
def before_trading_start(context, data):
    """
    Called every day before market open.
    """
    output = algo.pipeline_output('pipeline')
    output['signal_rank'] = transform(output, 'mkt_cap')
    
    output['mkt_cap_weights'] = output['mkt_cap']/output['mkt_cap'].sum()

    # print output['mkt_cap_weights']

    output['min_weights'] = 0 
    output['max_weights'] = 0.045 * np.ones(nStocks)
    
    context.min_weights = output['min_weights']
    context.max_weights = output[['max_weights','mkt_cap_weights']].min(axis=1)

    alpha = output['signal_rank']
    # alpha = alpha.sub(alpha.mean())
    context.alpha = alpha
    

    context.risk_factor_betas = algo.pipeline_output(
      'risk_pipe'
    ).dropna()

    context.beta_pipeline = algo.pipeline_output('beta_pipe')


def rebalance(context, data):
    
    min_weights = context.min_weights
    max_weights = context.max_weights

    print max_weights

    beta_pipeline = context.beta_pipeline
    alpha = context.alpha

    if not alpha.empty:
        # Create MaximizeAlpha objective
        # objective = opt.TargetWeights(weights)
        objective = opt.MaximizeAlpha(alpha)

        # Create position size constraint
        constrain_pos_size = opt.PositionConcentration(
            min_weights,
            max_weights# context.max_pos_size
        )

        # Constrain target portfolio's leverage
        max_leverage = opt.MaxGrossExposure(context.max_leverage)

        # Constrain portfolio turnover
        max_turnover = opt.MaxTurnover(context.max_turnover)
        
        factor_risk_constraints = opt.experimental.RiskModelExposure(
            context.risk_factor_betas,
            max_volatility = 0.2,
            version=opt.Newest
        )
        
        beta_market = opt.FactorExposure(
            beta_pipeline[['beta']],
            min_exposures={'beta': 0.90},
            max_exposures={'beta': 1.1},
        )

        
        # Rebalance portfolio using objective
        # and list of constraints
        algo.order_optimal_portfolio(
            objective=objective,
            constraints=[
                constrain_pos_size,
                max_leverage,
                # max_turnover,
                # factor_risk_constraints,
                beta_market
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
