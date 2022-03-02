import numpy as np
import math
import quantopian.optimize as opt
import quantopian.algorithm as algo
from quantopian.pipeline.experimental import risk_loading_pipeline
from quantopian.pipeline import CustomFactor, Pipeline
from quantopian.pipeline.data import Fundamentals
from quantopian.pipeline.data.builtin import USEquityPricing
from quantopian.pipeline.filters import QTradableStocksUS

nStocks = 50

def signalize(df):
   return ((df.rank() - 0.5)/df.count()).replace(np.nan,0.5)


def initialize(context):
    """
    Called once at the start of the algorithm.
    """

    context.max_leverage = 1.0
    context.max_pos_size = 0.015
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

class EnterpriseValue(CustomFactor):
    # Pre-declare inputs and window_length
    inputs = [Fundamentals.enterprise_value]
    window_length = 1

    # Compute market cap value
    def compute(self, today, assets, out, ev):
        out[:] = ev[-1]

        
class LogEV(CustomFactor):
    # Pre-declare inputs and window_length
    inputs = [Fundamentals.enterprise_value]
    window_length = 1

    # Compute market cap value
    def compute(self, today, assets, out, ev):
        out[:] = np.log(ev[-1])
 
        
class LogMarketCap(CustomFactor):
    # Pre-declare inputs and window_length
    inputs = [USEquityPricing.close, Fundamentals.shares_outstanding]
    window_length = 1

    # Compute market cap value
    def compute(self, today, assets, out, close, shares):
        out[:] = np.log(close[-1] * shares[-1])

class CashReturn(CustomFactor):   
    
    # Pre-declare inputs and window_length
    inputs = [Fundamentals.cash_return] 
    window_length = 1
    
    # Compute factor1 value
    def compute(self, today, assets, out, cash_return):       
        out[:] = cash_return[-1]


class RevenueGrowth(CustomFactor):   
    
    # Pre-declare inputs and window_length
    inputs = [Fundamentals.revenue_growth] 
    window_length = 1
    
    # Compute factor1 value
    def compute(self, today, assets, out, revenue_growth):       
        out[:] = revenue_growth[-1]


class ROE(CustomFactor):   
    
    # Pre-declare inputs and window_length
    inputs = [Fundamentals.roe] 
    window_length = 1
    
    # Compute factor1 value
    def compute(self, today, assets, out, roe):
        out[:] = roe[-1]


class ROIC(CustomFactor):   
    
    # Pre-declare inputs and window_length
    inputs = [Fundamentals.roic] 
    window_length = 1
    
    # Compute factor1 value
    def compute(self, today, assets, out, roic):
        out[:] = roic[-1]

class DilutedEPSGrowth(CustomFactor):   

    # Pre-declare inputs and window_length
    inputs = [Fundamentals.diluted_eps_growth] 
    window_length = 1
    
    # Compute factor1 value
    def compute(self, today, assets, out, deps_growth):
        out[:] = deps_growth[-1]


class AssetsTurnover(CustomFactor):   

    # Pre-declare inputs and window_length
    inputs = [Fundamentals.assets_turnover] 
    window_length = 1
    
    # Compute factor1 value
    def compute(self, today, assets, out, assets_turnover):
        out[:] = assets_turnover[-1]


class CommonEquityToAssets(CustomFactor):   

    # Pre-declare inputs and window_length
    inputs = [Fundamentals.common_equity_to_assets] 
    window_length = 1
    
    # Compute factor1 value
    def compute(self, today, assets, out, common_equity_to_assets):
        out[:] = common_equity_to_assets[-1]


class CurrentRatio(CustomFactor):   

    # Pre-declare inputs and window_length
    inputs = [Fundamentals.current_ratio] 
    window_length = 1
    
    # Compute factor1 value
    def compute(self, today, assets, out, var):
        out[:] = var[-1]

     
class PretaxMargin(CustomFactor):   

    # Pre-declare inputs and window_length
    inputs = [Fundamentals.pretax_margin] 
    window_length = 1
    
    # Compute factor1 value
    def compute(self, today, assets, out, var):
        out[:] = var[-1]


class SalesYield(CustomFactor):   

    # Pre-declare inputs and window_length
    inputs = [Fundamentals.sales_yield] 
    window_length = 1
    
    # Compute factor1 value
    def compute(self, today, assets, out, var):
        out[:] = var[-1]
        

class FcfToEV(CustomFactor):
    # Pre-declare inputs and window_length
    inputs = [Fundamentals.enterprise_value, Fundamentals.free_cash_flow] 
    window_length = 1
    
    # Compute factor1 value
    def compute(self, today, assets, out, ev, var):
        out[:] = var[-1]/ev[-1]


def make_pipeline():
    """
    A function to create our dynamic stock selector (pipeline). Documentation
    on pipeline can be found here:
    https://www.quantopian.com/help#pipeline-title
    """

    # Base universe set to the QTradableStocksUS
    base_universe = QTradableStocksUS()

    # Factor of yesterday's close price.
    yesterday_close = USEquityPricing.close.latest

    mkt_cap = MarketCap(mask = base_universe)
    log_mkt_cap = LogMarketCap(mask = base_universe)
    log_ev = LogEV(mask = base_universe)

    vol = Volatility(mask = base_universe)

    cash_return = CashReturn(mask = base_universe)
    revenue_growth = RevenueGrowth(mask = base_universe)
    roe = ROE(mask = base_universe)
    roic = ROIC(mask = base_universe)
    assets_turnover = AssetsTurnover(mask = base_universe)
    deps_growth = DilutedEPSGrowth(mask = base_universe)  
    common_equity_to_assets = CommonEquityToAssets(mask = base_universe)  
    current_ratio = CurrentRatio(mask = base_universe)
    pretax_margin = PretaxMargin(mask = base_universe)
    sales_yield = SalesYield(mask = base_universe)
    fcfToEv = FcfToEV(mask = base_universe)

    w_roe = roe * log_mkt_cap
    w_cash_return = cash_return * log_mkt_cap
    w_roic = roic * log_mkt_cap
    w_revenue_growth = revenue_growth * log_mkt_cap
    w_deps_growth = deps_growth * log_mkt_cap
    w_assets_turnover = assets_turnover * log_mkt_cap 
    w_common_equity_to_assets = common_equity_to_assets * log_mkt_cap
    w_current_ratio = current_ratio * log_mkt_cap
    w_pretax_margin = pretax_margin * log_mkt_cap
    w_sales_yield = sales_yield * log_ev


    mkt_cap_top = mkt_cap.top(1000)

    # Filter stocks out with Mcap < mcap of 100th stock in S&P500
    pipe_screen = mkt_cap_top & base_universe 
    # & cash_return_negative

    pipe_columns = {
        'mkt_cap': mkt_cap,
        'log_mkt_cap': log_mkt_cap,
        'vol': vol,
        'cash_return': cash_return,
        'revenue_growth': revenue_growth,
        'roe': roe,
        'roic': roic,
        'deps_growth': deps_growth,
        'assets_turnover': assets_turnover,
        'common_equity_to_assets':common_equity_to_assets,
        'current_ratio': current_ratio,
        'pretax_margin': pretax_margin,
        'fcfToEv':fcfToEv,
        'w_cash_return': w_cash_return,
        'w_revenue_growth': w_revenue_growth,
        'w_roe': w_roe,
        'w_roic': w_roic,
        'w_deps_growth': w_deps_growth,
        'w_assets_turnover': w_assets_turnover, #8.65%
        'w_common_equity_to_assets': w_common_equity_to_assets,
        'w_current_ratio': w_current_ratio,
        'w_pretax_margin': w_pretax_margin,
        'w_sales_yield': w_sales_yield
    }


    return Pipeline(columns=pipe_columns, screen=pipe_screen)

def transform(df, field, multiplier=1):
    return signalize(multiplier*df[field])
        
def before_trading_start(context, data):
    """
    Called every day before market open.
    """
    output = algo.pipeline_output('pipeline')
    output['combined'] = transform(output, 'fcfToEv', 1)
    # /transform(output, 'w_pretax_margin')
    # transform(output,'assets_turnover')
    output['signal_rank'] = transform(output, 'combined', 1)
   
    output = output.sort_values(by = 'signal_rank', ascending=True)
    alpha = output['signal_rank']
    alpha = alpha.sub(alpha.mean())
    context.alpha = alpha
    # print  output[:nStocks]['signal_rank']
    print alpha.head(10)
    print alpha.tail(10)
    
    context.risk_factor_betas = algo.pipeline_output(
      'risk_pipe'
    ).dropna()




def rebalance(context, data):
    
    alpha = context.alpha

    if not alpha.empty:
        # Create MaximizeAlpha objective
        objective = opt.MaximizeAlpha(alpha)

        # Create position size constraint
        constrain_pos_size = opt.PositionConcentration.with_equal_bounds(
            -context.max_pos_size,
            context.max_pos_size
        )

        # Constrain target portfolio's leverage
        max_leverage = opt.MaxGrossExposure(context.max_leverage)

        # Ensure long and short books
        # are roughly the same size
        dollar_neutral = opt.DollarNeutral()

        # Constrain portfolio turnover
        max_turnover = opt.MaxTurnover(context.max_turnover)
        
        factor_risk_constraints = opt.experimental.RiskModelExposure(
            context.risk_factor_betas,
            version=opt.Newest
        )
        
        # Rebalance portfolio using objective
        # and list of constraints
        algo.order_optimal_portfolio(
            objective=objective,
            constraints=[
                constrain_pos_size,
                max_leverage,
                dollar_neutral,
                max_turnover,
                factor_risk_constraints
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