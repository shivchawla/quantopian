import numpy as np
import math
import pandas as pd
import quantopian.optimize as opt
import quantopian.algorithm as algo
from quantopian.pipeline.experimental import risk_loading_pipeline
from quantopian.pipeline import CustomFactor, Pipeline
from quantopian.pipeline.data import Fundamentals 
from quantopian.pipeline.data.factset import Fundamentals as FstFundamentals
from quantopian.pipeline.data.builtin import USEquityPricing
from quantopian.pipeline.filters import QTradableStocksUS
from quantopian.pipeline.factors import SimpleBeta, Returns
from quantopian.pipeline.data.psychsignal import stocktwits
from quantopian.pipeline.factors import AverageDollarVolume
from cvxopt import matrix, solvers, spdiag, spmatrix, sparse

def signalize(df):
   return ((df.rank() - 0.5)/df.count()).replace(np.nan,0.5)
  
def initialize(context):
    """
    Called once at the start of the algorithm.
    """

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
    
    value_beta = SimpleBeta(
        target=sid(22010),
        regression_length=260,
    )
    
    growth_beta = SimpleBeta(
        target=sid(22009),
        regression_length=260,
    )
    
    large_beta = SimpleBeta(
        target=sid(22148),
        regression_length=260,
    )
    
    small_beta = SimpleBeta(
        target=sid(21508),
        regression_length=260,
    )
    
    
    beta_pipe = Pipeline(
        columns={
            'beta': beta,
            'value_beta': value_beta,
            'growth_beta': growth_beta,
            'small_beta': small_beta,
            'large_beta': large_beta,
        },
        screen = QTradableStocksUS() & beta.notnull(),
    )
    algo.attach_pipeline(beta_pipe, 'beta_pipe')

    
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


class Factor1(CustomFactor):   

    # Pre-declare inputs and window_length
    inputs = [Fundamentals.pcf_ratio] 
    window_length = 2
    
    # Compute factor1 value
    def compute(self, today, assets, out, var):
        out[:] = var[-2]
        
class Factor2(CustomFactor):   

    # Pre-declare inputs and window_length
    inputs = [Fundamentals.ps_ratio] 
    window_length = 2
    
    # Compute factor2 value
    def compute(self, today, assets, out, var):
        out[:] = var[-2]
    
class Factor3(CustomFactor):
    # Pre-declare inputs and window_length
    inputs = [Fundamentals.enterprise_value, Fundamentals.free_cash_flow, USEquityPricing.close, Fundamentals.shares_outstanding, Fundamentals.total_assets] 
    window_length = 2
    
    # Compute factor3 value
    def compute(self, today, assets, out, ev, var, close, shares, ta):
        out[:] = var[-2]/(ev[-2]*close[-2]*shares[-2]*ta[-2])**(1./3.)

class Factor4(CustomFactor):
    # Pre-declare inputs and window_length
    inputs = [Fundamentals.enterprise_value, Fundamentals.free_cash_flow, USEquityPricing.close, Fundamentals.shares_outstanding, Fundamentals.total_assets] 
    window_length = 2
    
    # Compute factor4 value
    def compute(self, today, assets, out, ev, var, close, shares, ta):
        out[:] = ta[-2]/(ev[-2]*close[-2]*shares[-2])**(1./2.)
        
class Factor5(CustomFactor):
    """
    TEM = standard deviation of past 6 quarters' reports
    """
    inputs = [Fundamentals.capital_expenditure, Fundamentals.enterprise_value] 
    window_length = 390
    
    # Compute factor5 value
    def compute(self, today, assets, out, capex, ev):
        values = capex/ev
        out[:] = values.std(axis = 0)

class Factor6(CustomFactor):  
        inputs = [Fundamentals.forward_earning_yield]  
        window_length = 2

        # Compute factor6 value  
        def compute(self, today, assets, out, syield):  
            out[:] =  syield[-2]

class Factor7(CustomFactor):  
        inputs = [Fundamentals.earning_yield]  
        window_length = 2

        # Compute factor6 value  
        def compute(self, today, assets, out, syield):  
            out[:] =  syield[-2]

class Factor8(CustomFactor):  
        inputs = [Fundamentals.sales_yield]  
        window_length = 2

        # Compute factor6 value  
        def compute(self, today, assets, out, syield):  
            out[:] =  syield[-2]

class Factor9(CustomFactor):
        inputs = [USEquityPricing.high, USEquityPricing.low, USEquityPricing.close, stocktwits.bull_scored_messages, stocktwits.bear_scored_messages, stocktwits.total_scanned_messages]
        window_length = 21
        window_safe = True
        def compute(self, today, assets, out, high, low, close, bull, bear, total):
            v = np.nansum((high-low)/close, axis=0)
            out[:] = v*np.nansum(total*(bear-bull), axis=0)

#Contest Entry#3 - 07/05/2019
class Factor10(CustomFactor):
    inputs = [Fundamentals.capital_expenditure, Fundamentals.cost_of_revenue] 
    window_length = 360
    
    def compute(self, today, assets, out, capex, cr):
        values = capex/cr
        out[:] = values.mean(axis = 0)
        
class Factor11(CustomFactor):
    inputs = [Fundamentals.revenue_growth] 
#     inputs = [Fundamentals.operation_revenue_growth3_month_avg] 
    window_length = 360
    def compute(self, today, assets, out, rate):
        out[:] = rate.mean(axis = 0)/rate.std(axis = 0)        

        
class Factor12(CustomFactor):
    inputs = [Fundamentals.gross_margin] 
    window_length = 360
    def compute(self, today, assets, out, rate):
        out[:] = rate.mean(axis = 0)/rate.std(axis = 0)        

        
class Factor13(CustomFactor):
    inputs = [Fundamentals.quick_ratio] 
    window_length = 360
    def compute(self, today, assets, out, rate):
        out[:] = rate.mean(axis = 0)/rate.std(axis = 0)        
 
        
class Factor14(CustomFactor):
    inputs = [Fundamentals.ebitda_margin] 
    window_length = 360
    def compute(self, today, assets, out, rate):
        out[:] = 1/rate.std(axis = 0)        

        
class Factor15(CustomFactor):
    inputs = [Fundamentals.current_ratio] 
    window_length = 360
    def compute(self, today, assets, out, rate):
        out[:] = rate.mean(axis = 0)/rate.std(axis = 0)        

        
class Factor16(CustomFactor):
    inputs = (USEquityPricing.close, Returns(window_length=126))
    window_length = 252
    window_safe = True

    def compute(self, today, assets, out, prices, returns):
        am = np.divide(
            (
                (prices[-21] - prices[-252]) / prices[-252] -
                prices[-1] - prices[-21]
            ) / prices[-21],
            np.nanstd(returns, axis=0)
        )

        out[:] = -am

                
        
def make_pipeline():
    """
    A function to create our dynamic stock selector (pipeline). Documentation
    on pipeline can be found here:
    https://www.quantopian.com/help#pipeline-title
    """

    # Base universe set to the QTradableStocksUS
    base_universe = QTradableStocksUS()

    mkt_cap = MarketCap(mask = base_universe)
    log_mkt_cap = LogMarketCap(mask = base_universe)
    vol = Volatility(mask = base_universe)
    
    f1 = Factor1(mask = base_universe)
    f2 = Factor2(mask = base_universe)
    f3 = Factor3(mask = base_universe)
    f4 = Factor4(mask = base_universe)
    f5 = Factor5(mask = base_universe)
    f6 = Factor6(mask = base_universe)
    f7 = Factor7(mask = base_universe)
    f8 = Factor8(mask = base_universe)
    f9 = Factor9(mask = base_universe)
    f10 = Factor10(mask = base_universe)
    f11 = Factor11(mask = base_universe)
    f12 = Factor12(mask = base_universe)
    f13 = Factor13(mask = base_universe)
    f14 = Factor14(mask = base_universe)
    f15 = Factor15(mask = base_universe)
    f16 = Factor16(mask = base_universe)

    # Filter stocks out with Mcap < mcap of 100th stock in S&P500
    pipe_screen = base_universe 

    pipe_columns = {
        'mkt_cap': mkt_cap,
        'log_mkt_cap': log_mkt_cap,
        # 'log_ev':log_ev,
        'vol': vol,
        'f1': f1,
        'f2': f2,
        'f3': f3,
        'f4': f4,
        'f5': f5,
        'f6': f6,
        'f7': f7,
        'f8': f8,
        'f9': f9,
        'f10': f10,
        'f11': f11,
        'f12': f12,
        'f13': f13,
        'f14': f14,
        'f15': f15,
        'f16': f16,
    }

    return Pipeline(columns=pipe_columns, screen=pipe_screen)

def transform(df, field, multiplier=1):
    return signalize(multiplier*df[field])

def weightedMean(df, weights):
    return df.mean()
    # return (df*weights).mean()/weights.mean()


def before_trading_start(context, data):
    """
    Called every day before market open.
    """
    output = algo.pipeline_output('pipeline')
    b_output = algo.pipeline_output('beta_pipe')
    
    beta_rank = transform(b_output, 'beta', 1)
    inv_beta_rank = transform(b_output, 'beta', -1)
    
    mkt_cap = output['mkt_cap']
    volatility = output['vol']
    
    mk_rank = transform(output, 'log_mkt_cap', 1)
    inv_mk_rank = transform(output, 'log_mkt_cap', -1)
    vol_rank = transform(output, 'vol', -1)
    
    # Alpha Factors
    alpha1 = signalize(transform(output, 'f1', 1)*mk_rank)
    alpha2 = signalize(signalize(transform(output, 'f2', 1)*mk_rank)*vol_rank)
    alpha3 = transform(output, 'f3', 1)
    alpha4 = transform(output, 'f4', -1)
    alpha5 = transform(output, 'f5', -1)
    alpha6 = transform(output, 'f6', -1)
    alpha7 = transform(output, 'f7', -1)
    alpha8 = transform(output, 'f8', -1)
    alpha9 = signalize(transform(output, 'f9', 1)*vol_rank)
    alpha10 = transform(output, 'f10', 1)
    alpha13 = signalize(transform(output, 'f13', 1)*mkt_cap)
    alpha14 = signalize(transform(output, 'f14', 1)*(1/mkt_cap))
    alpha15 = signalize(transform(output, 'f15', 1)*(1/mkt_cap))
    alpha16 = signalize(transform(output, 'f16', 1)*(1/volatility))
    
    alpha1 = alpha1.sub(alpha1.mean())
    alpha2 = alpha2.sub(alpha2.mean())
    alpha3 = alpha3.sub(alpha3.mean())
    alpha3[alpha3 < 0] = 0
    alpha4 = alpha4.sub(alpha4.mean())
    alpha5 = alpha5.sub(alpha5.mean())
    alpha6 = alpha6.sub(alpha6.mean())
    alpha7 = alpha7.sub(alpha7.mean())
    alpha8 = alpha8.sub(alpha8.mean())
    alpha9 = alpha9.sub(alpha9.mean())
    alpha10 = alpha10.sub(alpha10.mean())
    alpha13 = alpha13.sub(alpha13.mean())
    alpha14 = alpha14.sub(alpha14.mean())
    alpha15 = alpha15.sub(alpha15.mean())
    alpha16 = alpha16.sub(alpha16.mean())
    

    ra = alpha1 + alpha2 + alpha3 + alpha4 + alpha5 + alpha6 + alpha7 + alpha8 + alpha9 + alpha10 + alpha13 + alpha14 + alpha15 + alpha16
    
    output['signal_rank'] = ra 
    # + alpha17
    context.alpha = output['signal_rank']
    
    output['min_weights'] = -context.max_pos_size
    output['max_weights'] = context.max_pos_size
    
    context.min_weights = output['min_weights']
    context.max_weights = output['max_weights']
    
    context.risk_factor_betas = algo.pipeline_output(
      'risk_pipe'
    ).dropna()
    
    # print context.risk_factor_betas

    context.beta_pipeline = algo.pipeline_output('beta_pipe')

    
def optimize(context):
    alpha = context.alpha
    # print alpha
    factor_loadings = context.risk_factor_betas.loc[alpha.index.tolist(), :]
    beta = context.beta_pipeline[['beta']].loc[alpha.index.tolist(), :]
    # print beta
    
    nstocks = alpha.size
    initial_portfolio = context.initial_portfolio
    min_turnover = context.min_turnover
    max_turnover = context.max_turnover - min_turnover
    
    print "Max Turnover"
    print max_turnover
    
    print "Initial Portfolio"
    print initial_portfolio
    
    print alpha.values
    # w = wl + ws
    #Number of variables = 
    #    2*nStocks(weights) + 
    #    nStocks(turnover link vars)
    
    #Number of Inequality group restrictions = 
    #    nFactors (factors) + 
    #    1 (net exposure restrictions)
    #    1 (gross exposure restriction) +
    #    1 (turnover restriction)
    #    nStocks(turnover link vars) +
    #    nStocks(turnover link vars) +
    #    1 (market beta exposure) +
    #    nStocks (Weight upper bound)
    #    nStocks (Weight lower bound)
    #    nStocks (turnover link vars lower bound)
     
    ones_matrix = matrix(1.0, (nstocks, 1), 'd')
    zeros_matrix = matrix(0.0, (nstocks, 1), 'd')
    
    # Group Constraints - 1/2
    # min_exposure < Risk Loading transpose * W < ma_exposure
    sector_exp_matrix = matrix()
    nsectors = len(context.sectors)
    sector_exp_bounds = matrix(context.max_sector_exposure, (nsectors,1), 'd')

    for col_name in context.sectors:
        # print "Factor Loadings"
        # print col_name
        # print factor_loadings[col_name].values
        # print len(factor_loadings[col_name].values)
        
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
    # dw >= w - wo #new variable  (Number of linear restrictions = nstocks)
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
    
    # print turnover_lnk_bounds
    
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
    # Z = (B1*W1 + B2*W2 + ... + BnWn) 
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
    
    # print sector_exp_matrix.size
    # print style_exp_matrix.size
    # print net_exp_matrix.size
    # print gross_exp_matrix.size
    # print gross_exp_var_lnk_matrix_1.size
    # print turnover_matrix.size
    # print turnover_lnk_matrix_1.size
    # print market_beta_exp_matrix.size
    # print weight_var_matrix.size
    # print gross_exp_var_matrix.size
    # print turnover_var_matrix.size
    
    # print "Total Constraint"
    # print 2*sector_exp_matrix.size[0] + 2*style_exp_matrix.size[0] + 2*net_exp_matrix.size[0] + 2*gross_exp_matrix.size[0] + gross_exp_var_lnk_matrix_1.size[0] + turnover_matrix.size[0] + turnover_lnk_matrix_1.size[0] + 2*market_beta_exp_matrix.size[0] + 2*weight_var_matrix.size[0] + gross_exp_var_matrix.size[0] + turnover_var_matrix.size[0]
    
    # Combine all Group Restrictons
    G = sparse([
            # sector_exp_matrix, 
            # -1*sector_exp_matrix,
            # style_exp_matrix,
            # -1*style_exp_matrix,
            # net_exp_matrix,
            # -1*net_exp_matrix,
            gross_exp_matrix,
            # -1*gross_exp_matrix,
            # gross_exp_var_lnk_matrix_1,
            # gross_exp_var_lnk_matrix_2,
            
            turnover_matrix,
            turnover_lnk_matrix_1,
            turnover_lnk_matrix_2,
            
            # market_beta_exp_matrix,
            # -1*market_beta_exp_matrix,
            
            # -1*weight_var_matrix_long,
            # weight_var_matrix_long,
            
            # -1*weight_var_matrix_short,
            # weight_var_matrix_short,
            
            # gross_exp_var_matrix,
            # turnover_var_matrix
        ])
    
    print "G-size"
    print G.size
    
    h = matrix([
            # sector_exp_bounds,
            # sector_exp_bounds,
            # style_exp_bounds,
            # style_exp_bounds,
            # net_exp_bounds,
            # net_exp_bounds,
            gross_exp_bounds_u,
            # gross_exp_bounds_l,

            turnover_bound,
            turnover_lnk_bounds,
            -1*turnover_lnk_bounds,
            
            # market_beta_exp_bound,
            # market_beta_exp_bound,

            # zeros_matrix, # xL > 0
            # matrix(context.max_pos_size, (nstocks, 1), 'd'), #xL < U
            
            # matrix(context.max_pos_size, (nstocks, 1), 'd'), #-xS < U
            # zeros_matrix, #xS < 0
            
            # zeros_matrix
        ])
    
    print "H-Size"
    print h.size
           
    P = spmatrix(0.0, range(3*nstocks), range(3*nstocks))
    
    #Objective Function - Maximize Alpha
    q = matrix([
        matrix(alpha.values, (nstocks,1)),
        matrix(alpha.values, (nstocks,1)),
        zeros_matrix])
    
    print "C-size"
    print q.size
    
    # try:
    sol = solvers.qp(P, -q, G, h)
    print sol['x']
    # except:
    #     print "Oops Exception"
    
def rebalance(context, data):
    
    alpha = context.alpha
    beta_pipeline = context.beta_pipeline
    # print beta_pipeline
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
        
        optimize(context)
        
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