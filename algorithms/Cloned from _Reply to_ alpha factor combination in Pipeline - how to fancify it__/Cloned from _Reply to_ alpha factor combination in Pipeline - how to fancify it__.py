from quantopian.algorithm import attach_pipeline, pipeline_output, order_optimal_portfolio
from quantopian.pipeline import Pipeline
from quantopian.pipeline.factors import CustomFactor, SimpleBeta, Returns
from quantopian.pipeline.data.builtin import USEquityPricing
from quantopian.pipeline.data import Fundamentals
import quantopian.optimize as opt
from sklearn import preprocessing
from quantopian.pipeline.experimental import risk_loading_pipeline
from quantopian.pipeline.filters import QTradableStocksUS
from quantopian.pipeline.data.psychsignal import stocktwits
from scipy.stats.mstats import winsorize
from zipline.utils.numpy_utils import (
    repeat_first_axis,
    repeat_last_axis,
)
from quantopian.pipeline.data import factset
 
import numpy as np

#############################
# algo settings

WIN_LIMIT = 0 # winsorize limit in factor preprocess function

# Optimize API constraints
MAX_POSITION_SIZE = 0.01 # set to 0.01 for ~100 positions
BETA_EXPOSURE = 0
USE_MaxTurnover = True # set to True to use Optimize API MaxTurnover constraint
MIN_TURN = 0.15 # Optimize API MaxTurnover constraint (if optimize fails, incrementally higher constraints will be attempted)

#############################

def preprocess(a):
    
    a = a.astype(np.float64)
    
    a[np.isinf(a)] = np.nan
    
    a = np.nan_to_num(a - np.nanmean(a))
    
    a = winsorize(a, limits=[WIN_LIMIT,WIN_LIMIT])
    
    return preprocessing.scale(a)

def make_factors():
    
    class MessageSum(CustomFactor):
        inputs = [USEquityPricing.high, USEquityPricing.low, USEquityPricing.close, stocktwits.bull_scored_messages, stocktwits.bear_scored_messages, stocktwits.total_scanned_messages]
        window_length = 21
        window_safe = True
        def compute(self, today, assets, out, high, low, close, bull, bear, total):
            v = np.nansum((high-low)/close, axis=0)
            out[:] = preprocess(v*np.nansum(total*(bear-bull), axis=0))
                
    class fcf(CustomFactor):
        inputs = [Fundamentals.fcf_yield]
        window_length = 1
        window_safe = True
        def compute(self, today, assets, out, fcf_yield):
            out[:] = preprocess(np.nan_to_num(fcf_yield[-1,:]))
                
    class Direction(CustomFactor):
        inputs = [USEquityPricing.open, USEquityPricing.close]
        window_length = 21
        window_safe = True
        def compute(self, today, assets, out, open, close):
            p = (close-open)/close
            out[:] = preprocess(np.nansum(-p,axis=0))
                
    class mean_rev(CustomFactor):   
        inputs = [USEquityPricing.high,USEquityPricing.low,USEquityPricing.close]
        window_length = 30
        window_safe = True
        def compute(self, today, assets, out, high, low, close):
            
            p = (high+low+close)/3
 
            m = len(close[0,:])
            n = len(close[:,0])
                
            b = np.zeros(m)
            a = np.zeros(m)
                
            for k in range(10,n+1):
                price_rel = np.nanmean(p[-k:,:],axis=0)/p[-1,:]
                wt = np.nansum(price_rel)
                b += wt*price_rel
                price_rel = 1.0/price_rel
                wt = np.nansum(price_rel)
                a += wt*price_rel
                
            out[:] = preprocess(b-a)
                
    class volatility(CustomFactor):
        inputs = [USEquityPricing.high, USEquityPricing.low, USEquityPricing.close, USEquityPricing.volume]
        window_length = 5
        window_safe = True
        def compute(self, today, assets, out, high, low, close, volume):
            vol = np.nansum(volume,axis=0)*np.nansum(np.absolute((high-low)/close),axis=0)
            out[:] = preprocess(-vol)
                
    class growthscore(CustomFactor):
        inputs = [Fundamentals.growth_score]
        window_length = 1
        window_safe = True
        def compute(self, today, assets, out, growth_score):
            out[:] = preprocess(growth_score[-1,:])
                
    class peg_ratio(CustomFactor):
        inputs = [Fundamentals.peg_ratio]
        window_length = 1
        window_safe = True
        def compute(self, today, assets, out, peg_ratio):
            out[:] = preprocess(-1.0/peg_ratio[-1,:])
                
    class MoneyflowVolume5d(CustomFactor):
        inputs = (USEquityPricing.close, USEquityPricing.volume)
 
        # we need one more day to get the direction of the price on the first
        # day of our desired window of 5 days
        window_length = 6
        window_safe = True
            
        def compute(self, today, assets, out, close_extra, volume_extra):
            # slice off the extra row used to get the direction of the close
            # on the first day
            close = close_extra[1:]
            volume = volume_extra[1:]
                
            dollar_volume = close * volume
            denominator = dollar_volume.sum(axis=0)
                
            difference = np.diff(close_extra, axis=0)
            direction = np.where(difference > 0, 1, -1)
            numerator = (direction * dollar_volume).sum(axis=0)
                
            out[:] = preprocess(-np.divide(numerator, denominator))
                
    class Trendline(CustomFactor):
        inputs = [USEquityPricing.close]
        window_length = 252
        window_safe = True
            
        _x = np.arange(window_length)
        _x_var = np.var(_x)
 
        def compute(self, today, assets, out, close):
            
            x_matrix = repeat_last_axis(
            (self.window_length - 1) / 2 - self._x,
            len(assets),
            )
 
            y_bar = np.nanmean(close, axis=0)
            y_bars = repeat_first_axis(y_bar, self.window_length)
            y_matrix = close - y_bars
 
            out[:] = preprocess(-np.divide(
            (x_matrix * y_matrix).sum(axis=0) / self._x_var,
            self.window_length
            ))
                
    class SalesGrowth(CustomFactor):
        inputs = [factset.Fundamentals.sales_gr_qf]
        window_length = 2*252
        window_safe = True
        def compute(self, today, assets, out, sales_growth):
            sales_growth = np.nan_to_num(sales_growth)
            sales_growth = preprocessing.scale(sales_growth,axis=0)
            out[:] = preprocess(sales_growth[-1])
 
    class GrossMarginChange(CustomFactor):
        window_length = 2*252
        window_safe = True
        inputs = [factset.Fundamentals.ebit_oper_mgn_qf]
        def compute(self, today, assets, out, ebit_oper_mgn):
            ebit_oper_mgn = np.nan_to_num(ebit_oper_mgn)
            ebit_oper_mgn = preprocessing.scale(ebit_oper_mgn,axis=0)
            out[:] = preprocess(ebit_oper_mgn[-1])
 
    class Gross_Income_Margin(CustomFactor):
        #Gross Income Margin:
        #Gross Profit divided by Net Sales
        #Notes:
        #High value suggests that the company is generating large profits
        inputs = [Fundamentals.cost_of_revenue, Fundamentals.total_revenue]
        window_length = 1
        window_safe = True
        def compute(self, today, assets, out, cost_of_revenue, sales):
            gross_income_margin = sales[-1]/sales[-1] - cost_of_revenue[-1]/sales[-1]
            out[:] = preprocess(-gross_income_margin)
 
    class MaxGap(CustomFactor): 
        # the biggest absolute overnight gap in the previous 90 sessions
        inputs = [USEquityPricing.close] ; window_length = 90
        window_safe = True
        def compute(self, today, assets, out, close):
            abs_log_rets = np.abs(np.diff(np.log(close),axis=0))
            max_gap = np.max(abs_log_rets, axis=0)
            out[:] = preprocess(max_gap)
        
    class CapEx_Vol(CustomFactor):
        inputs=[
            factset.Fundamentals.capex_assets_qf]
        window_length = 2*252
        window_safe = True
        def compute(self, today, assets, out, capex_assets):
                 
            out[:] = preprocess(-np.ptp(capex_assets,axis=0))
                
    class fcf_ev(CustomFactor):
        inputs=[
            Fundamentals.fcf_per_share,
            Fundamentals.shares_outstanding,
            Fundamentals.enterprise_value,]
        window_length = 1
        window_safe = True
        def compute(self, today, assets, out, fcf, shares, ev):
            v = fcf*shares/ev
            v[np.isinf(v)] = np.nan
                 
            out[:] = preprocess(v[-1])
                
    class DebtToTotalAssets(CustomFactor):
        inputs = [Fundamentals.long_term_debt,
            Fundamentals.current_debt,
            Fundamentals.cash_and_cash_equivalents,
            Fundamentals.total_assets]
        window_length = 1
        window_safe = True
            
        def compute(self, today, assets, out, ltd, std, cce, ta):
            std_part = np.maximum(std - cce, np.zeros(std.shape))
            v = np.divide(ltd + std_part, ta)
            v[np.isinf(v)] = np.nan
            out[:] = preprocess(np.ravel(v))
                
    class TEM(CustomFactor):
        """
        TEM = standard deviation of past 6 quarters' reports
        """
        inputs=[factset.Fundamentals.capex_qf_asof_date,
            factset.Fundamentals.capex_qf,
            factset.Fundamentals.assets]
        window_length = 390
        window_safe = True
        def compute(self, today, assets, out, asof_date, capex, total_assets):
            values = capex/total_assets
            values[np.isinf(values)] = np.nan
            out_temp = np.zeros_like(values[-1,:])
            for column_ix in range(asof_date.shape[1]):
                _, unique_indices = np.unique(asof_date[:, column_ix], return_index=True)
                quarterly_values = values[unique_indices, column_ix]
                if len(quarterly_values) < 6:
                    quarterly_values = np.hstack([
                    np.repeat([np.nan], 6 - len(quarterly_values)),
                    quarterly_values,
                    ])
            
                out_temp[column_ix] = np.std(quarterly_values[-6:])
                
            out[:] = preprocess(-out_temp)
                
    class Piotroski(CustomFactor):
        inputs = [
                Fundamentals.roa,
                Fundamentals.operating_cash_flow,
                Fundamentals.cash_flow_from_continuing_operating_activities,
                Fundamentals.long_term_debt_equity_ratio,
                Fundamentals.current_ratio,
                Fundamentals.shares_outstanding,
                Fundamentals.gross_margin,
                Fundamentals.assets_turnover,
                ]
 
        window_length = 100
        window_safe = True
    
        def compute(self, today, assets, out,roa, cash_flow, cash_flow_from_ops, long_term_debt_ratio, current_ratio, shares_outstanding, gross_margin, assets_turnover):
            
            profit = (
                        (roa[-1] > 0).astype(int) +
                        (cash_flow[-1] > 0).astype(int) +
                        (roa[-1] > roa[0]).astype(int) +
                        (cash_flow_from_ops[-1] > roa[-1]).astype(int)
                    )
        
            leverage = (
                        (long_term_debt_ratio[-1] < long_term_debt_ratio[0]).astype(int) +
                        (current_ratio[-1] > current_ratio[0]).astype(int) + 
                        (shares_outstanding[-1] <= shares_outstanding[0]).astype(int)
                        )
        
            operating = (
                        (gross_margin[-1] > gross_margin[0]).astype(int) +
                        (assets_turnover[-1] > assets_turnover[0]).astype(int)
                        )
        
            out[:] = preprocess(profit + leverage + operating)
            
    class Altman_Z(CustomFactor):
        inputs=[factset.Fundamentals.zscore_qf]
        window_length = 1
        window_safe = True
        def compute(self, today, assets, out, zscore_qf):
            out[:] = preprocess(zscore_qf[-1])
                
    class Quick_Ratio(CustomFactor):
        inputs=[factset.Fundamentals.quick_ratio_qf]
        window_length = 1
        window_safe = True
        def compute(self, today, assets, out, quick_ratio_qf):
            out[:] = preprocess(quick_ratio_qf[-1])
                
    class AdvancedMomentum(CustomFactor):
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
                
            out[:] = preprocess(-am)

    factors = [
            MessageSum,
            fcf,
            Direction,
            mean_rev,
            volatility,
            growthscore,
            peg_ratio,
            MoneyflowVolume5d,
            Trendline,
            SalesGrowth,
            GrossMarginChange,
            Gross_Income_Margin,
            MaxGap,
            CapEx_Vol,
            fcf_ev,
            DebtToTotalAssets,
            TEM,
            Piotroski,
            Altman_Z,
            Quick_Ratio,
            AdvancedMomentum,
        ]
    
    return factors
 
def factor_pipeline():
    
    factors = make_factors()
    
    pipeline_columns = {}
    for k,f in enumerate(factors):
        pipeline_columns['alpha_'+str(k)] = f(mask=QTradableStocksUS())
 
    pipe = Pipeline(columns = pipeline_columns,
    screen = QTradableStocksUS())
    return pipe

def beta_pipeline():
    
    beta = SimpleBeta(target=sid(8554),regression_length=260,
                      allowed_missing_percentage=1.0
                     )
    
    pipe = Pipeline(columns = {'beta': beta},
    screen = QTradableStocksUS())
    return pipe
    
def initialize(context):
 
    attach_pipeline(factor_pipeline(), 'factor_pipeline')
    attach_pipeline(risk_loading_pipeline(), 'risk_loading_pipeline')
    attach_pipeline(beta_pipeline(), 'beta_pipeline')
 
    # Schedule my rebalance function
    schedule_function(func=rebalance,
                      date_rule=date_rules.every_day(),
                      time_rule=time_rules.market_open(minutes=60),
                      half_days=True)
    # record my portfolio variables at the end of day
    schedule_function(func=recording_statements,
                      date_rule=date_rules.every_day(),
                      time_rule=time_rules.market_close(),
                      half_days=True)
    
    context.init = True
    
    # set_commission(commission.PerShare(cost=0, min_trade_cost=0))
    # set_slippage(slippage.FixedSlippage(spread=0))
    
def before_trading_start(context, data):
 
    risk_loadings = pipeline_output('risk_loading_pipeline')
    risk_loadings.fillna(risk_loadings.median(), inplace=True)
    context.risk_loadings = risk_loadings
    context.beta_pipeline = pipeline_output('beta_pipeline')
    
    context.combined_alpha = pipeline_output('factor_pipeline').sum(axis=1)
    
def recording_statements(context, data):
 
    record(num_positions=len(context.portfolio.positions))
    record(leverage=context.account.leverage)
 
def rebalance(context, data):
    
    combined_alpha = context.combined_alpha
 
    # demean and normalize
    combined_alpha = combined_alpha - combined_alpha.mean()
    denom = combined_alpha.abs().sum()
    combined_alpha = combined_alpha/denom
    
    objective = opt.MaximizeAlpha(combined_alpha)
    
    constraints = []
    
    constraints.append(opt.MaxGrossExposure(1.0))
    
    constraints.append(opt.DollarNeutral())
    
    constraints.append(
        opt.PositionConcentration.with_equal_bounds(
            min=-MAX_POSITION_SIZE,
            max=MAX_POSITION_SIZE
        ))
    
    risk_model_exposure = opt.experimental.RiskModelExposure(
        context.risk_loadings,
        version=opt.Newest,
    )
      
    constraints.append(risk_model_exposure)
    
    beta_neutral = opt.FactorExposure(
        loadings=context.beta_pipeline[['beta']],
        min_exposures={'beta':-BETA_EXPOSURE},
        max_exposures={'beta':BETA_EXPOSURE}
        )
    constraints.append(beta_neutral)
    
    if context.init:
        order_optimal_portfolio(
                objective=objective,
                constraints=constraints,
                )
        if USE_MaxTurnover:
            context.init = False
        return
    
    turnover = np.linspace(MIN_TURN,0.65,num=100)
    
    for max_turnover in turnover:
        
        constraints.append(opt.MaxTurnover(max_turnover))
        
        try:
            order_optimal_portfolio(
                objective=objective,
                constraints=constraints,
                )
            constraints = constraints[:-1]
            record(max_turnover = max_turnover)
            return
        except:
            constraints = constraints[:-1]