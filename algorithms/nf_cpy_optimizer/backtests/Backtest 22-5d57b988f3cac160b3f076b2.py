from quantopian.algorithm import attach_pipeline, pipeline_output
from quantopian.pipeline import Pipeline
from quantopian.pipeline.factors import CustomFactor, SimpleBeta, Returns
from quantopian.pipeline.data.builtin import USEquityPricing
from quantopian.pipeline.data import Fundamentals
import quantopian.optimize as opt
import quantopian.algorithm as algo
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

from scipy.stats.mstats import gmean
from sklearn.cluster import SpectralClustering

import math
import numpy as np
import pandas as pd

from collections import Counter


from scipy import stats, linalg
import cvxpy as cp
from scipy import sparse as scipysp


WIN_LIMIT = 0
N_FACTORS = None
N_FACTOR_WINDOW = 5 # trailing window of alpha factors exported to before_trading_start

class Volatility(CustomFactor):  
        inputs = [USEquityPricing.close]  
        window_length = 20  
        def compute(self, today, assets, out, close):  
            # [0:-1] is needed to remove last close since diff is one element shorter  
            daily_returns = np.diff(close, axis = 0) / close[0:-1]  
            out[:] = daily_returns.std(axis = 0) * math.sqrt(252)

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

    class STA(CustomFactor):  
        inputs = [Fundamentals.operating_cash_flow,  
                  Fundamentals.net_income_continuous_operations,  
                  Fundamentals.total_assets]  
        window_length = 1
        window_safe = True
        def compute(self, today, assets, out, ocf, ni, ta):  
            ta = np.where(np.isnan(ta), 0, ta)  
            ocf = np.where(np.isnan(ocf), 0, ocf)  
            ni = np.where(np.isnan(ni), 0, ni)  
            out[:] = preprocess(abs(ni[-1] - ocf[-1])/ ta[-1])
            
    class SNOA(CustomFactor):  
        inputs = [Fundamentals.total_assets,  
                 Fundamentals.cash_and_cash_equivalents,  
                 Fundamentals.current_debt, # same as short-term debt?  
                 Fundamentals.minority_interest_balance_sheet,  
                 Fundamentals.long_term_debt, # check same?  
                 Fundamentals.preferred_stock] # check same?  
        window_length = 1
        window_safe = True
        def compute(self, today, assets, out, ta, cace, cd, mi, ltd, ps):  
            ta = np.where(np.isnan(ta), 0, ta)  
            cace = np.where(np.isnan(cace), 0, cace)  
            cd = np.where(np.isnan(cd), 0, cd)  
            mi = np.where(np.isnan(mi), 0, mi)  
            ltd = np.where(np.isnan(ltd), 0, ltd)  
            ps = np.where(np.isnan(ps), 0, ps)  
            results = ((ta[-1]-cace[-1])-(ta[-1]-cace[-1]-ltd[-1]-cd[-1]-ps[-1]-mi[-1]))/ta[-1]  
            out[:] = preprocess(np.where(np.isnan(results),0,results))
            
    class ROA(CustomFactor):  
        inputs = [Fundamentals.roa]  
        window_length = 1
        window_safe = True
        def compute(self, today, assets, out, roa):  
            out[:] = preprocess(np.where(roa[-1]>0,1,0))
            
    class FCFTA(CustomFactor):  
        inputs = [Fundamentals.free_cash_flow,  
                 Fundamentals.total_assets]  
        window_length = 1
        window_safe = True
        def compute(self, today, assets, out, fcf, ta):  
            out[:] = preprocess(np.where(fcf[-1]/ta[-1]>0,1,0))
            
    class ROA_GROWTH(CustomFactor):  
        inputs = [Fundamentals.roa]  
        window_length = 252
        window_safe = True
        def compute(self, today, assets, out, roa):  
            out[:] = np.where(roa[-1]>roa[-252],1,0)
            
    class FCFTA_ROA(CustomFactor):  
        inputs = [Fundamentals.free_cash_flow,  
                  Fundamentals.total_assets,  
                  Fundamentals.roa]  
        window_length = 1
        window_safe = True
        def compute(self, today, assets, out, fcf, ta, roa):  
            out[:] = preprocess(np.where(fcf[-1]/ta[-1]>roa[-1],1,0))
            
    class FCFTA_GROWTH(CustomFactor):  
        inputs = [Fundamentals.free_cash_flow,  
                  Fundamentals.total_assets]  
        window_length = 252
        window_safe = True
        def compute(self, today, assets, out, fcf, ta):  
            out[:] = preprocess(np.where(fcf[-1]/ta[-1]>fcf[-252]/ta[-252],1,0))
            
    class LTD_GROWTH(CustomFactor):  
        inputs = [Fundamentals.total_assets,  
                  Fundamentals.long_term_debt]  
        window_length = 252
        window_safe = True
        def compute(self, today, assets, out, ta, ltd):  
            out[:] = preprocess(np.where(ltd[-1]/ta[-1]<ltd[-252]/ta[-252],1,0))
            
    class CR_GROWTH(CustomFactor):  
        inputs = [Fundamentals.current_ratio]  
        window_length = 252
        window_safe = True
        def compute(self, today, assets, out, cr):  
            out[:] = preprocess(np.where(cr[-1]>cr[-252],1,0))
            
    class GM_GROWTH(CustomFactor):  
        inputs = [Fundamentals.gross_margin]  
        window_length = 252
        window_safe = True
        def compute(self, today, assets, out, gm):  
            out[:] = preprocess(np.where(gm[-1]>gm[-252],1,0))
            
    class ATR_GROWTH(CustomFactor):  
        inputs = [Fundamentals.assets_turnover]  
        window_length = 252
        window_safe = True
        def compute(self, today, assets, out, atr):  
            out[:] = preprocess(np.where(atr[-1]>atr[-252],1,0))
            
    class NEQISS(CustomFactor):  
        inputs = [Fundamentals.shares_outstanding]  
        window_length = 252
        window_safe = True
        def compute(self, today, assets, out, so):  
            out[:] = preprocess(np.where(so[-1]-so[-252]<1,1,0))
            
    class GM_GROWTH_2YR(CustomFactor):  
        inputs = [Fundamentals.gross_margin]  
        window_length = 504
        window_safe = True
        def compute(self, today, assets, out, gm):  
            out[:] = preprocess(gmean([gm[-1]+1, gm[-252]+1,gm[-504]+1])-1)
            
    class GM_STABILITY_2YR(CustomFactor):  
        inputs = [Fundamentals.gross_margin]  
        window_length = 504
        window_safe = True
        def compute(self, today, assets, out, gm):  
            out[:] = preprocess(np.std([gm[-1]-gm[-252],gm[-252]-gm[-504]],axis=0)) 
            
    class ROA_GROWTH_2YR(CustomFactor):  
        inputs = [Fundamentals.roa]  
        window_length = 504
        window_safe = True
        def compute(self, today, assets, out, roa):  
            out[:] = preprocess(gmean([roa[-1]+1, roa[-252]+1,roa[-504]+1])-1)
            
    class ROIC_GROWTH_2YR(CustomFactor):  
        inputs = [Fundamentals.roic]  
        window_length = 504
        window_safe = True
        def compute(self, today, assets, out, roic):  
            out[:] = preprocess(gmean([roic[-1]+1, roic[-252]+1,roic[-504]+1])-1)
            
    class GM_GROWTH_8YR(CustomFactor):  
        inputs = [Fundamentals.gross_margin]  
        window_length = 8
        window_safe = True
        def compute(self, today, assets, out, gm):  
            out[:] = preprocess(gmean([gm[-1]+1, gm[-2]+1, gm[-3]+1, gm[-4]+1, gm[-5]+1, gm[-6]+1, gm[-7]+1, gm[-8]+1])-1)        
            
    class GM_STABILITY_8YR(CustomFactor):  
        inputs = [Fundamentals.gross_margin]  
        window_length = 9
        window_safe = True
        def compute(self, today, assets, out, gm):  
            out[:] = preprocess(gm[-8]) 
            
    class ROA_GROWTH_8YR(CustomFactor):  
        inputs = [Fundamentals.roa]  
        window_length = 9
        window_safe = True
        def compute(self, today, assets, out, roa):  
            out[:] = preprocess(gmean([roa[-1]/100+1, roa[-2]/100+1,roa[-3]/100+1,roa[-4]/100+1,roa[-5]/100+1,roa[-6]/100+1,roa[-7]/100+1,roa[-8]/100+1])-1) 
            
    class ROIC_GROWTH_8YR(CustomFactor):  
        inputs = [Fundamentals.roic]  
        window_length = 9
        window_safe = True
        def compute(self, today, assets, out, roic):  
            out[:] = preprocess(gmean([roic[-1]/100+1, roic[-2]/100+1,roic[-3]/100+1,roic[-4]/100+1,roic[-5]/100+1,roic[-6]/100+1,roic[-7]/100+1,roic[-8]/100+1])-1)
            
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
            
            # fcf_ev,
            # DebtToTotalAssets,
            # TEM,
            # Piotroski,
            # Altman_Z,
            # Quick_Ratio,
        
            # AdvancedMomentum,
            # STA,
            # SNOA, 
            # ROA,  
            # FCFTA,  
            # ROA_GROWTH,  
            # FCFTA_ROA,
            # FCFTA_GROWTH, 
            # LTD_GROWTH,  
            # CR_GROWTH,  
            # GM_GROWTH,  
            # ATR_GROWTH, 
            # NEQISS,
            # GM_GROWTH_2YR,  
            # GM_STABILITY_2YR, 
            # ROA_GROWTH_2YR,  
            # ROIC_GROWTH_2YR,  
            # GM_STABILITY_8YR,  
            # ROA_GROWTH_8YR,  
            # ROIC_GROWTH_8YR,  
        ]
    
    return factors

class Factor_N_Days_Ago(CustomFactor):
    
    def compute(self, today, assets, out, input_factor):
        out[:] = input_factor[0]

def factor_pipeline():
    
    factors = make_factors()
    
    pipeline_columns = {}
    for k,f in enumerate(factors):
        for days_ago in range(N_FACTOR_WINDOW):
            pipeline_columns['alpha_'+str(k)+'_'+str(days_ago)] = Factor_N_Days_Ago([f(mask=QTradableStocksUS())], window_length=days_ago+1, mask=QTradableStocksUS())
    
    pipe = Pipeline(columns = pipeline_columns,
    screen = QTradableStocksUS())
    
    return pipe

def beta_pipeline():
    
    beta = SimpleBeta(target=sid(8554),
                      regression_length=126)
                      # allowed_missing_percentage=1.0
                     # )
    
    pipe = Pipeline(columns = {'mkt_beta': beta},
    screen = QTradableStocksUS())
    return pipe
    
def volatility_pipeline():
    
    volatility = Volatility(mask = QTradableStocksUS())   
  
    pipe = Pipeline(columns = {'volatility': volatility},
    screen = QTradableStocksUS())

    return pipe


def initialize(context):
    context.max_leverage = 1.005
    context.min_leverage = 0.995
    context.max_pos_size = 0.015
    context.max_turnover = 0.95
    context.max_beta = 0.005
    context.max_net_exposure = 0.005
    context.max_sector_exposure = 0.18
    context.max_style_exposure = 0.38
    context.target_mkt_beta = 0
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
    
    context.beta_calc_days = 126
    context.init = True
    
    attach_pipeline(risk_loading_pipeline(), 'risk_loading_pipeline')
    attach_pipeline(beta_pipeline(), 'beta_pipeline')
    attach_pipeline(factor_pipeline(), 'factor_pipeline')
    attach_pipeline(volatility_pipeline(), 'volatility_pipeline')
    
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
    
   
def recording_statements(context, data):
 
    record(num_positions=len(context.portfolio.positions))
    record(leverage=context.account.leverage)
    
    
def before_trading_start(context, data):
 
    risk_loadings = pipeline_output('risk_loading_pipeline')
    risk_loadings.fillna(risk_loadings.median(), inplace=True)
    context.risk_loadings = risk_loadings
    context.beta_pipeline = pipeline_output('beta_pipeline')
    context.volatility = pipeline_output('volatility_pipeline')['volatility']
    
    alphas = pipeline_output('factor_pipeline').dropna()
    
    n_factors = len(alphas.columns)/N_FACTOR_WINDOW
    n_stocks = len(alphas.index)
    
    alphas_flattened = np.zeros((n_factors,n_stocks*N_FACTOR_WINDOW))
    
    for f in range(n_factors):
        a = alphas.iloc[:,f*N_FACTOR_WINDOW:(f+1)*N_FACTOR_WINDOW].values
        alphas_flattened[f,:] = np.ravel(a)
    
    clustering = SpectralClustering(n_clusters=3,assign_labels="discretize",random_state=0).fit(alphas_flattened)
    
    weights = np.zeros(n_factors)
    for k,w in enumerate(clustering.labels_):
        weights[k] = Counter(clustering.labels_)[w]
    
    alphas_current = alphas.ix[:,::N_FACTOR_WINDOW]
    
    context.combined_alpha = pd.Series(np.zeros_like(alphas_current.iloc[:,1].values),index=alphas_current.index)
    for k in range(n_factors):
        context.combined_alpha += alphas_current.iloc[:,k]/weights[k]
        
def rebalance(context, data):
    
    alpha = context.combined_alpha
 
    # demean and normalize
    alpha = alpha - alpha.mean()
    denom = alpha.abs().sum()
    alpha = alpha/denom
    
    beta = context.beta_pipeline
    factor_loadings = context.risk_loadings
    vol = context.volatility
     
    validSecurities = list(
        set(alpha.index.values.tolist()) & 
        set(beta.index.values.tolist()) &
        set(factor_loadings.index.values.tolist()))
    
    alpha = alpha.loc[validSecurities]
    
    print "Alpha"
    print alpha
    
    factor_loadings = factor_loadings.loc[validSecurities, :]
    beta = beta.loc[validSecurities, :]   
    vol = vol.loc[validSecurities]
    #Variance = Square of volatility
    variance = vol * vol
    
    context.min_turnover = 0.0
    int_port = np.zeros(len(alpha))
    allPos = context.portfolio.positions
    currentSecurities = list(allPos.keys())
    defunctSecurities = list(set(currentSecurities) - set(validSecurities))    
    portfolio_value = context.portfolio.portfolio_value

    for (i,sec) in enumerate(validSecurities):
        if allPos[sec]:
           int_port[i] = (allPos[sec].amount*allPos[sec].last_sale_price)/portfolio_value
     
    for (i,sec) in enumerate(defunctSecurities):
        if allPos[sec]:
           context.min_turnover += allPos[sec].amount/portfolio_value
    
    context.initial_portfolio = pd.Series(int_port, index=validSecurities)

    if not alpha.empty:
        
        covariance = compute_covariance(context, data, validSecurities)
        # Running optimization implementation (testing)         
        my_opt_weights = optimize(context, alpha, factor_loadings, beta, covariance, variance)
        my_opt = pd.Series(my_opt_weights, index=validSecurities) 
        algo.order_optimal_portfolio(
            opt.TargetWeights(my_opt),
            constraints = []
        )
        
def compute_covariance(context, data, securities):
    
    #1. Get factor loadings
    #2. Get 63 days historical returns on stocks
    #3. Get Factor returns by multiplying the factor loadings with st-returns
    factor_loadings = context.risk_loadings
    factor_loadings = factor_loadings.loc[securities, :]
        
    price_history = data.history(securities, fields="price", bar_count=64, frequency="1d")
    pct_change = price_history.pct_change()
    
    for factor_name in context.styles:
        factor_loadings = factor_loadings.sort_values(by = factor_name)
        factor_loadings.loc[-50:, factor_name] = 1.0
        factor_loadings.loc[:50, factor_name] = -1.0
        
        
    factor_loadings[np.abs(factor_loadings) != 1.0] = 0        
    factor_returns = pct_change.dot(factor_loadings)
    cov = factor_returns.cov(min_periods=63)
    
    factor_list = context.sectors + context.styles
    cov = cov[factor_list]
    cov = cov.reindex(factor_list)
    
    # print "Covariance"
    # print cov
    
    return cov
    
def optimize(context, alpha, factor_loadings, beta, covar, var):
            
    nstocks = alpha.size
    ini_port = context.initial_portfolio
    
    max_wt = pd.Series(np.ones(nstocks)*context.max_pos_size, index = alpha.index.tolist())
   
    min_wt = pd.Series(-np.ones(nstocks)*context.max_pos_size, index = alpha.index.tolist())
    
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
                np.matrix(beta[['mkt_beta']].values.reshape((nstocks,))))
    
    market_beta_exp_bound = np.full((1), context.max_beta)
    
    # Create optimization variables
    w = cp.Variable(nstocks)
    
    # Systematic risk
    F = cp.vstack(
            sector_exp_matrix, 
            style_exp_matrix)
    
    f = F*w
    
    D = np.diag(var.values)
    chlskyD = linalg.cholesky(D, lower=True)
    fD = chlskyD*w
    
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
        w <= max_wt.values.reshape((nstocks,)),
        w >= min_wt.values.reshape((nstocks,)),
        cp.sum_entries(w) <= context.max_net_exposure,
        cp.sum_entries(w) >= -context.max_net_exposure,
        cp.norm(w,1) <= context.max_leverage,
        cp.norm(w-ini_port.values.reshape((nstocks,)),1) <= context.max_turnover
    ]
   
        
    #Objective Function - Maximize Alpha
    c = alpha.values*400
    
    total_alpha = c.T*w
    gamma_sys = cp.Parameter(sign = "positive")
    gamma_sys.value = 5.0 * context.normalizing_constant
    
    gamma_unsys = cp.Parameter(sign = "positive")
    gamma_unsys.value = 100.0 * context.normalizing_constant

    gamma_beta = cp.Parameter(sign = "positive")
    gamma_beta.value = 140.0 * context.normalizing_constant
    
    risk_sys = cp.quad_form(f, covar.as_matrix()) 
    risk_unsys = cp.sum_squares(fD)
    
    mkt_beta_deviation = cp.sum_squares(market_beta_exp_matrix*w - context.target_mkt_beta)
    
    beta_deviation = mkt_beta_deviation
    objective = cp.Maximize(total_alpha - gamma_sys*risk_sys - gamma_unsys*risk_unsys - gamma_beta*beta_deviation)
    # objective = cp.Maximize(total_alpha)
    
    if context.init:
        prob = cp.Problem(objective, constraints)
        try:
            prob.solve()
        
            w = np.asarray(w.value).flatten().reshape((nstocks,))
            # w[abs(w) <= 0.0001] = 0
        
            print "Total ALpha"
            print total_alpha.value
            context.init = False
            
            return w
        except Exception as e:
            print "Error solving optimization"
            print e
            context.init = False
            return ini_port
        
    turnover = np.linspace(0.15, 0.65, num=100)
    
    for max_turnover in turnover:

        # Optimization Problem Constraints (Group + Variable)
        constraints = [
            A*w <= Ub,
            A*w >= Lb,
            w <= max_wt.values.reshape((nstocks,)),
            w >= min_wt.values.reshape((nstocks,)),
            cp.sum_entries(w) <= context.max_net_exposure,
            cp.sum_entries(w) >= -context.max_net_exposure,
            cp.norm(w,1) <= context.max_leverage,
            cp.norm(w-ini_port.values.reshape((nstocks,)),1) <= max_turnover
        ]
    
        prob = cp.Problem(objective, constraints)
        try:
            prob.solve()
        
            w = np.asarray(w.value).flatten().reshape((nstocks,))
            # w[abs(w) <= 0.0001] = 0
        
            print "Total ALpha"
            print total_alpha.value
        
            return w
        except Exception as e:
            print "Updating constraints for turnover"
            # print e
            # return ini_port