import numpy as np
import math
import pandas as pd
import quantopian.optimize as opt
import quantopian.algorithm as algo
from quantopian.pipeline.experimental import risk_loading_pipeline
from quantopian.pipeline import CustomFactor, Pipeline
from quantopian.pipeline.data import Fundamentals
from quantopian.pipeline.data.builtin import USEquityPricing
from quantopian.pipeline.filters import QTradableStocksUS, StaticAssets
from quantopian.pipeline.factors import SimpleBeta, Returns, PercentChange
from quantopian.pipeline.data.psychsignal import stocktwits
from scipy import stats, linalg
from quantopian.pipeline.factors import AverageDollarVolume
import cvxpy as cp
from scipy import sparse as scipysp
from quantopian.pipeline.data.factset.estimates import PeriodicConsensus
import quantopian.pipeline.data.factset.estimates as fe
from quantopian.pipeline.data import factset
from sklearn import preprocessing
from scipy.stats.mstats import winsorize
from sklearn.decomposition import PCA


WIN_LIMIT = 0
QL = 80

def preprocess(a):
    
    a = a.astype(np.float64)
    a[np.isinf(a)] = np.nan
    a = np.nan_to_num(a - np.nanmean(a))
    a = winsorize(a, limits=[WIN_LIMIT,WIN_LIMIT])
    
    return preprocessing.scale(a)

def signalize(df):
   # return ((df.rank() - 0.5)/df.count()).replace(np.nan,0.5)
    z = (df.rank() - 0.5)/df.count()
    return z.replace(np.nan, z.mean())
  
def initialize(context):
    set_slippage(slippage.FixedSlippage(spread=0.0))
    set_commission(commission.PerShare(cost=0, min_trade_cost=0))

    """
    Called once at the start of the algorithm.
    """
    context.max_leverage = 1.001
    context.min_leverage = 0.999
    context.max_pos_size = 0.4
    context.max_turnover = 0.95
    context.max_beta = 0.05
    context.max_net_exposure = 0.001
    context.max_volatility = 0.05
    context.max_sector_exposure = 0.02
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
    
    context.etfs = symbols('VAW', 'VCR', 'VFH', 'VNQ', 'VDC', 'VHT', 'VPU', 'VOX', 'VDE', 'VIS', 'VGT')

    context.beta_calc_days = 126
    context.init = True
    
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
    
    # Create our dynamic stock selector.
    algo.attach_pipeline(make_pipeline(), 'pipeline')   
    algo.attach_pipeline(make_pipeline_sector(context), 'pipeline_sector')   
    
    algo.attach_pipeline(
        risk_loading_pipeline(),
        'risk_pipe'
    )
    
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
        
class Sector(CustomFactor):  
    inputs = [Fundamentals.morningstar_sector_code]  
    window_length = 1 
    def compute(self, today, assets, out, code):  
        out[:] = code[-1]
            
class Factor3(CustomFactor):
    # Pre-declare inputs and window_length
    inputs = [Fundamentals.enterprise_value, Fundamentals.free_cash_flow, USEquityPricing.close, Fundamentals.shares_outstanding, Fundamentals.total_assets] 
    window_length = 2
    # Compute factor3 value
    def compute(self, today, assets, out, ev, var, close, shares, ta):
        out[:] = var[-2]/(ev[-2]*close[-2]*shares[-2]*ta[-2])**(1./3.)

        
class Factor5(CustomFactor):
    """
    TEM = standard deviation of past 6 quarters' reports
    """
    inputs = [Fundamentals.capital_expenditure, Fundamentals.enterprise_value] 
    window_length = 390
    # Compute factor5 value
    def compute(self, today, assets, out, capex, ev):
        values = capex/ev
        out[:] = -values.std(axis = 0)

class Factor9(CustomFactor):
        inputs = [USEquityPricing.high, USEquityPricing.low, USEquityPricing.close, stocktwits.bull_scored_messages, stocktwits.bear_scored_messages, stocktwits.total_scanned_messages]
        window_length = 21
        window_safe = True
        def compute(self, today, assets, out, high, low, close, bull, bear, total):
            v = np.nansum((high-low)/close, axis=0)
            out[:] = -v*np.nansum(total*(bear-bull), axis=0)
        
            
class Var_growth_mean(CustomFactor):    
    window_length = QL*9
    window_safe = True
    
    def compute(self, today, assets, out, var):  
        arr = [var[-1],
            var[-QL],
            var[-QL*2],
            var[-QL*3],
            var[-QL*4],
            var[-QL*5],
            var[-QL*6],
           var[-QL*7],
         var[-QL*8]]
        
        arr = np.diff(arr, axis=0)/arr[1:]
        mean = np.nanmean(arr, axis=0)
        out[:] = preprocess(mean)
 

class Var_to_TA_mean(CustomFactor):  
    window_length = QL*9
    window_safe = True
    
    def compute(self, today, assets, out, v, ta):  
        var  = v/ta
        arr = [var[-1],
            var[-QL],
            var[-QL*2],
            var[-QL*3],
            var[-QL*4],
            var[-QL*5],
            var[-QL*6],
           var[-QL*7],
         var[-QL*8]]
        
        mean = np.nanmean(arr, axis=0)
        out[:] = preprocess(mean)
        
class Var_to_TA_stability(CustomFactor):  
    window_length = QL*9
    window_safe = True
    
    def compute(self, today, assets, out, v, ta):  
        var  = v/ta
        arr = [var[-1],
            var[-QL],
            var[-QL*2],
            var[-QL*3],
            var[-QL*4],
            var[-QL*5],
            var[-QL*6],
           var[-QL*7],
         var[-QL*8]]
        
        mean = np.nanmean(arr, axis=0)
        std = np.nanstd(arr, axis=0)
        out[:] = preprocess(mean/std)
        
class Var_to_TL_mean(CustomFactor):  
    window_length = QL*9
    window_safe = True
    
    def compute(self, today, assets, out, v, ta):  
        var  = v/ta
        arr = [var[-1],
            var[-QL],
            var[-QL*2],
            var[-QL*3],
            var[-QL*4],
            var[-QL*5],
            var[-QL*6],
           var[-QL*7],
         var[-QL*8]]
        
        mean = np.nanmean(arr, axis=0)
        out[:] = preprocess(mean)
        
        
class Var_to_TL_stability(CustomFactor):  
    window_length = QL*9
    window_safe = True
    
    def compute(self, today, assets, out, v, ta):  
        var  = v/ta
        arr = [var[-1],
            var[-QL],
            var[-QL*2],
            var[-QL*3],
            var[-QL*4],
            var[-QL*5],
            var[-QL*6],
           var[-QL*7],
         var[-QL*8]]
        
        mean = np.nanmean(arr, axis=0)
        std = np.nanstd(arr, axis=0)
        out[:] = preprocess(mean/std)
        
class Var_to_TC_stability(CustomFactor):  
    window_length = QL*9
    window_safe = True
    
    def compute(self, today, assets, out, v, ta):  
        var  = v/ta
        arr = [var[-1],
            var[-QL],
            var[-QL*2],
            var[-QL*3],
            var[-QL*4],
            var[-QL*5],
            var[-QL*6],
           var[-QL*7],
         var[-QL*8]]
        
        mean = np.nanmean(arr, axis=0)
        std = np.nanstd(arr, axis=0)
        out[:] = preprocess(mean/std)
        
class Var_to_EV_mean(CustomFactor):  
    window_length = QL*9
    window_safe = True
    
    def compute(self, today, assets, out, v, ta):  
        var  = v/ta
        arr = [var[-1],
            var[-QL],
            var[-QL*2],
            var[-QL*3],
            var[-QL*4],
            var[-QL*5],
            var[-QL*6],
           var[-QL*7],
         var[-QL*8]]
        
        mean = np.nanmean(arr, axis=0)
        out[:] = preprocess(mean)

class Var_to_EV_stability(CustomFactor):  
    window_length = QL*9
    window_safe = True
    
    def compute(self, today, assets, out, v, ta):  
        var  = v/ta
        arr = [var[-1],
            var[-QL],
            var[-QL*2],
            var[-QL*3],
            var[-QL*4],
            var[-QL*5],
            var[-QL*6],
           var[-QL*7],
         var[-QL*8]]
        
        mean = np.nanmean(arr, axis=0)
        std = np.nanstd(arr, axis=0)
        out[:] = preprocess(mean/std)        
        
        
class Var_to_TR_mean(CustomFactor):  
    window_length = QL*9
    window_safe = True
    
    def compute(self, today, assets, out, v, ta):  
        var  = v/ta
        arr = [var[-1],
            var[-QL],
            var[-QL*2],
            var[-QL*3],
            var[-QL*4],
            var[-QL*5],
            var[-QL*6],
           var[-QL*7],
         var[-QL*8]]
        
        mean = np.nanmean(arr, axis=0)
        out[:] = preprocess(mean)

class Var_to_MCap_mean(CustomFactor):  
    window_length = QL*9
    window_safe = True
    
    def compute(self, today, assets, out, v, close, shares):  
        var  = v/(close*shares)
        arr = [var[-1],
            var[-QL],
            var[-QL*2],
            var[-QL*3],
            var[-QL*4],
            var[-QL*5],
            var[-QL*6],
           var[-QL*7],
         var[-QL*8]]
        
        mean = np.nanmean(arr, axis=0)
        out[:] = preprocess(mean)

        
class Var_to_TR_stability(CustomFactor):  
    window_length = QL*9
    window_safe = True
    
    def compute(self, today, assets, out, v, ta):  
        var  = v/ta
        arr = [var[-1],
            var[-QL],
            var[-QL*2],
            var[-QL*3],
            var[-QL*4],
            var[-QL*5],
            var[-QL*6],
           var[-QL*7],
         var[-QL*8]]
        
        mean = np.nanmean(arr, axis=0)
        std = np.nanstd(arr, axis=0)
        out[:] = preprocess(mean/std)

        
def make_pipeline():
    """
    A function to create our dynamic stock selector (pipeline). Documentation
    on pipeline can be found here:
    https://www.quantopian.com/help#pipeline-title
    """
    # Base universe set to the QTradableStocksUS
    base_universe = QTradableStocksUS()

    mkt_cap = MarketCap(mask = base_universe)
    vol = Volatility(mask = base_universe)
    adv = AverageDollarVolume(mask = base_universe, window_length = 22)
    sector = Sector(mask = base_universe)

    f181 = Var_growth_mean(inputs = [Fundamentals.total_expenses], mask = base_universe)
    
    f144 =  Var_to_TA_mean(inputs=[Fundamentals.purchase_of_intangibles, Fundamentals.total_assets], mask = base_universe) 

    f95 =  Var_to_TA_stability(inputs=[Fundamentals.cost_of_revenue, Fundamentals.total_assets], mask = base_universe) 
    
    f86 =  Var_to_TL_mean(inputs=[Fundamentals.dps_growth , Fundamentals.total_debt], mask = base_universe) 
    
    f107 = Var_to_TL_stability(inputs=[Fundamentals.classesof_cash_payments, Fundamentals.total_debt], mask = base_universe) 

    f157 = Var_to_TL_stability(inputs=[Fundamentals.increase_decrease_in_deposit, Fundamentals.total_debt], mask = base_universe) 
    
    f49 =  Var_to_TC_stability(inputs=[Fundamentals.gain_loss_on_sale_of_business, Fundamentals.total_capitalization], mask = base_universe) 
    
    f110 =  Var_to_EV_mean(inputs=[Fundamentals.capital_expenditure, Fundamentals.enterprise_value], mask = base_universe) 
    
    f311 =  Var_to_EV_stability(inputs=[Fundamentals.gain_loss_on_sale_of_business, Fundamentals.enterprise_value], mask = base_universe) 
    
    f814 =  Var_to_TR_mean(inputs=[Fundamentals.capital_expenditure, Fundamentals.total_revenue], mask = base_universe) 

    f1314 = Var_to_TR_mean(inputs=[Fundamentals.earningsfrom_equity_interest_net_of_tax, Fundamentals.total_revenue], mask = base_universe) 
    
    f518 =  Var_to_MCap_mean(inputs=[Fundamentals.non_interest_expense, USEquityPricing.close, Fundamentals.shares_outstanding], mask = base_universe) 
    
    f1315 =  Var_to_TR_stability(inputs=[Fundamentals.gross_profit, Fundamentals.total_revenue], mask = base_universe) 

    #Factors from Sector Alpha - 1
    f3 = Factor3(mask = base_universe)
    f5 = Factor5(mask = base_universe)
    f9 = Factor9(mask = base_universe)
    
    # Filter stocks out with Mcap < mcap of 100th stock in S&P500
    pipe_screen = base_universe 

    pipe_columns = {
        'mkt_cap': mkt_cap,
        'vol': vol,
        'adv':adv,
        'sector': sector,
        
        'f3':f3,
        'f5':f5,
        'f9':f9,
        
        'f181':f181,
        'f144':f144,
        'f95': f95,
        'f86': f86,
        'f107': f107,
        'f157': f157,
        
         'f110': f110, 
         'f49': f49,
         'f311': f311, 
         'f1314': f1314, 
         'f814': f814, 
         'f518': f518, 
         'f1315': f1315
        
    }

    return Pipeline(columns=pipe_columns, screen=pipe_screen)



def make_pipeline_sector(context):
    """
    A function to create our dynamic stock selector (pipeline). Documentation
    on pipeline can be found here:
    https://www.quantopian.com/help#pipeline-title
    """
    # Base universe set to the QTradableStocksUS
    base_universe = StaticAssets(context.etfs)
    vol = Volatility(mask = base_universe)

    pipe_screen = base_universe 

    pipe_columns = {'vol': vol}
    
    return Pipeline(columns=pipe_columns, screen=pipe_screen) 



def transform(df, field, multiplier=1):
    return signalize(multiplier*df[field])

def weightedMean(df, weights):
    return df.mean()

def before_trading_start(context, data):
    """
    Called every day before market open.
    """
    output = algo.pipeline_output('pipeline')
    output_sector = algo.pipeline_output('pipeline_sector')
    
    output.index.name = "esid" 
    
    mkt_cap = output['mkt_cap']
    vol = output['vol']
    adv = output['adv']
    
    allnames = []
    # factors = [181,144,95, 86, 107, 157]
    factors = [3,5,9,181,144,95, 86, 107, 157, 110, 49, 311, 1314, 814, 518, 1315]

    #Compute Weighted Factor value (within sector)
    for x in factors:
        wfname = 'weightedFactor' + str(x)
        tfname = 'f'+str(x)
        output['flag'] = 0
        output.loc[output[tfname].notnull(), 'flag'] = 1
        output[wfname] = output[tfname]*output['mkt_cap']*output['flag']
        output['mkt_cap_'+tfname] = output['mkt_cap'] #* output['flag']
        allnames.append(wfname)
        allnames.append('mkt_cap_'+tfname)

    #Compute Sectoral Market Cap
    agg_sector_ = output.groupby(['sector'])[allnames].sum().reset_index()
    
    # print(agg_sector_)
    
    for x in factors:
        wfname = 'weightedFactor' + str(x)
        tfname = 'f'+str(x)
        agg_sector_[wfname] = agg_sector_[wfname]/agg_sector_['mkt_cap_'+tfname]
        agg_sector_.drop(['mkt_cap_'+tfname], axis=1, inplace=True)
    
    # agg_sector_.rename(columns = {'mkt_cap':'sector_mkt_cap'}, inplace=True)
    # agg_sector_.drop(['sector_mkt_cap'], axis=1, inplace=True)
    agg_sector_['sector'] = agg_sector_['sector'].astype(str)
    # print agg_sector_.head()
    
    sector_codes= ['101.0','102.0','103.0','104.0','205.0','206.0','207.0','308.0','309.0','310.0','311.0']
    
    # symbols('XLB','XLY','XLF','IYR','XLP','XLV','XLU','XLE','XLI','XLK')

    sector_code_sid = pd.DataFrame({
        'sector': sector_codes,
            'sid': context.etfs})
    
    # print sector_code_sid
    df = pd.merge(agg_sector_, sector_code_sid, on='sector',     how='outer').dropna()  
    df.set_index(['sid'], inplace=True)
    df.index.name = None

    # output.reset_index(inplace=True)

    # output.drop(allnames, axis=1, inplace=True)
    
    # output['sector'] = output['sector'].astype(str)
    # output = pd.merge(output, df, on='sector', how='outer')
    
    # output = output.groupby(["sector"]).apply(lambda x: x.sort_values(["mkt_cap"], ascending = False)).reset_index(drop=True)
    # select top N rows within each continent
    # h = output.groupby('sector').head(50)
    # output = output.groupby('sector').tail(100)
    
    # fd = h.append(dt, ignore_index = True) 

    # output = output.sort_values(['mkt_cap'], ascending=[1])
    
    # output = output[output['mkt_cap'] > 1000000000]
    # output.set_index(['esid'], inplace=True)
    
    # print(output.head(100))

    for sym in context.etfs:
        if not data.can_trade(sym):
            df = df.drop([sym])

    df['alpha3'] = signalize(df['weightedFactor3']) 
    df['alpha5'] = signalize(df['weightedFactor5']) 
    df['alpha9'] = signalize(df['weightedFactor9']) 
  
    df['alpha181'] = signalize(df['weightedFactor181']) 
    df['alpha144'] = signalize(df['weightedFactor144'])
    
    df['alpha95'] = signalize(df['weightedFactor95']) 
    df['alpha86'] = signalize(df['weightedFactor86'])

    df['alpha107'] = signalize(df['weightedFactor107'])
    df['alpha157'] = signalize(df['weightedFactor157'])
    
    df['alpha110'] = signalize(df['weightedFactor110']) 
    df['alpha49'] = signalize(df['weightedFactor49'])
    
    df['alpha311'] = signalize(df['weightedFactor311']) 
    df['alpha1314'] = signalize(df['weightedFactor1314'])

    df['alpha814'] = signalize(df['weightedFactor814'])
    df['alpha518'] = signalize(df['weightedFactor518'])
    df['alpha1315'] = signalize(df['weightedFactor1315'])
        
    df['alpha_vol'] = signalize(-output_sector['vol'])
    
    # print(output_sector['vol'])

    # for name in factor_names:
    #     df[name] = df[name].sub(df[name].mean())
    
    net_alpha1 = df['alpha110'] + df['alpha49'] + df['alpha311'] - df['alpha1314'] + df['alpha814'] + df['alpha518'] + df['alpha1315'] + df['alpha86'] + df['alpha144'] + df['alpha95'] - df['alpha157'] - df['alpha107'] + df['alpha181'] 
    
    net_alpha2 =  df['alpha3'] + df['alpha5'] + df['alpha9']
    
    # net_alpha3 =  df['alpha_vol']
    
    # output['alpha3'] = signalize(output['weightedFactor3']) 
    # output['alpha5'] = signalize(output['weightedFactor5']) 
    # output['alpha9'] = signalize(output['weightedFactor9']) 
  
    # output['alpha181'] = signalize(output['weightedFactor181']) 
    # output['alpha144'] = signalize(output['weightedFactor144'])
    
    # output['alpha95'] = signalize(output['weightedFactor95']) 
    # output['alpha86'] = signalize(output['weightedFactor86'])

    # output['alpha107'] = signalize(output['weightedFactor107'])
    # output['alpha157'] = signalize(output['weightedFactor157'])
    
    # output['alpha110'] = signalize(output['weightedFactor110']) 
    # output['alpha49'] = signalize(output['weightedFactor49'])
    
    # output['alpha311'] = signalize(output['weightedFactor311']) 
    # output['alpha1314'] = signalize(output['weightedFactor1314'])

    # output['alpha814'] = signalize(output['weightedFactor814'])
    # output['alpha518'] = signalize(output['weightedFactor518'])
    # output['alpha1315'] = signalize(output['weightedFactor1315'])
    
    # net_alpha2 = output['alpha3'] + output['alpha5'] + output['alpha9']
    
    # net_alpha1 = output['alpha110'] + output['alpha49'] + output['alpha311'] - output['alpha1314'] + output['alpha814'] + output['alpha518'] + output['alpha1315'] + output['alpha86'] + output['alpha144'] + output['alpha95'] - output['alpha157'] - output['alpha107'] + output['alpha181']
    
    # context.alpha = net_alpha1 
    # print(net_alpha1)
    context.alpha = net_alpha1 + net_alpha2 
    
    output['min_weights'] = -context.max_pos_size
    output['max_weights'] = context.max_pos_size
    
    context.min_weights = output['min_weights']
    context.max_weights = output['max_weights']
    
    context.risk_factor_betas = algo.pipeline_output(
      'risk_pipe'
    ).dropna()
    
    context.beta_pipeline = algo.pipeline_output('beta_pipe')
    context.volatility = vol
    context.adv = adv
    

def rebalance(context, data):
    
    alpha = context.alpha
                       
    if not alpha.empty:
        
        alpha = alpha.sub(alpha.mean())
        try_opt = alpha/alpha.abs().sum()
        
        # Running optimization implementation (testing)         
        # my_opt_weights = optimize(context, alpha, factor_loadings, beta, cov_style, cov_sector, variance, adv_wt, try_opt)
        # my_opt = pd.Series(my_opt_weights, index=validSecurities) 
        # algo.order_optimal_portfolio(
        #     opt.TargetWeights(try_opt),
        #     constraints = [])
                
        min_weights = context.min_weights.copy(deep = True)
        max_weights = context.max_weights.copy(deep = True)
        
        constrain_pos_size = opt.PositionConcentration(
            min_weights,
            max_weights
        )

        max_leverage = opt.MaxGrossExposure(context.max_leverage)

        # Ensure long and short books
        # are roughly the same size
        dollar_neutral = opt.DollarNeutral(context.max_net_exposure)
        
        # Constrain portfolio turnover
        max_turnover = opt.MaxTurnover(context.max_turnover)
        
        algo.order_optimal_portfolio(
            opt.TargetWeights(try_opt.dropna()),
            constraints = [
                # constrain_pos_size,
                max_leverage,
                dollar_neutral,
            ])
       
        

def record_vars(context, data):
    """
    Plot variables at the end of each day.
    """
    record("Num Positions", len(context.portfolio.positions))
    record("Leverage", context.account.leverage)
    pass

def handle_data(context, data):
    """
    Called every minute.
    """
    pass