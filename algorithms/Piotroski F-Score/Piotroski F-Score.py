import numpy as np
import math
from pandas import Series, DataFrame
import pandas as pd

import quantopian.optimize as opt
import quantopian.algorithm as algo

from quantopian.pipeline.data.builtin import USEquityPricing
from quantopian.pipeline import CustomFactor, Pipeline
from quantopian.pipeline.factors import SimpleMovingAverage
from quantopian.algorithm import attach_pipeline, pipeline_output
from quantopian.pipeline.data import morningstar
from quantopian.pipeline.data import Fundamentals
from quantopian.pipeline.filters import QTradableStocksUS
from quantopian.pipeline.experimental import risk_loading_pipeline
from quantopian.pipeline.factors import SimpleBeta


"""
    Creating an algorithm based off the Piotroski Score index which is based off of a score (0-9)
    Each of the following points in Profitablity, Leverage & Operating Effificieny means one point. 
    We are going to select 

    Profitability 
    - Positive ROA
    - Positive Operating Cash Flow
    - Higher ROA in current year versus last year
    - Cash flow from operations > ROA of current year

    Leverage
    - Current ratio of long term debt < last year's ratio of long term debt
    - Current year's current_ratio > last year's current_ratio
    - No new shares issued this year

    Operating Efficiency
    - Higher gross margin compared to previous year
    - Higher asset turnover ratio compared to previous year

"""

class Piotroski(CustomFactor):
    inputs = [
        morningstar.operation_ratios.roa,
        morningstar.cash_flow_statement.operating_cash_flow,
        morningstar.cash_flow_statement.cash_flow_from_continuing_operating_activities,
        
        morningstar.operation_ratios.long_term_debt_equity_ratio,
        morningstar.operation_ratios.current_ratio,
        morningstar.valuation.shares_outstanding,
        
        morningstar.operation_ratios.gross_margin,
        morningstar.operation_ratios.assets_turnover,
    ]
    window_length = 22
    
    def compute(self, today, assets, out,
                roa, cash_flow, cash_flow_from_ops,
                long_term_debt_ratio, current_ratio, shares_outstanding,
                gross_margin, assets_turnover):
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
        
        out[:] = profit + leverage + operating

class ROA(CustomFactor):
    window_length = 1
    inputs = [morningstar.operation_ratios.roa]
    
    def compute(self, today, assets, out, roa):
        out[:] = (roa[-1] > 0).astype(int)
        
class ROAChange(CustomFactor):
    window_length = 22
    inputs = [morningstar.operation_ratios.roa]
    
    def compute(self, today, assets, out, roa):
        out[:] = (roa[-1] > roa[0]).astype(int)
        
class CashFlow(CustomFactor):
    window_length = 1
    inputs = [morningstar.cash_flow_statement.operating_cash_flow]
    
    def compute(self, today, assets, out, cash_flow):
        out[:] = (cash_flow[-1] > 0).astype(int)
        
class CashFlowFromOps(CustomFactor):
    window_length = 1
    inputs = [morningstar.cash_flow_statement.cash_flow_from_continuing_operating_activities, morningstar.operation_ratios.roa]
    
    def compute(self, today, assets, out, cash_flow_from_ops, roa):
        out[:] = (cash_flow_from_ops[-1] > roa[-1]).astype(int)
        
class LongTermDebtRatioChange(CustomFactor):
    window_length = 22
    inputs = [morningstar.operation_ratios.long_term_debt_equity_ratio]
    
    def compute(self, today, assets, out, long_term_debt_ratio):
        out[:] = (long_term_debt_ratio[-1] < long_term_debt_ratio[0]).astype(int)
        
class CurrentDebtRatioChange(CustomFactor):
    window_length = 22
    inputs = [morningstar.operation_ratios.current_ratio]
    
    def compute(self, today, assets, out, current_ratio):
        out[:] = (current_ratio[-1] > current_ratio[0]).astype(int)
        
class SharesOutstandingChange(CustomFactor):
    window_length = 22
    inputs = [morningstar.valuation.shares_outstanding]
    
    def compute(self, today, assets, out, shares_outstanding):
        out[:] = (shares_outstanding[-1] <= shares_outstanding[0]).astype(int)
        
class GrossMarginChange(CustomFactor):
    window_length = 22
    inputs = [morningstar.operation_ratios.gross_margin]
    
    def compute(self, today, assets, out, gross_margin):
        out[:] = (gross_margin[-1] > gross_margin[0]).astype(int)
        
class AssetsTurnoverChange(CustomFactor):
    window_length = 22
    inputs = [morningstar.operation_ratios.assets_turnover]
    
    def compute(self, today, assets, out, assets_turnover):
        out[:] = (assets_turnover[-1] > assets_turnover[0]).astype(int) 

class MarketCap(CustomFactor):
    # Pre-declare inputs and window_length
    inputs = [USEquityPricing.close, Fundamentals.shares_outstanding]
    window_length = 1

    # Compute market cap value
    def compute(self, today, assets, out, close, shares):
        out[:] = close[-1] * shares[-1]

        
def initialize(context):

    context.max_leverage = 1.0
    context.max_pos_size = 0.01
    context.max_turnover = 0.95

    pipe = Pipeline()
    # pipe = attach_pipeline(pipe, name='piotroski')
    
    base_universe =  QTradableStocksUS()

    profit = ROA() + ROAChange(mask = base_universe) + CashFlow(mask = base_universe) + CashFlowFromOps(mask = base_universe)
    leverage = LongTermDebtRatioChange(mask = base_universe) + CurrentDebtRatioChange(mask = base_universe) + SharesOutstandingChange(mask = base_universe)
    operating = GrossMarginChange(mask = base_universe) + AssetsTurnoverChange(mask = base_universe)
    piotroski = profit + leverage + operating
    
    mkt_cap = MarketCap(mask = base_universe)
    mkt_cap_top_200 = mkt_cap.top(500)
    pipe_screen = base_universe & ~mkt_cap_top_200

    pipe.add(piotroski, 'piotroski')

    pipe.set_screen(((piotroski >= 7) | (piotroski <= 3 and piotroski >=1)) & pipe_screen)

    context.is_month_end = False
    # schedule_function(set_month_end, date_rules.month_end(1)) 
    # schedule_function(trade_long, date_rules.month_end(), time_rules.market_open(minutes=30))
    # schedule_function(trade_short, date_rules.month_end(), time_rules.market_open(hours=4))
    # schedule_function(trade, date_rules.month_end(), time_rules.market_close())
    # Rebalance every day, 1 hour after market open.
    algo.schedule_function(
        rebalance,
        algo.date_rules.every_day(),
        algo.time_rules.market_close(hours=1),
    )

    algo.attach_pipeline(pipe, 'piotroski')
   
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
            'beta': beta,
        },
        screen=beta.notnull(),
    )

    algo.attach_pipeline(beta_pipe, 'beta_pipe')

def set_month_end(context, data):
    print "---- Set Month End -----"
    context.is_month_end = True
    
def before_trading_start(context, data):
    """
    Called every day before market open.
    """
    output = algo.pipeline_output('piotroski')
    output['signal_rank'] = transform(output, 'piotroski')
    
    alpha = output['signal_rank']
    alpha = alpha.sub(alpha.mean())
    context.alpha = alpha

    context.risk_factor_betas = algo.pipeline_output(
      'risk_pipe'
    ).dropna()

    context.beta_pipeline = algo.pipeline_output('beta_pipe')

def rebalance(context, data):
    
    beta_pipeline = context.beta_pipeline
    alpha = context.alpha

    if not alpha.empty:
        # Create MaximizeAlpha objective
        objective = opt.MaximizeAlpha(alpha)

        # Create position size constraint
        constrain_pos_size = opt.PositionConcentration.with_equal_bounds(
            -context.max_pos_size,
            0
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
            min_exposures={'beta': -1.05},
            max_exposures={'beta': -0.95},
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

def signalize(df):
   return ((df.rank() - 0.5)/df.count()).replace(np.nan,0.5)

def transform(df, field, multiplier=1):
    return signalize(multiplier*df[field])


def track_orders(context, data):
    '''
    Show orders when made and filled.
    Info: https://www.quantopian.com/posts/track-orders
    '''
    c = context
    if 'trac' not in c:
        c.t_options = {           # __________    O P T I O N S    __________
            'log_neg_cash': 1,    # Show cash only when negative.
            'log_cash'    : 0,    # Show cash values in logging window or not.
            'log_ids'     : 1,    # Include order id's in logging window or not.
            'log_unfilled': 1,    # When orders are unfilled. (stop & limit excluded).
        }    # Move these to initialize() for better efficiency.
        c.trac = {}
        c.t_dates  = {  # To not overwhelm the log window, start/stop dates can be entered.
            'active': 0,
            'start' : [],   # Start dates, option like ['2007-05-07', '2010-04-26']
            'stop'  : []    # Stop  dates, option like ['2008-02-13', '2010-11-15']
        }
    from pytz import timezone     # Python only does once, makes this portable.
                                  #   Move to top of algo for better efficiency.
    # If 'start' or 'stop' lists have something in them, triggers ...
    if c.t_dates['start'] or c.t_dates['stop']:
        date = str(get_datetime().date())
        if   date in c.t_dates['start']:    # See if there's a match to start
            c.t_dates['active'] = 1
        elif date in c.t_dates['stop']:     #   ... or to stop
            c.t_dates['active'] = 0
    else: c.t_dates['active'] = 1           # Set to active b/c no conditions.
    if c.t_dates['active'] == 0: return     # Skip if not active.
    def _minute():   # To preface each line with the minute of the day.
        bar_dt = get_datetime().astimezone(timezone('US/Eastern'))
        return str((bar_dt.hour * 60) + bar_dt.minute - 570).rjust(3) # (-570 = 9:31a)
    def _trac(to_log):      # So all logging comes from the same line number,
        log.info(to_log)    #   for vertical alignment in the logging window.

    for oid in c.trac.copy():               # Existing known orders
      o = get_order(oid)
      if o.dt == o.created: continue        # No chance of fill yet.
      cash = ''
      prc  = data.current(o.sid, 'price') if data.can_trade(o.sid) else c.portfolio.positions[o.sid].last_sale_price
      if (c.t_options['log_neg_cash'] and c.portfolio.cash < 0) or c.t_options['log_cash']:
        cash = 'cash {}'.format(int(c.portfolio.cash))
      if o.status == 2:                      # Canceled
        do = 'Buy' if o.amount > 0 else 'Sell' ; style = ''
        if o.stop:
          style = ' stop {}'.format(o.stop)
          if o.limit: style = ' stop {} limit {}'.format(o.stop, o.limit)
        elif o.limit: style = ' limit {}'.format(o.limit)
        _trac(' {}     Canceled {} {} {}{} at {}   {}  {}'.format(_minute(), do, o.amount,
           o.sid.symbol, style, prc, cash, o.id[-4:] if c.t_options['log_ids'] else ''))
        del c.trac[o.id]
      elif o.filled:                             # Filled at least some.
        filled = '{}'.format(o.amount)
        filled_amt = 0
        if o.status == 1:            # Complete
          if 0 < c.trac[o.id] < o.amount:
            filled   = 'all {}/{}'.format(o.filled - c.trac[o.id], o.amount)
          filled_amt = o.filled
        else:                                    # c.trac[o.id] value is previously filled total
          filled_amt = o.filled - c.trac[o.id]   # filled this time, can be 0
          c.trac[o.id] = o.filled                # save fill value for increments math
          filled = '{}/{}'.format(filled_amt, o.amount)
        if filled_amt:
          now = ' ({})'.format(c.portfolio.positions[o.sid].amount) if c.portfolio.positions[o.sid].amount else ' _'
          pnl = ''  # for the trade only
          amt = c.portfolio.positions[o.sid].amount ; style = ''
          if (amt - o.filled) * o.filled < 0:    # Profit-taking scenario including short-buyback
            cb = c.portfolio.positions[o.sid].cost_basis
            if cb:
              pnl  = -filled_amt * (prc - cb)
              sign = '+' if pnl > 0 else '-'
              pnl  = '  ({}{})'.format(sign, '%.0f' % abs(pnl))
          if o.stop:
            style = ' stop {}'.format(o.stop)
            if o.limit: style = ' stop () limit {}'.format(o.stop, o.limit)
          elif o.limit: style = ' limit {}'.format(o.limit)
          if o.filled == o.amount: del c.trac[o.id]
          _trac(' {}      {} {} {}{} at {}{}{}'.format(_minute(),
            'Bot' if o.amount > 0 else 'Sold', filled, o.sid.symbol, now,
            '%.2f' % prc, pnl, style).ljust(52) + '  {}  {}'.format(cash, o.id[-4:] if c.t_options['log_ids'] else ''))
      elif c.t_options['log_unfilled'] and not (o.stop or o.limit):
        _trac(' {}         {} {}{} unfilled  {}'.format(_minute(), o.sid.symbol, o.amount,
         ' limit' if o.limit else '', o.id[-4:] if c.t_options['log_ids'] else ''))

    oo = get_open_orders().values()
    if not oo: return                       # Handle new orders
    cash = ''
    if (c.t_options['log_neg_cash'] and c.portfolio.cash < 0) or c.t_options['log_cash']:
      cash = 'cash {}'.format(int(c.portfolio.cash))
    for oo_list in oo:
      for o in oo_list:
        if o.id in c.trac: continue         # Only new orders beyond this point
        prc = data.current(o.sid, 'price') if data.can_trade(o.sid) else c.portfolio.positions[o.sid].last_sale_price
        c.trac[o.id] = 0 ; style = ''
        now  = ' ({})'.format(c.portfolio.positions[o.sid].amount) if c.portfolio.positions[o.sid].amount else ' _'
        if o.stop:
          style = ' stop {}'.format(o.stop)
          if o.limit: style = ' stop {} limit {}'.format(o.stop, o.limit)
        elif o.limit: style = ' limit {}'.format(o.limit)
        _trac(' {}   {} {} {}{} at {}{}'.format(_minute(), 'Buy' if o.amount > 0 else 'Sell',
          o.amount, o.sid.symbol, now, '%.2f' % prc, style).ljust(52) + '  {}  {}'.format(cash, o.id[-4:] if c.t_options['log_ids'] else ''))


            
# Will be called on every trade event for the securities you specify. 
def handle_data(context, data):
    track_orders(context, data)