from pandas import Series, DataFrame
import pandas as pd
import statsmodels
import statsmodels.api 
import datetime as dt
import datetime as datetime
import numpy as np
from brokers.ib import VWAPBestEffort

def initialize(context):
    context.benchmarkSecurity = symbol('IWM')
    schedule_function(func=regular_allocation,
                      date_rule=date_rules.week_start(),
                      time_rule=time_rules.market_open(minutes=1),
                      half_days=True
                      )
    schedule_function(bookkeeping)
    set_slippage(slippage.FixedSlippage(spread=0.00))
    set_commission(commission.PerShare(cost=0, min_trade_cost=None))

def bookkeeping(context, data):   
    short_count = 0
    long_count = 0
    for sid in context.portfolio.positions:
        if context.portfolio.positions[sid].amount > 0.0:
            long_count = long_count + 1
        if context.portfolio.positions[sid].amount < 0.0:
            short_count = short_count + 1
    record(long_count=long_count)
    record(short_count=short_count)
    # gross leverage should be 2, net leverage should be 0!
    record(leverage=context.account.leverage)
    
def handle_data(context, data):
    pass
    
def add_ebit_ev(df):
    ev = df['enterprise_value']
    ev[ev < 0.0] = 1.0
    df['enterprise_value'] = ev
    df['ebit_ev'] = df['ebit'] / df['enterprise_value']
    return df

def before_trading_start(context):         
    df = get_fundamentals(
        query(fundamentals.valuation.market_cap,
              fundamentals. valuation.shares_outstanding,
              fundamentals.income_statement.ebit,
              fundamentals.income_statement.ebit_as_of,
              fundamentals.valuation.market_cap,
              fundamentals.valuation.enterprise_value,
              fundamentals.valuation.enterprise_value_as_of,
              fundamentals.share_class_reference.symbol,
              fundamentals.company_reference.standard_name,
              fundamentals.operation_ratios.total_debt_equity_ratio
              )
        .filter(fundamentals.operation_ratios.total_debt_equity_ratio != None)
        .filter(fundamentals.valuation.market_cap != None)
        .filter(fundamentals.valuation.shares_outstanding != None)  
        .filter(fundamentals.company_reference.primary_exchange_id != "OTCPK") # no pink sheets
        .filter(fundamentals.company_reference.primary_exchange_id != "OTCBB") # no pink sheets
        .filter(fundamentals.asset_classification.morningstar_sector_code != None) # require sector
        .filter(fundamentals.share_class_reference.security_type == 'ST00000001') # common stock only
        .filter(~fundamentals.share_class_reference.symbol.contains('_WI')) # drop when-issued
        .filter(fundamentals.share_class_reference.is_primary_share == True) # remove ancillary classes
        .filter(((fundamentals.valuation.market_cap*1.0) / (fundamentals.valuation.shares_outstanding*1.0)) > 1.0)  # stock price > $1
        .filter(fundamentals.share_class_reference.is_depositary_receipt == False) # !ADR/GDR
        .filter(fundamentals.valuation.market_cap > 30000000) # cap > $30MM
        .filter(~fundamentals.company_reference.standard_name.contains(' LP')) # exclude LPs
        .filter(~fundamentals.company_reference.standard_name.contains(' L P'))
        .filter(~fundamentals.company_reference.standard_name.contains(' L.P'))
        .filter(fundamentals.balance_sheet.limited_partnership == None) # exclude LPs
        .order_by(fundamentals.valuation.market_cap.desc()) 
        .offset(0)
        .limit(2500) 
        ).T
    df = add_ebit_ev(df)
    context.longs = df.sort(['ebit_ev'],ascending=False)[0:100]
    context.shorts = df.sort(['ebit_ev'],ascending=True)[0:100]
    context.universe = np.union1d(context.longs.index.values, context.shorts.index.values)
    update_universe(context.universe)
    
def regular_allocation(context, data):
    prices = history(500,'1d','price')
    longs = context.longs.index
    shorts = context.shorts.index
    # now allocate our longs and shorts
    # first sell anything we no longer want
    desiredSids = longs.union(shorts)
    holdingSids = Series(context.portfolio.positions.keys())
    gettingTheBoot = holdingSids[holdingSids.isin(desiredSids) == False]
    for (ix,sid) in gettingTheBoot.iteritems():
        order_target_percent(sid, 0.0)
        
    # calculate naive "beta" of each portfolio
    prices_longs = prices[longs.intersection(prices.columns)]
    prices_shorts = prices[shorts.intersection(prices.columns)]
    prices_spy = prices[context.benchmarkSecurity]
    rets_long_port = prices_longs.pct_change().sum(axis=1)
    rets_short_port = prices_shorts.pct_change().sum(axis=1)
    rets_spy_port = prices_spy.pct_change()
    beta_span = 250
    benchVar = pd.stats.moments.ewmvar(rets_spy_port, span=beta_span)[beta_span:]
    long_cov = pd.stats.moments.ewmcov(rets_long_port, rets_spy_port, span=beta_span)[beta_span:]
    short_cov = pd.stats.moments.ewmcov(rets_short_port, rets_spy_port, span=beta_span)[beta_span:]
    long_beta = (long_cov / benchVar).iloc[-1]
    short_beta = (short_cov / benchVar).iloc[-1]
    beta_ratio = long_beta / short_beta
    target_lev_per_side = 2.0
    scale = target_lev_per_side / (1 + beta_ratio)
    long_each = (scale * 1.0) / len(longs)
    short_each = (scale * beta_ratio) / len(shorts)
    # now buy our longs, scaled by ex ante beta
    for sid in longs:
        if sid in data: # this is so stupid
            order_target_percent(sid, long_each)
    # sell our shorts, scaled by ex ante beta
    for sid in shorts:
        if sid in data: # stupid 
            order_target_percent(sid, -short_each)
    # our long-short portfolio might now have more gross leverage than 2.0, but 
    # should have an expected beta of 0
           
           
    