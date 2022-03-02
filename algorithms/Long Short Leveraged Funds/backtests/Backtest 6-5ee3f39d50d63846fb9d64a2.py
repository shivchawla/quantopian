"""
This is a template algorithm on Quantopian for you to adapt and fill in.
"""
import quantopian.algorithm as algo
# from quantopian.pipeline import Pipeline, CustomFactor
# from quantopian.pipeline.data.builtin import USEquityPricing
# from quantopian.pipeline.filters import QTradableStocksUS, Q500US
# from quantopian.pipeline.factors import AverageDollarVolume


def initialize(context):
    """
    Called once at the start of the algorithm.
    """
    # Rebalance every day, 1 hour after market open.
    algo.schedule_function(
        closepositions,
        algo.date_rules.every_day(),
        algo.time_rules.market_close(minutes=10),
    )
    
    algo.schedule_function(
        record_leverage,
        algo.date_rules.every_day(),
        algo.time_rules.market_close(minutes=30),
    )

    algo.schedule_function(
        openpositions,
        algo.date_rules.every_day(),
        algo.time_rules.market_open(minutes=15),
    )
    #set_asset_restrictions(security_lists.restrict_leveraged_etfs)
    set_commission(commission.PerShare(cost=0, min_trade_cost=0))
    set_slippage(slippage.FixedSlippage(spread=0))
    
    context.LONG = sid(39214)
    context.SHORT = sid(39211)
    
    context.securities = [context.LONG, context.SHORT]
    context.invested_capital = 0

    
    
#     # # Create our dynamic stock selector.
#     algo.attach_pipeline(make_pipeline(context), 'pipeline')

# class CloseToLowReturnsMean(CustomFactor):
#     inputs = [USEquityPricing.close, USEquityPricing.low]
#     window_length = 66
    
#     def compute(self, today, assets, out, close, low):  
#         # [0:-1] is needed to remove last close since diff is one element shorter  
#         # daily_returns = np.diff(close, axis = 0) / close[0:-1]
#         daily_returns = (low[1:] - close[0:-1])/close[0:-1]
#         out[:] = daily_returns.mean(axis = 0)
        

# class CloseToHighReturnsMean(CustomFactor):
#     inputs = [USEquityPricing.close, USEquityPricing.high]
#     window_length = 66
    
#     def compute(self, today, assets, out, close, high):  
#         # [0:-1] is needed to remove last close since diff is one element shorter  
#         # daily_returns = np.diff(close, axis = 0) / close[0:-1]
#         daily_returns = (high[1:] - close[0:-1])/close[0:-1]
#         out[:] = daily_returns.mean(axis = 0)
        
    
# def make_pipeline(context):
#     """
#     A function to create our dynamic stock selector (pipeline). Documentation
#     on pipeline can be found here:
#     https://www.quantopian.com/help#pipeline-title
#     """

#     # Base universe set to the QTradableStocksUS
#     dollar_volume = AverageDollarVolume(window_length=30)
#     high_dollar_volume = dollar_volume.top(context.nStocks)

#     # Factor of yesterday's close price.
#     yesterday_close = USEquityPricing.close.latest
#     cthMean = CloseToHighReturnsMean()
#     ctlMean = CloseToLowReturnsMean()
    
#     pipe = Pipeline(
#         columns={
#             'close': yesterday_close,
#             'cthMean': cthMean,
#             'ctlMean': ctlMean,
#         },
#         screen=high_dollar_volume & Q500US()
#     )
    
#     return pipe


def before_trading_start(context, data):
    """
    Called every day before market open.
    """
    context.loss = 0.01
    context.profit = 0.03

#     context.capital_each = context.account.cash/
    
#     context.output = algo.pipeline_output('pipeline')

#     # These are the securities that we are interested in trading each day.
#     context.security_list = context.output.index
    
#     context.high = {}
#     context.low = {}
#     context.already_traded = {}
#     context.invested_capital = 0


def openpositions(context, data):
    """
    Execute orders according to our schedule_function() timing.
    """
    
    for sec in context.securities:
        order_target_percent(sec, 0.5)
        #context.invested_capital = 0.05*context.portfolio.portfolio_value


def closepositions(context, data):
    """
    Execute orders according to our schedule_function() timing.
    """
    positions = context.portfolio.positions
    
    for sec in positions:
        order_target_percent(sec, 0) 
    

def record_leverage(context, data):
    """
    Execute orders according to our schedule_function() timing.
    """
    positions = context.portfolio.positions
    portfolio_val = context.portfolio.portfolio_value
    
    absPosValue = 0
    for sec in positions:
        absPosValue += abs(positions[sec].last_sale_price*positions[sec].amount)
    
    leverage = absPosValue/portfolio_val
    
    record("Leverage", leverage)
    
def record_vars(context, data):
    """
    Plot variables at the end of each day.
    """
    pass

def check_positions_for_loss_or_profit(context, data):
    
    loss = context.loss
    profit = context.profit

    # Sell our positions on longs/shorts for profit or loss
    for security in context.portfolio.positions:
        # is_stock_held = context.stocks_held.get(security) >= 0
        if data.can_trade(security) and not get_open_orders(security):
            current_position = context.portfolio.positions[security].amount  
            cost_basis = context.portfolio.positions[security].cost_basis  
            price = data.current(security, 'price')
            # On Long & Profit
            if price >= cost_basis * (1 + profit) and current_position > 0:  
                order_target_percent(security, 0)  
                # log.info( str(security) + ' Sold Long for Profit')  
                #context.invested_capital -= context.capital_each
                # del context.stocks_held[security]  
            # On Short & Profit
            if price <= cost_basis* (1 - profit)  and current_position < 0:
                order_target_percent(security, 0)  
                # log.info( str(security) + ' Sold Short for Profit') 
                #context.invested_capital -= context.capital_each
                # del context.stocks_held[security]
            # On Long & Loss
            if price <= cost_basis * (1 - loss) and current_position > 0:  
                order_target_percent(security, 0)
                context.loss = 0 #-0.005
                # log.info( str(security) + ' Sold Long for Loss') 
                #context.invested_capital -= context.capital_each
                # del context.stocks_held[security]  
            # On Short & Loss
            if price >= cost_basis * (1 + loss) and current_position < 0:  
                order_target_percent(security, 0)  
                # log.info( str(security) + ' Sold Short for Loss') 
                #context.invested_capital -= context.capital_each
                # del context.stocks_held[security]  
            


def handle_data(context, data):
    """
    Called every minute.
    """
    positions = context.portfolio.positions
    for sec in context.securities: 
        check_positions_for_loss_or_profit(context, data)
                
                
    #log.info("Invested Capital Pct: "+ str(context.invested_capital))