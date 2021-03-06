from quantopian.pipeline.data.builtin import USEquityPricing
import statsmodels.api as sm
import numpy as np
 
#---Global Parameters---:    
Enter = 2.0
N_days = 30
Exit = 0.2
n_days = 1
Lag = 1

def initialize(context):
    
    set_commission(commission.PerShare(cost=0, min_trade_cost=0))
    set_slippage(slippage.FixedSlippage(spread=0))
    
    #---45 pairs---#
    # context.stock_list = [ symbol('BAC') , symbol('TSLA') , symbol('AAPL')  , symbol('SBUX') , symbol('YHOO') , symbol('GLD') , symbol('SLV') , symbol('USO') , symbol('UNG') , symbol('DBC') ]
  
    context.stock_list = [sid(41959), sid(19658)] 

    context.stock_pairs = np.array([])
    context.stock_pairs_symbols = np.array([])
    for i in range(len(context.stock_list)):
        for j in range(len(context.stock_list)):
            if i<j :
                context.stock_pairs = np.append(context.stock_pairs,[context.stock_list[i],context.stock_list[j]])
                context.stock_pairs_symbols = np.append(context.stock_pairs_symbols,[context.stock_list[i].symbol,context.stock_list[j].symbol])         
                
    context.stock_pairs = context.stock_pairs.reshape(context.stock_pairs.size/2,2)
    context.stock_pairs_symbols = context.stock_pairs_symbols.reshape(context.stock_pairs.size/2,2)
    
    context.high = [False]*context.stock_pairs[:,0].size
    context.low = [False]*context.stock_pairs[:,0].size
    
    context.hedge = np.ndarray((context.stock_pairs[:,0].size,1))
    context.spread = np.ndarray((context.stock_pairs[:,0].size,1))
            
    print('My stock pairs list : \n%s '%context.stock_pairs_symbols)
    
    schedule_function(my_rebalance, date_rules.every_day(), time_rules.market_open())
    schedule_function(my_record_vars, date_rules.every_day(), time_rules.market_close())
         
def my_rebalance(context,data):
    
    hedge_0 = np.zeros( context.stock_pairs[:,0].size )
    spread_0 = np.zeros( context.stock_pairs[:,0].size )
    context.rel_price = np.zeros( context.stock_pairs[:,0].size )
    
    l = float(context.stock_pairs[:,0].size)
    
    for i in range(context.stock_pairs[:,0].size) :
        
        stock1,stock2 = context.stock_pairs[i,:]
        context.stock_X = data.history(stock1, 'price' , N_days , '1d' )
        context.stock_Y = data.history(stock2, 'price' , N_days , '1d' )

        hedge_0[i] = sm.OLS(context.stock_Y,context.stock_X).fit().params
        # spread_0[i] = context.stock_Y[-1] - hedge_0[i]*context.stock_X[-1] 
        spread_0[i] = context.stock_Y[-1] - context.stock_X[-1] 

         
#---Create the hedged spread---#       
    context.hedge = np.hstack([context.hedge,hedge_0.reshape(context.stock_pairs[:,0].size,1)])[:,-N_days:]   
    context.spread = np.hstack([context.spread,spread_0.reshape(context.stock_pairs[:,0].size,1)])[:,-N_days:]
    
    if context.hedge[0,:].size < Lag:
        return
    
    for i in range(context.stock_pairs[:,0].size) :
        
        stock1,stock2 = context.stock_pairs[i,:]
        
        context.mean_short = np.mean(context.spread[i,-n_days:])
        context.mean_long = np.mean(context.spread[i,-N_days:])
        context.std_short = np.std(context.spread[i,-n_days:])
        context.std_long = np.std(context.spread[i,-N_days:])
        
        if context.std_long != 0 :
            context.rel_price[i] = ( context.mean_short - context.mean_long )/context.std_long
        else :
            context.rel_price[i] = 0
        
        if context.spread[i,:].size < N_days:
            return
        
        hedge = context.hedge[i,-Lag]
        
        if context.rel_price[i] > Enter and context.high[i] == False and context.low[i] == False and all(data.can_trade([stock1,stock2])):
            order_target_percent(stock1, 1.0/l )
            order_target_percent(stock2, -1.0/l)
            context.low[i] = False
            context.high[i] = True
            print('Go Short  : %d-th pair ; %f @ %s and %f @ %s   '%((i+1),hedge/l,stock1.symbol,-1.0/l,stock2.symbol))
            
        if context.rel_price[i] < -Enter and context.low[i] == False and context.high[i] == False and all(data.can_trade([stock1,stock2])):
            order_target_percent(stock1, -1.0/l)
            order_target_percent(stock2, 1.0/l)
            context.low[i] = True
            context.high[i] = False
            print('Go Long : %d-th pair ; %f @ %s and %f @ %s   '%((i+1),-hedge/l,stock1.symbol,1.0/l,stock2.symbol))
            
        if context.rel_price[i] > -Exit and context.low[i] == True and context.high[i] == False and all(data.can_trade([stock1,stock2])):
            order_target_percent(stock1, 0)
            order_target_percent(stock2, 0)
            context.low[i] = False
            context.high[i] = False
            print('Ex Long : %d-th pair ; %f @ %s and %f @ %s   '%((i+1),0,stock1.symbol,0,stock2.symbol))
        
        if context.rel_price[i] < Exit and context.low[i] == False and context.high[i] == True and all(data.can_trade([stock1,stock2])):
            order_target_percent(stock1, 0)
            order_target_percent(stock2, 0)
            context.low[i] = False
            context.high[i] = False
            print('Ex Short  : %d-th pair ; %f @ %s and %f @ %s   '%((i+1),0,stock1.symbol,0,stock2.symbol))
        
def my_record_vars(context, data):

    record( Enter_Long=-Enter , Enter_Short=Enter )
#---Record z-scores of the first 3 pairs---#
    # record( RelPrice1=context.rel_price[0], RelPrice2=context.rel_price[1],RelPrice3=context.rel_price[2] )
    print('--------------------N-E-X-T---D-A-Y--------------------------')