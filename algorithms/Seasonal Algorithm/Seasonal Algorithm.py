# Put any initialization logic here.  The context object will be passed to
# the other methods in your algorithm.
def initialize(context):
    context.spy = symbol('SPY')
    context.tlt = symbol('TLT')
    context.qqq = symbol('QQQ')
    
    
    # Sel an open position 1 minute aafter market open
    # BUy a new position 10 minutes after market open
    
    cT = 1
    oT = 10
    
    # Monday Open - Sell SPY
    schedule_function(sellSPY, 
                      date_rule = date_rules.week_start(),
                      time_rule = time_rules.market_open(minutes = cT))
   
    # Monday Open - BUY TLT
    schedule_function(buyTLT, 
                      date_rule = date_rules.week_start(),
                      time_rule = time_rules.market_open(minutes =  oT))
    
    
    # Wednesday Open - Sell TLT
    schedule_function(sellTLT, 
                      date_rule = date_rules.week_start(2),
                      time_rule = time_rules.market_open(minutes = cT))
    
    # Wedesday Open - Buy QQQ
    schedule_function(buyQQQ, 
                      date_rule = date_rules.week_start(2),
                      time_rule = time_rules.market_open(minutes = oT))
    
    # Friday Open - Sell QQQ
    schedule_function(sellQQQ, 
                      date_rule = date_rules.week_start(4),
                      time_rule = time_rules.market_open(minutes = cT))
    
    # Friday Open - buy SPY
    schedule_function(buySPY, 
                      date_rule = date_rules.week_start(4),
                      time_rule = time_rules.market_open(minutes = oT))
    
    
    
def buySPY(context, data):
    order_target_percent(context.spy, 1)
    
def buyQQQ(context, data):
    order_target_percent(context.qqq, 1)
    
def buyTLT(context, data):
    order_target_percent(context.tlt, 1)

def sellSPY(context, data):
    order_target_percent(context.spy, 0)
    
def sellQQQ(context, data):
    order_target_percent(context.qqq, 0)
    
def sellTLT(context, data):
    order_target_percent(context.tlt, 0)
    
    
    
# Will be called on every trade event for the securities you specify. 
def handle_data(context, data):
   pass