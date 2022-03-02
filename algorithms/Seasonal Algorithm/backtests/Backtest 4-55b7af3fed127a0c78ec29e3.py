# Put any initialization logic here.  The context object will be passed to
# the other methods in your algorithm.
def initialize(context):
    context.spy = symbol('SPY')
    context.tlt = symbol('TLT')
    context.qqq = symbol('QQQ')
    
    schedule_function(buyTLT, 
                      date_rule = date_rules.week_start(),
                      time_rule = time_rules.market_open(minutes = 10))
    
    schedule_function(sellTLT, 
                      date_rule = date_rules.week_start(1),
                      time_rule = time_rules.market_close(minutes = 10))
    
    
    schedule_function(buySPY, 
                      date_rule = date_rules.week_start(4),
                      time_rule = time_rules.market_open(minutes = 1))
    
    schedule_function(sellSPY, 
                      date_rule = date_rules.week_start(),
                      time_rule = time_rules.market_open(minutes = 1))
    
    
    schedule_function(buyQQQ, 
                      date_rule = date_rules.week_start(2),
                      time_rule = time_rules.market_open(minutes = 10))
    
    schedule_function(sellQQQ, 
                      date_rule = date_rules.week_start(3),
                      time_rule = time_rules.market_close(minutes = 10))
    
    
    
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