
def initialize(context):
    pass

# Will be called on every trade event for the securities you specify. 
def handle_data(context, data):

    order_target_percent(sid(22472), 1)
    order_target_percent(sid(22472), 0)
    
