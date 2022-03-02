
def initialize(context):
    pass

# Will be called on every trade event for the securities you specify. 
def handle_data(context, data):

    order_target_percent(sid(3468), 1)
    order_target_percent(sid(3468), 0)
    
