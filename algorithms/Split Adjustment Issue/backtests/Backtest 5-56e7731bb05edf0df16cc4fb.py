# Put any initialization logic here.  The context object will be passed to
# the other methods in your algorithm.
def initialize(context):
    pass

# Will be called on every trade event for the securities you specify. 
def handle_data(context, data):
    # Implement your algorithm logic here.

    # data[sid(X)] holds the trade event data for that security.
    # context.portfolio holds the current portfolio state.

    # Place orders with the order(SID, amount) method.

    # TODO: implement your own logic here.
    order_target_percent(sid(27104), 1)
