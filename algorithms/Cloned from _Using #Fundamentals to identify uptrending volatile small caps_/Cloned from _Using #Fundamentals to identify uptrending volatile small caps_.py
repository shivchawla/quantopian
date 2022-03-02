#bld  20141231   18:44:17
# Below required to allow printing of current time in a specific timezone
from pytz import timezone

# Below required to determine dates when market closes early
from zipline.utils.tradingcalendar import get_early_closes


# Below of OLS method of least squares
import statsmodels.api as sm

# Below to let us sort dictionary by value
import operator
def initialize(context):

   # Algos must have at least one equity defined
   context.secs=symbols('SPY')

   # List to hold stock objects that are candidates
   context.MyList=[]

   # Dictionary to hold last stop price
   context.LastStop={}

   # Define how many stocks we will hold at once
   context.MaxHoldings=14

   # Close approximation for IB retail customers
   set_commission(commission.PerShare(cost=0.01, min_trade_cost=1.0)) 

   return
def before_trading_start(context): 
    
   num_stocks = 100
    
   dfFund = get_fundamentals(
      query(
            fundamentals.valuation.shares_outstanding,
            fundamentals.valuation.market_cap
      )
      .filter(fundamentals.valuation.market_cap != None)
      .filter(fundamentals.valuation.market_cap < 2000000000 )
      .filter(fundamentals.valuation.shares_outstanding != None)
      .filter(fundamentals.cash_flow_statement.free_cash_flow > 0)
      .filter(fundamentals.earnings_report.basic_eps > 0)
      .order_by(fundamentals.valuation.market_cap.desc())
      .limit(num_stocks)
   )

   context.MyList=[]
   for S in dfFund:
      if (('market_cap' in dfFund[S]) and ('shares_outstanding' in dfFund[S])) :
         # Approximate price - will test in handle_data before we buy
         ShareP=dfFund[S]['market_cap']/dfFund[S]['shares_outstanding']
         if ((ShareP < 25 ) and (ShareP > 2 )) :
            context.MyList.append(S)
        
   # Update our universe
   update_universe(context.MyList)
    
   return
def handle_data(context, data):

   # Start day by canceling any open orders
   D=get_open_orders()
   for L in D:
      for ID in D[L]:
         cancel_order(ID)

   # Create exit orders for every stock we are holding
   FracS = 0.900   # Stop Loss floor
   FracP = 1.140   # Profit taker

   for S in context.portfolio.positions:

      # Cost Basis
      CostB=context.portfolio.positions[S].cost_basis 
      if S in context.LastStop :
         StopV=context.LastStop[S]
      else :
         # First bar after purchase set floor price
         StopV=CostB * FracS

      # current price
      CurV=data[S].price

      # Keep moving stop lost up if we can
      if (FracS * CurV) > StopV:
         StopV=CurV * FracS

      order_target(S,0,style=StopOrder(StopV))

      # Update our list of the Stop prices
      context.LastStop[S]=StopV

      # Exit with a profit taker limit order
      order_target(S,0,style=LimitOrder(CostB * FracP))


   # Number of Holdings we have
   NPos=len(context.portfolio.positions)

   MyCash=context.portfolio.cash 
   PortV=context.portfolio.portfolio_value

   # Record some custom signals
   record(Positions=NPos,CashPerC=(100*MyCash/PortV))

   # If we are at limit we are done
   if NPos >= context.MaxHoldings :
      return

   # If we are low on cash dont buy 
   if MyCash < ( PortV / context.MaxHoldings) :
      return

   # Look for something to buy
   dfHistD = history(bar_count=30, frequency='1d', field='price')

   TrendList=[]
   for S in context.MyList :
      if S in data :
         CurP=data[S].price
         if (( S not in context.portfolio.positions) and (CurP < 20) and (CurP > 9 )):
            # Is stock trending up ?
            Y=dfHistD[S].values/dfHistD[S].mean()
            X=range(len(dfHistD))
            # Add column of ones so we get intercept
            A=sm.add_constant(X)
            results = sm.OLS(Y,A).fit()
            InterB,SlopeM=results.params
            if SlopeM >= 0.0 :
               # It is trending up  
               TrendList.append(S)

   # Determine standard deviation for each candidate   
   StdList={}
   for S in TrendList:
      StdList[S] = dfHistD[S].std()

   # Sort with most volatile stocks first
   BuyList=sorted(StdList.items(),key=operator.itemgetter(1),reverse=True)

   NBuyList=len(BuyList)
   if NBuyList==0 :
      # Nothing to buy today so return
      return

   OrderLimit=2  # Maximum purchases in one day
   NOrders=0

   # Determine order size
   OrderV = (MyCash - (PortV / context.MaxHoldings)) / (context.MaxHoldings - NPos)

   # Place orders for most volatile stocks
   for SL in BuyList :
      NPos += 1
      NOrders += 1
      if (( NPos <= context.MaxHoldings) and (NOrders <= OrderLimit)) :
         S=SL[0]
         order_value(S,1.5*OrderV)

         # Ensure our stop price record starts fresh on next bar
         if S in context.LastStop :
            del context.LastStop[S]
   
   return

#cks  a5132be3130241d51f768a1244725979 168 4815
