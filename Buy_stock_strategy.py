# Backtest the strategic return according to the buy list
# Run under the joinquant quantitative financial platform, https://www.joinquant.com/algorithm/index/list

import pandas as pd
from six import BytesIO
import pickle

from jqdata import *

dict_interval = {20: 0, 50: 1, 100: 2}
interval = 100
buy_list = read_file('buy_list.json')
buy_list = json.loads(buy_list)


# Function to initialize, set the benchmark, commission charge and so on.
def initialize(context):
    # Set China Securities Index 300 as benchmark
    set_benchmark('000300.XSHG')
    # Set dynamic rehabilitation mode(real price)
    set_option('use_real_price', True)

    # Set commission charge of each trade
    set_order_cost(OrderCost(close_tax=0.001, open_commission=0.0003, close_commission=0.0003,
                             min_commission=5), type='stock')

    # Run before market opening
    run_daily(before_market_open, time='09:00', reference_security='000300.XSHG')
    # Run at market opening
    run_daily(market_open, time='09:30', reference_security='000300.XSHG')


# Function to be run before market opening
def before_market_open(context):
    current_date = context.current_dt.date().strftime('%Y-%m-%d')
    if current_date in buy_list:
        print('调仓日期：%s' % context.current_dt.date())
        # Obtain buy list of current date
        g.buy_list = list(buy_list[current_date][dict_interval[interval]])
        print(g.buy_list)


# Function to be run at market opening
def market_open(context):
    current_date = context.current_dt.date().strftime('%Y-%m-%d')
    if current_date in buy_list:
        # Sell the stocks not in current_date's buy list
        sell_list = set(context.portfolio.positions.keys()) - set(g.buy_list)
        for stock in sell_list:
            order_target_value(stock, 0)

        # Equal weight to buy stocks
        optimized_weight = pd.Series(data=[1.0 / len(g.buy_list)] * len(g.buy_list),
                                     index=g.buy_list)

        # Give a warning if failed
        if type(optimized_weight) == type(None):
            print('Warning')

        # Carry out the buying operation
        else:
            # Obtain the total capital in account
            total_value = context.portfolio.total_value
            for stock in optimized_weight.keys():
                # Equal weight to buy stocks
                value = total_value * optimized_weight[stock]
                # Adjust the target to the target weight
                order_target_value(stock, value)