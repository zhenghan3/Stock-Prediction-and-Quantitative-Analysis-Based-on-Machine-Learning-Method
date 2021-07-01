# Retrieve the factor data from the quantitative financial database

import pandas as pd
import numpy as np
import time
import datetime
import pickle
import os
from retrying import retry
import math

import tushare as ts
from jqdatasdk import *
from jqdatasdk.technical_analysis import *
from Authorization import *

import warnings

warnings.filterwarnings('ignore')


# Function to check if the API implementation is successful, retry if failed
wait_random_min = 500
wait_random_max = 1000

def retry_on_exception(exception):
    for exp in [BaseException]:
        if isinstance(exception, exp):
            print('Exception arose while getting the data from the web with API, retrying')
            print(exception)
            return True
    return False


# Implement the tushare API to retrieve the basic stock information of the A stock market
@retry(retry_on_exception=retry_on_exception, wait_random_min=wait_random_min, wait_random_max=wait_random_min)
def API_ts_stocks_basic():
    return (pd.concat([pro.stock_basic(exchange='', list_status='L',
                                       fields='ts_code,name,enname,list_status,list_date,delist_date'),
                       pro.stock_basic(exchange='', list_status='D',
                                       fields='ts_code,name,enname,list_status,list_date,delist_date'),
                       pro.stock_basic(exchange='', list_status='P',
                                       fields='ts_code,name,enname,list_status,list_date,delist_date')]
                      , axis=0))


# Implement the tushare API to retrieve the stocks has been suspended at the selected trade date
@retry(retry_on_exception=retry_on_exception, wait_random_min=wait_random_min, wait_random_max=wait_random_min)
def API_ts_suspend_d(trade_date):
    return (pro.suspend_d(suspend_type='S', trade_date=trade_date))


# Implement the tushare API to retrieve the stocks' daily trading information
@retry(retry_on_exception=retry_on_exception, wait_random_min=wait_random_min, wait_random_max=wait_random_min)
def API_ts_daily_basic(ts_code, start_date, end_date, fields=''):
    return (pro.daily_basic(ts_code=ts_code, start_date=start_date, end_date=end_date, fields=fields))


# Function to save the data locally in the specified format
def save_to_file(df, file_name, path='Database', csv_label=False, pkl_label=True, overwrite=True):
    if not os.path.exists(path):
        os.makedirs(path)

    if csv_label and pkl_label:
        if not os.path.exists(path + '/csv'):
            os.makedirs(path + '/csv')
        if not os.path.exists(path + '/pkl'):
            os.makedirs(path + '/pkl')
        if (not overwrite) and os.path.exists(path + '/pkl/' + file_name + '.pkl'):
            return False
        df.to_csv(path + '/csv/' + file_name + '.csv', encoding='utf-8_sig')
        with open(path + '/pkl/' + file_name + '.pkl', 'wb') as pk_file:
            pickle.dump(df, pk_file)
    elif csv_label:
        path = path + '/' + file_name
        if (not overwrite) and os.path.exists(path + '.csv'):
            return False
        df.to_csv(path + '.csv', encoding='utf-8_sig')
    elif pkl_label:
        path = path + '/' + file_name
        if (not overwrite) and os.path.exists(path + '.pkl'):
            return False
        with open(path + '.pkl', 'wb') as pk_file:
            pickle.dump(df, pk_file)


# Function to load the data from local
def read_from_file(file_name, path='Database', pkl_folder_label=False):
    if pkl_folder_label:
        path = path + '/' + 'pkl/' + file_name
    else:
        path = path + '/' + file_name
    with open(path + '.pkl', 'rb') as pk_file:
        df = pickle.load(pk_file)
    return df


# Function to retrieve the date_list within start_date and end_date
# Data is obtained every 20 days by default
def get_date_list(start_date, end_date, base_date, interval=20):
    """
    :param start_date, end_date: date within the range
    :param base_date: date is calculated from base_date to start_date and end_date
    :param interval: the date of trading days that been taken once every other interval
    :return: date list in list format
    """
    file_name = start_date + '_' + end_date + '_' + base_date + '_' + str(interval)
    if os.path.exists('Database/date_list/' + file_name + '.pkl'):
        return read_from_file(file_name, 'Database/date_list')

    # Obtain all the trade days within the range
    all_trade_days = (get_trade_days(start_date=start_date, end_date=end_date)).tolist()
    index = all_trade_days.index(datetime.date(*map(int, base_date.split('-'))))
    # Split all the trade days using interval
    date_list = all_trade_days[index::-interval] + all_trade_days[index::interval]

    date_list = [x.strftime('%Y-%m-%d') for x in date_list]
    date_list.sort()
    del date_list[date_list.index(base_date)]

    save_to_file(date_list, file_name, 'Database/date_list', csv_label=False)

    return (date_list)


# Factor_dict predefined for obtaining the fundamental factors
# Factors' comments are displayed in the format of (English name, Chinese name)
factor_dict = {
    'MC': valuation.market_cap,  # total market capitalisation, 总市值
    'CMC': valuation.circulating_market_cap,  # circulating market capitalisation, 流通市值
    'TR': valuation.turnover_ratio,  # turnover ratio, 换手率
    'PE': valuation.pe_ratio,  # price / earning ratio, 市盈率
    'PB': valuation.pb_ratio,  # price / book ratio, 市净率
    'PR': valuation.pcf_ratio,  # price / cash flow ratio, 市现率
    'ROE': indicator.roe,  # return of equity, 净资产收益率
    'DER': balance.total_liability / balance.equities_parent_company_owners,  # equity ratio, 产权比率 = 负债合计/归属母公司所有者权益合计
    'NPTTR': indicator.net_profit_to_total_revenue,  # net profit / total revenue, 净利润/营业总收入(%)
    'GPM': indicator.gross_profit_margin,  # gross profit margin, 销售毛利率 = 毛利/营业收入
    'ETTR': indicator.expense_to_total_revenue,  # expense / total revenue, 营业总成本/营业总收入(%)
    'OETTR': indicator.operating_expense_to_total_revenue,  # operating expense / total revenue, 营业费用/营业总收入
    'GETTR': indicator.ga_expense_to_total_revenue,  # management expense / total revenue, 管理费用/营业总收入(%)
    'OPTP': indicator.operating_profit_to_profit,  # operating profit / profit, 经营活动净收益/利润总额(%)
    'APTP': indicator.adjusted_profit / income.net_profit,  # adjusted profit / net profit, 扣除非经常损益后的净利润(元)/净利润
    'GSASTR': indicator.goods_sale_and_service_to_revenue,
    # cash received from goods sell and services provided / revenue, 销售商品提供劳务收到的现金/营业收入(%)
    'ITRYOY': indicator.inc_total_revenue_year_on_year,  # year-on-year growth rate of total revenue, 营业总收入同比增长率(%)
    'IRYOY': indicator.inc_revenue_year_on_year,  # year-on-year growth rate of revenue, 营业收入同比增长率(%)
    'IOPYOY': indicator.inc_operation_profit_year_on_year,  # year-on-year growth rate of operation profit, 营业利润同比增长率(%)
    'INPYOY': indicator.inc_net_profit_year_on_year,  # year-on-year growth rate of net profit, 净利润同比增长率(%)
    'INPA': indicator.inc_net_profit_annual,  # quarter-on-quarter growth rate of net profit, 净利润环比增长率(%)
    'INPTSA': indicator.inc_net_profit_to_shareholders_annual,
    # quarter-on-quarter growth rate of net profit to shareholders, 归属母公司股东的净利润环比增长率(%)

    # The factors below hasn't been chosen but retrieved in case of using
    'PS': valuation.ps_ratio,  # PS 市销率
    'GP': indicator.gross_profit_margin * income.operating_revenue,  # 毛利润
    'OP': income.operating_profit,  # 经营活动净收益(元)
    'OR': income.operating_revenue,  # 营业收入
    'NP': income.net_profit,  # 净利润
    'EV': valuation.market_cap + balance.shortterm_loan + balance.non_current_liability_in_one_year + balance.longterm_loan + balance.bonds_payable + balance.longterm_account_payable - cash_flow.cash_and_equivalents_at_end,

    'TOE': balance.total_owner_equities,  # 股东权益合计(元)
    'TOR': income.total_operating_revenue,  # 营业总收入
    'EBIT': income.net_profit + income.financial_expense + income.income_tax_expense,

    'TOC': income.total_operating_cost,  # 营业总成本
    'NOCF/MC': cash_flow.net_operate_cash_flow / valuation.market_cap,  # 经营活动产生的现金流量净额/总市值
    'OTR': indicator.ocf_to_revenue,  # 经营活动产生的现金流量净额/营业收入(%)

    'GPOA': indicator.gross_profit_margin * income.operating_revenue / balance.total_assets,  # 毛利润 / 总资产 = 毛利率*营业收入 / 总资产
    'OPM': income.operating_profit / income.operating_revenue,  # 营业利润率
    'NPM': indicator.net_profit_margin,  # 净利率
    'ROA': indicator.roa,  # ROA
    'INC': indicator.inc_return,  # 净资产收益率(扣除非经常损益)(%)
    'EPS': indicator.eps,  # 净资产收益率(扣除非经常损益)(%)
    'AP': indicator.adjusted_profit,  # 扣除非经常损益后的净利润(元)
    'OP': indicator.operating_profit,  # 经营活动净收益(元)
    'VCP': indicator.value_change_profit,  # 价值变动净收益(元) = 公允价值变动净收益+投资净收益+汇兑净收益

    'OPTTR': indicator.operation_profit_to_total_revenue,  # 营业利润/营业总收入(%)
    'FETTR': indicator.financing_expense_to_total_revenue,  # 财务费用/营业总收入(%)

    'IPTP': indicator.invesment_profit_to_profit,  # 价值变动净收益/利润总额(%)
    'OTOP': indicator.ocf_to_operating_profit,  # 经营活动产生的现金流量净额/经营活动净收益(%)

    'ITRA': indicator.inc_total_revenue_annual,  # 营业总收入环比增长率(%)
    'IRA': indicator.inc_revenue_annual,  # 营业收入环比增长率(%)
    'IOPA': indicator.inc_operation_profit_annual,  # 营业利润环比增长率(%)
    'INPTSYOY': indicator.inc_net_profit_to_shareholders_year_on_year,  # 归属母公司股东的净利润同比增长率(%)

    'ROIC': (income.net_profit + income.financial_expense + income.income_tax_expense) / (
            balance.total_owner_equities + balance.shortterm_loan + balance.non_current_liability_in_one_year + balance.longterm_loan + balance.bonds_payable + balance.longterm_account_payable),
    'OPTT': income.operating_profit / income.total_profit,  # 营业利润占比
    'TP/TOR': income.total_profit / income.total_operating_revenue,  # 利润总额/营业总收入
    'OP/TOR': income.operating_profit / income.total_operating_revenue,
    'NP/TOR': income.net_profit / income.total_operating_revenue,

    'TA': balance.total_assets,  # 总资产

    'FCFF/TNCL': (cash_flow.net_operate_cash_flow - cash_flow.net_invest_cash_flow) / balance.total_non_current_liability,  # 自由现金流比非流动负债
    'NOCF/TL': cash_flow.net_operate_cash_flow / balance.total_liability,  # 经营活动产生的现金流量净额/负债合计
    'TCA/TCL': balance.total_current_assets / balance.total_current_liability,  # 流动比率

    'TOR/TA': income.total_operating_revenue / balance.total_assets,  # 总资产周转率
    'TOR/FA': income.total_operating_revenue / balance.fixed_assets,  # 固定资产周转率
    'TOR/TCA': income.total_operating_revenue / balance.total_current_assets,  # 流动资产周转率
    'LTL/OC': balance.longterm_loan / income.operating_cost,  # 长期借款/营业成本

    'TL/TA': balance.total_liability / balance.total_assets,  # 总资产/总负债
    'TL/TOE': balance.total_liability / balance.total_owner_equities,  # 负债权益比

}


# Obtain fundamental factors' data of the query_date
def get_fundamental_factor_data_bydate(stock_list, factor_list, query_date):
    """
    :param stock_list: list of stocks to be queried
    :param factor_list: list of factors to be queried
    :param query_date: date to be queried
    :return: queried data in dataframe
    """
    # Query the remaining number of API calls
    print(get_query_count())

    q = query(valuation.code)
    for fac in factor_list:
        q = q.add_column(factor_dict[fac])
    q = q.filter(valuation.code.in_(stock_list))
    factor_df = get_fundamentals(q, query_date)

    factor_df.index = factor_df['code']
    factor_df.columns = ['code'] + factor_list
    return factor_df

# Obtain technical factors' data of the query_date
def get_technical_factor_data_bydate(stock_list, query_date):
    # Query the remaining number of API calls
    print(get_query_count())

    # Implement get_factor values API to retrieve selected factors
    jq_factor = get_factor_values(stock_list,
                  factors=['VOL20',  # average turnover ratio on 20 days, 20日平均换手率
                           'sharpe_ratio_20',  # sharpe ratio on 20 days, 20日夏普比率
                           'BIAS20',  # average deviation rate on 20 days, 20日乖离率
                           'ROC20',  # price rate of change on 20 days, 20日变动速率
                           'MFI14',  # money flow index, 资金流量指标

                           'momentum',  # momentum, 动量
                           'residual_volatility',  # residual volatility, 残差波动率
                           'liquidity',  # liquidity, 流动性
                           'earnings_yield',  # profitability 盈利能力
                           'leverage',  # leverage, 杠杆
                           ],
                  end_date=query_date, count=1)
    df_jq_factor = pd.DataFrame(index=stock_list)

    for i in jq_factor.keys():
        df_jq_factor[i] = jq_factor[i].iloc[0, :]

    # ATR - average true range, 真实波幅
    # n=14
    mtr, atr = ATR(stock_list, query_date, timeperiod=14)
    df_atr = pd.DataFrame(list(atr.values()), columns=['ATR'], index=stock_list)

    # MTM - momentum line, 动量线
    # n=20
    mtm = MTM(stock_list, query_date, timeperiod=20)
    df_mtm = pd.DataFrame(list(mtm.values()), columns=['MTM'], index=stock_list)

    # MACD - moving average convergence divergence, 平滑异同平均
    # short=12, long=26, mid=9
    dif, dea, macd = MACD(stock_list, query_date, SHORT=12, LONG=26, MID=9)
    df_macd = pd.DataFrame(list(macd.values()), columns=['MACD'], index=stock_list)

    # RSI - relative strength index, 相对强弱指标
    # n=6
    rsi = RSI(stock_list, query_date, N1=6)
    df_rsi = pd.DataFrame(list(rsi.values()), columns=['RSI'], index=stock_list)

    # PSY - psychological line, 心理线
    # n=12
    psy = PSY(stock_list, query_date, timeperiod=12)
    df_psy = pd.DataFrame(list(psy.values()), columns=['PSY'], index=stock_list)

    # CYR - relative market strength index, 市场强弱
    # n=13, m=5
    cyr, macyr = CYR(stock_list, query_date, N=13, M=5)
    df_cyr = pd.DataFrame(list(cyr.values()), columns=['CYR'], index=stock_list)

    df = pd.concat([df_jq_factor, df_atr, df_mtm, df_macd, df_rsi, df_psy, df_cyr], axis=1)
    df.index.name = 'code'
    return df

# Implement tushare API to retrieve turnover ratio data for calculating wgt factors
# Retrieve date from query date to N days ago(default 380 days)
def get_turnover_ratio_data_ts(stock_list, query_date, trade_days, days=380):
    stock_list_deleted = stock_list.copy()
    df_turnover_ratio = pd.DataFrame(index=stock_list_deleted)
    df_turnover_ratio.index.name = 'code'

    query_date = datetime.datetime.strptime(query_date, '%Y-%m-%d')
    previous_date = query_date - datetime.timedelta(days=days)
    query_date = query_date.strftime('%Y%m%d')
    previous_date = previous_date.strftime('%Y%m%d')

    df = pd.DataFrame()
    code = ''
    length = 0
    while len(stock_list_deleted) > 0:
        length += 1
        stock = stock_list_deleted[0]
        # Convert the format of stock code from joinquant to tushare
        stock = stock.replace('XSHG', 'SH')
        stock = stock.replace('XSHE', 'SZ')
        code = code + stock + ','
        del (stock_list_deleted[0])
        # Set the interval to prevent API calling restrictions
        if length == 15:
            code = code[:-1]
            df = pd.concat([df, API_ts_daily_basic(ts_code=code, start_date=previous_date, end_date=query_date,
                                                   fields='ts_code,trade_date,turnover_rate')])
            code = ''
            length = 0
    if code != '':
        df = pd.concat([df, API_ts_daily_basic(ts_code=code, start_date=previous_date, end_date=query_date,
                                               fields='ts_code,trade_date,turnover_rate')])

    # trade_days=list(df.drop_duplicates('trade_date')['trade_date'])
    # reconstruct the data into joinquant format and saved in dataframe
    df_turnover_ratio = pd.DataFrame(index=stock_list)
    df_turnover_ratio.index.name = 'code'
    for current_date in trade_days:
        current_date = current_date.strftime('%Y%m%d')
        df_temp = df.loc[df['trade_date'] == current_date].rename(columns={'ts_code': 'code'})
        df_temp['code'] = df_temp['code'].str.replace('SH', 'XSHG')
        df_temp['code'] = df_temp['code'].str.replace('SZ', 'XSHE')
        df_temp = df_temp.set_index('code')
        df_turnover_ratio = pd.merge(df_turnover_ratio, df_temp['turnover_rate'], how='left', on='code')
        df_turnover_ratio = df_turnover_ratio.rename(
            columns={'turnover_rate': datetime.datetime.strptime(current_date, '%Y%m%d')})

    df_turnover_ratio = df_turnover_ratio.T

    # take 1 second to retrieve one stock, abadonded due to the time consuming
    '''
    for stock in stock_list:
        print(stock_list.index(stock))
        stock = stock.replace('XSHG', 'SH')
        stock = stock.replace('XSHE', 'SZ')
        df = API_ts_daily_basic(ts_code=stock, start_date=previous_date, end_date=query_date,
                                fields='trade_date,turnover_rate')
        stock = stock.replace('SH', 'XSHG')
        stock = stock.replace('SZ', 'XSHE')
        df = df.rename(columns={'turnover_rate': stock})
        df = df.set_index('trade_date').T
        df.index.name = 'code'
        if stock_list.index(stock) == 0:
            df_turnover_ratio = pd.concat([df_turnover_ratio, df], axis=1)
        else:
            df_turnover_ratio.update(df)
    '''
    return df_turnover_ratio

# Implement joinquant API to retrieve turnover ratio data for calculating wgt factors
# Used in conjunction with the get_turnover_ratio_data_ts due to the limitation of daily data acquisition,
def get_turnover_ratio_data_jq(stock_list, trade_days):
    df_turnover_ratio = pd.DataFrame(index=stock_list)
    df_turnover_ratio.index.name = 'code'
    for current_date in trade_days:
        print(get_query_count())
        print(trade_days.index(current_date))
        q = query(valuation.code, valuation.turnover_ratio).filter(valuation.code.in_(stock_list))
        df_temp = get_fundamentals(q, current_date)
        df_turnover_ratio = pd.merge(df_turnover_ratio, df_temp, how='left', on='code')
        # df_turnover_ratio=df_turnover_ratio.rename(columns={'turnover_ratio':current_date.strftime('%Y-%m-%d')})
        df_turnover_ratio = df_turnover_ratio.rename(columns={'turnover_ratio': current_date})
    df_turnover_ratio = df_turnover_ratio.set_index('code').T
    return df_turnover_ratio

# Retrieve wgt factors(momentum reversal) of the query date
def get_wgt_factor_data_bydate(stock_list, query_date, days=250):
    trade_days = list(get_trade_days(end_date=query_date, count=days))
    print(get_query_count())

    # Retrieve turnover ratio data
    df_turnover_ratio = get_turnover_ratio_data_ts(stock_list, query_date, trade_days)
    # df_turnover_ratio=get_turnover_ratio_data_jq(stock_list,trade_days)

    # retrieve the daily price data
    df_daily = get_price(stock_list, count=days, end_date=query_date, frequency='daily', fields=['close'])
    df_daily.sort_values(by='code', inplace=True)
    print(get_query_count())
    # df_daily['time']=df_daily['time'].apply(lambda x:x.strftime('%Y-%m-%d'))
    df_pchg = pd.DataFrame(index=stock_list)
    df_pchg.index.name = 'code'
    for current_date in trade_days:
        df_temp = df_daily.loc[df_daily['time'] == current_date].set_index('code')
        df_pchg = pd.merge(df_pchg, df_temp['close'], how='left', on='code')
        df_pchg = df_pchg.rename(columns={'close': current_date})

    df_pchg = df_pchg.T
    df_pchg = df_pchg.pct_change()

    # Another method to set the value of the DataFrame
    '''
    for i in range(len(df_daily)):
        df_temp=df_daily.iloc[i]
        df_pchg.loc[df_temp['code'],df_temp['time']]=df_temp['close']
    '''

    # Calculate the wgt factors of the different time periods
    df = pd.DataFrame(index=stock_list)
    df.index.name = 'code'
    df['wgt_return_1m'] = np.mean((df_pchg.iloc[-20:] * df_turnover_ratio.iloc[-20:]))
    df['wgt_return_3m'] = np.mean(df_pchg.iloc[-60:] * df_turnover_ratio.iloc[-60:])
    df['wgt_return_6m'] = np.mean(df_pchg.iloc[-120:] * df_turnover_ratio.iloc[-120:])
    df['wgt_return_12m'] = np.mean(df_pchg.iloc[-240:] * df_turnover_ratio.iloc[-240:])

    # Calculate the exp wgt factors of the different time periods
        # The daily turnover ratio is multiplied by the function exp(-i / N / 4) and
        # multiplied by the daily yield to get the arithmetic average in the last N months.
    df_temp = pd.DataFrame(index=df_turnover_ratio[-240:].index, columns=stock_list)
    temp = []
    for i in range(240):
        if i / 20 < 1:
            temp.append(math.exp(-i / 1 / 4))
        elif i / 20 < 3:
            temp.append(math.exp(-i / 3 / 4))
        elif i / 20 < 6:
            temp.append(math.exp(-i / 6 / 4))
        elif i / 20 < 12:
            temp.append(math.exp(-i / 12 / 4))
    temp.reverse()
    for i in stock_list:
        df_temp[i] = temp
    df['exp_wgt_return_1m'] = np.mean(df_pchg.iloc[-20:] * df_temp.iloc[-20:] * df_turnover_ratio.iloc[-20:])
    df['exp_wgt_return_3m'] = np.mean(df_pchg.iloc[-60:] * df_temp.iloc[-60:] * df_turnover_ratio.iloc[-60:])
    df['exp_wgt_return_6m'] = np.mean(df_pchg.iloc[-120:] * df_temp.iloc[-120:] * df_turnover_ratio.iloc[-120:])
    df['exp_wgt_return_12m'] = np.mean(df_pchg.iloc[-240:] * df_temp.iloc[-240:] * df_turnover_ratio.iloc[-240:])

    return df

# Retrieve the stocks' price data of the query date
def get_price_data_bydate(stock_list, query_date, next_query_date):
    df_price = get_price(stock_list, query_date, next_query_date, '1d')
    df = pd.DataFrame(index=stock_list)
    df.index.name = 'code'
    for stock in stock_list:
        df_temp = df_price.loc[df_price['code'] == stock]
        df.loc[stock, 'close'] = df_temp.iloc[0]['close']
        df.loc[stock, 'pchg'] = df_temp.iloc[-1]['close'] / df_temp.iloc[0]['close'] - 1
    return df


# Remove the stocks that do not meet the requirements on trading days
    # stocks listed less than 90 days
    # stocks has been delisted
    # stocks suspended on the trading day
def delete_stock(stock_list, current_date, n=90):
    stock_list_deleted = []

    # Transfer the str date to datetime.date
    current_date = datetime.datetime.strptime(current_date, '%Y-%m-%d')

    df_suspend = API_ts_suspend_d(trade_date=current_date.strftime('%Y%m%d'))
    df_suspend['ts_code'] = df_suspend['ts_code'].str.replace('SH', 'XSHG')
    df_suspend['ts_code'] = df_suspend['ts_code'].str.replace('SZ', 'XSHE')

    for stock in stock_list:
        try:
            # Remove the stocks listed less than 90 days
            list_date = datetime.datetime.strptime(
                df_basic.loc[df_basic['ts_code'] == stock]['list_date'].reset_index(drop=True)[0], '%Y%m%d')
            delist_date = df_basic.loc[df_basic['ts_code'] == stock]['delist_date'].reset_index(drop=True)[0]
        except:
            continue

        # Remove the stocks have been delisted
        # Remove the stocks suspended on the trading day
        if (list_date < current_date - datetime.timedelta(days=n)) and \
                ((delist_date == None) or ((delist_date != None) and (
                        current_date < datetime.datetime.strptime(delist_date, '%Y%m%d')))) and \
                (stock not in df_suspend['ts_code'].values):
            stock_list_deleted.append(stock)

    return stock_list_deleted


# Obtain the index constituent stocks of the specified index
def get_stock_list(stock_index, current_date):
    if os.path.exists('Database/stock_list/' + current_date + '.pkl'):
        return read_from_file(current_date, 'Database/stock_list')

    if stock_index == 'HS300':
        stock_list = get_index_stocks('000300.XSHG', current_date)
    elif stock_index == 'ZZ500':
        stock_list = get_index_stocks('399905.XSHE', current_date)
    elif stock_index == 'ZZ800':
        stock_list = get_index_stocks('399906.XSHE', current_date)
    elif stock_index == 'CYBZ':
        stock_list = get_index_stocks('399006.XSHE', current_date)
    elif stock_index == 'ZXBZ':
        stock_list = get_index_stocks('399005.XSHE', current_date)
    elif stock_index == 'A':
        stock_list = get_index_stocks('000002.XSHG', current_date) + get_index_stocks('399107.XSHE', current_date)

    # Remove the ST(Special Treatment) stocks
    st_data = get_extras('is_st', stock_list, count=1, end_date=current_date)
    stock_list = [stock for stock in stock_list if not st_data[stock][0]]

    # Call the delete_stock function
    stock_list = delete_stock(stock_list, current_date)

    save_to_file(stock_list, current_date, 'Database/stock_list', csv_label=False)

    return stock_list

# Function to combine the fundamental, technical, wgt and price factors together
def get_all_factor_data(factor_list, date_list, wgt_factor=False):
    global interval
    for current_date in date_list:
        print(current_date)
        stock_list = get_stock_list('ZZ800', current_date)
        # current_date='2020-07-10'
        # Retrieve the fundamental factor data
        if not os.path.exists('Database/fundamental_factor/pkl/' + current_date + '.pkl'):
            df = get_fundamental_factor_data_bydate(stock_list, factor_list, current_date)
            save_to_file(df, current_date, 'Database/fundamental_factor', csv_label=True)

        # Retrieve the technical factors data
        if not os.path.exists('Database/technical_factor/pkl/' + current_date + '.pkl'):
            df = get_technical_factor_data_bydate(stock_list, current_date)
            save_to_file(df, current_date, 'Database/technical_factor', csv_label=True)
        if wgt_factor:
            if not os.path.exists('Database/wgt_factor/pkl/' + current_date + '.pkl'):
                df = get_wgt_factor_data_bydate(stock_list, current_date)
                save_to_file(df, current_date, 'Database/wgt_factor', csv_label=True)

        # Retrieve the price data
        if not os.path.exists('Database/price/pkl/' + current_date + '.pkl'):
            if (interval == 20) and (current_date != date_list[-1]):
                df = get_price_data_bydate(stock_list, current_date, date_list[date_list.index(current_date) + 1])
                save_to_file(df, current_date, 'Database/price', csv_label=True)
            if (interval == 5) and (date_list.index(current_date) < len(date_list) - 4):
                df = get_price_data_bydate(stock_list, current_date, date_list[date_list.index(current_date) + 4])
                save_to_file(df, current_date, 'Database/price', csv_label=True)

# Function to retrieve the wgt factors in random order for multiple threading use
def get_wgt_factor_data(date_list):
    from random import choice
    while True:
        if not date_list:
            break
        current_date = choice(date_list)
        print(current_date)
        if not os.path.exists('Database/wgt_factor/pkl/' + current_date + '.pkl'):
            stock_list = get_stock_list('ZZ800', current_date)
            df = get_wgt_factor_data_bydate(stock_list, current_date)
            save_to_file(df, current_date, 'Database/wgt_factor', csv_label=True)
        date_list.remove(current_date)

if __name__ == '__main__':
    # Get authorization from Joinquant and Tushare
    pro = authorization(1)

    interval = 20
    start_date = '2007-01-15'
    end_date = '2020-07-25'
    base_date = '2015-12-31'

    # Obtain the date list of the trading days of the specified interval
    date_list = get_date_list(start_date, end_date, base_date, interval=interval)

    # securities = get_all_securities()
    # save_to_file(securities, 'all_stocks', pkl_label=False)

    # Prepare the df_basic for function using
    df_basic=API_ts_stocks_basic()
    df_basic['ts_code']=df_basic['ts_code'].str.replace('SH', 'XSHG')
    df_basic['ts_code']=df_basic['ts_code'].str.replace('SZ', 'XSHE')

    factor_list = list(factor_dict)

    # get_wgt_factor_data(date_list)
    get_all_factor_data(factor_list, date_list)
    # get_all_factor_data(factor_list,date_list,wgt_factor=True)
