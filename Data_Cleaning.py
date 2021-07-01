# Implement the data cleaning of the factor data

import statsmodels.api as sm
from sklearn.utils import shuffle

from Get_factor_Data import *

dev_set_index_dict=[0,0,1,1,2,2,3,3,3,4,4,4]
interval_dict = {20: 12, 5: 48}

# Factor dict that contains all the factors has been chosen
factor_selected = {
    'MC': valuation.market_cap,  # total market capitalisation, 总市值
    'CMC': valuation.circulating_market_cap,  # circulating market capitalisation, 流通市值
    'TR': valuation.turnover_ratio,  # turnover ratio, 换手率
    'PE': valuation.pe_ratio,  # price / earning ratio, 市盈率
    'PB': valuation.pb_ratio,  # price / book ratio, 市净率
    'PR': valuation.pcf_ratio,  # price / cash flow ratio, 市现率
    'ROE': indicator.roe,  # return of equity, 净资产收益率
    'leverage':1,  # leverage, 杠杆
    'DER': balance.total_liability / balance.equities_parent_company_owners,  # equity ratio, 产权比率 = 负债合计/归属母公司所有者权益合计
    'NPTTR': indicator.net_profit_to_total_revenue,  # net profit / total revenue, 净利润/营业总收入(%)
    'GPM': indicator.gross_profit_margin,  # gross profit margin, 销售毛利率 = 毛利/营业收入
    'ETTR': indicator.expense_to_total_revenue,  # expense / total revenue, 营业总成本/营业总收入(%)
    'OETTR': indicator.operating_expense_to_total_revenue,  # operating expense / total revenue, 营业费用/营业总收入
    'GETTR': indicator.ga_expense_to_total_revenue,  # management expense / total revenue, 管理费用/营业总收入(%)
    'OPTP': indicator.operating_profit_to_profit,  # operating profit / profit, 经营活动净收益/利润总额(%)
    'APTP': indicator.adjusted_profit / income.net_profit,  # adjusted profit / net profit, 扣除非经常损益后的净利润(元)/净利润
    'GSASTR': indicator.goods_sale_and_service_to_revenue,  # cash received from goods sell and services provided / revenue, 销售商品提供劳务收到的现金/营业收入(%)
    'ITRYOY': indicator.inc_total_revenue_year_on_year,  # year-on-year growth rate of total revenue, 营业总收入同比增长率(%)
    'IRYOY': indicator.inc_revenue_year_on_year,  # year-on-year growth rate of revenue, 营业收入同比增长率(%)
    'IOPYOY': indicator.inc_operation_profit_year_on_year,  # year-on-year growth rate of operation profit, 营业利润同比增长率(%)
    'INPYOY': indicator.inc_net_profit_year_on_year,  # year-on-year growth rate of net profit, 净利润同比增长率(%)
    'INPA': indicator.inc_net_profit_annual,  # quarter-on-quarter growth rate of net profit, 净利润环比增长率(%)
    'INPTSA': indicator.inc_net_profit_to_shareholders_annual,  # quarter-on-quarter growth rate of net profit to shareholders, 归属母公司股东的净利润环比增长率(%)

    'VOL20':1,  # average turnover ratio on 20 days, 20日平均换手率
    'sharpe_ratio_20':1,  # average sharp ratio on 20 days, 20日夏普比率
    'BIAS20':1,  # average deviation rate on 20 days, 20日乖离率
    'ROC20':1,  # price rate of change on 20 days, 20日变动速率
    'MFI14':1,  # money flow index, 资金流量指标

    'momentum':1,  # momentum, 动量
    'residual_volatility':1,  # residual_volatility, 残差波动率
    'liquidity':1,  # liquidity, 流动性
    'earnings_yield':1,  # profitability, 盈利能力

    'ATR':1,  # average true range, 真实波幅
    'MTM':1,  # momentum line, 动量线
    'MACD':1,  # moving average convergence divergence, 平滑异同平均
    'RSI':1,  # relative strength index, 相对强弱指标
    'PSY':1,  # psychological line, 心理线
    'CYR':1,  # relative market strength index, 市场强弱

    # momentum reversal
    'wgt_return_1m': 1,
    'wgt_return_3m': 1,
    'wgt_return_6m': 1,
    'wgt_return_12m': 1,
    'exp_wgt_return_1m': 1,
    'exp_wgt_return_3m': 1,
    'exp_wgt_return_6m': 1,
    'exp_wgt_return_12m': 1,

    # relative increase
    'pchg':1,
}

# Load the factors' data from local and combined together
def factor_read_from_file(current_date):
    df_fundamental=read_from_file(current_date,'Database/fundamental_factor',pkl_folder_label=True)
    df_fundamental.drop(['code'],axis=1,inplace=True)
    df_technical=read_from_file(current_date,'Database/technical_factor',pkl_folder_label=True)
    df_wgt=read_from_file(current_date,'Database/wgt_factor',pkl_folder_label=True)
    df_price=read_from_file(current_date,'Database/price',pkl_folder_label=True)
    df=pd.concat([df_fundamental,df_technical,df_wgt,df_price],axis=1)
    df=pd.DataFrame(df,columns=list(factor_selected))
    df['MC']=df['MC']*100000000
    return df

# Plot the heat map of the factor correlation
def corr_heat_map(df):
    import seaborn as sns
    import matplotlib.pyplot as plt

    sns.set(style="white")
    corr = df.corr()
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    f, ax = plt.subplots(figsize=(20, 18))
    # Set the color style
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(corr, mask=mask,  cmap=cmap, vmax=1 , linewidths=.5)
    plt.show()

# Retrieve the stocks' corresponding industry within the stock_list
def get_stock_industry(industry_name,current_date,stock_list):
    """
    :param industry_name:
        "sw_l1": shenwan first-class industry(default), 申万一级行业
        "sw_l2": shenwan second-class industry, 申万二级行业
        "sw_l3": shenwan third-class industry, 申万三级行业
        "zjw": securities regulatory commission, 证监会行业
    :param current_date: date to be queried
    :param stock_list: stocks to be queried
    :return: industry code in dataframe
    """
    if os.path.exists('Database/stock_industry.pkl'):
        industries = read_from_file('stock_industry')
    else:
        industries = list(get_industries(industry_name).index)
        save_to_file(industries,'stock_industry')

    df = pd.DataFrame(index=stock_list)
    # Set the stock's industry that does not belong to any industry as '000000'
    df['industry_code'] = '000000'
    for ind in industries:
        if os.path.exists('Database/industry_stocks/'+current_date+'/'+ind+'.pkl'):
            industry_stocks = read_from_file(ind,'Database/industry_stocks/'+current_date)
        else:
            industry_stocks = get_industry_stocks(ind, current_date)
            save_to_file(industry_stocks,ind,'Database/industry_stocks/'+current_date)
        # Stocks in the current industry intersect with the stock_list
        industry_stocks = set(df.index) & set(industry_stocks)
        df['industry_code'][industry_stocks] = ind
    stock_industry = df['industry_code'].to_frame()
    return stock_industry

# Fill in the vacancy values with industry mean
def fillna_with_industry(df, current_date, industry_name='sw_l1'):
    stock_industry = get_stock_industry(industry_name, current_date,list(df.index))
    df_merge = pd.concat([df,stock_industry],axis=1)
    columns = list(df.columns)
    df_res = []
    # Calculate the industry mean
    df_groupby_mean = df_merge.groupby('industry_code').mean()
    # Merge the original table with the mean table, _x represents original table, _y represents mean table
    df_merge = df_merge.merge(df_groupby_mean, left_on='industry_code', right_index=True, how='left')
    for column in columns:
        if type(df[column][0]) != str:
            # Fill in the nan value with the industry mean
            df_merge[column + '_x'][pd.isnull(df_merge[column + '_x'])] = df_merge[column + '_y'][
                pd.isnull(df_merge[column + '_x'])]

            df_merge[column] = df_merge[column + '_x']
            df_res.append(df_merge[column])

    df_res = pd.concat(df_res, axis=1)
    # Fill in the rest nan value with overall mean if there is no industry code
    mean = df_res.mean()
    for i in df_res.columns:
        df_res[i].fillna(mean[i], inplace=True)
    return df_res

# Neutralize the factor data with industry dummy and logarithm of market cap
def neutralize(df, current_date, industry_name='sw_l1'):
    stock_industry = get_stock_industry(industry_name, current_date,list(df.index))
    market_cap = df['MC'].to_frame()
    index = list(df.index)

    # Obtain the industry dummy using Statsmodels library
    stock_industry = np.array(stock_industry['industry_code'])
    industry_dummy = sm.categorical(stock_industry, drop=True)
    industry_dummy = pd.DataFrame(industry_dummy, index=index)
    # Take logarithm of market cap
    market_cap = np.log(market_cap)
    # Combine the two together as independent variables
    x = pd.concat([industry_dummy, market_cap], axis=1)
    model = sm.OLS(df, x)
    result = model.fit()
    # Calculate the dependent variable y
    y_fitted = result.fittedvalues
    # Remove the part(y) that is affected by both from the original
    df_neu = df - y_fitted
    return df_neu

# Median extremum elimination with MAD method
def winsorize(df,n):
    '''
    :param n: intercept range
    '''

    # Obtain the median in the data
    median = df.quantile(0.5)
    # Obtain the median of median and the data difference
    new_median = ((df - median).abs()).quantile(0.50)
    # Calculate upper and lower limits
    max_range = median + n*new_median
    min_range = median - n*new_median
    # Replace the data outside the limit with the upper and lower limits
    return np.clip(df,min_range,max_range,axis=1)

# Implement the Z-score method to standardize the data
def standardize(df):
    df = (df - df.mean())/df.std()
    return df

# Function to implement the data cleaning
# Order: winsorize -> fillna -> neutralize -> standardize
def preprocess(df,current_date):
    #df_factor.to_clipboard(sep=',')
    df_factor=df.drop(['pchg','label'],axis=1)
    df_factor=winsorize(df_factor,5)
    df_factor=fillna_with_industry(df_factor,current_date)
    df_factor=neutralize(df_factor,current_date)
    df_factor=standardize(df_factor)
    #corr_heat_map(df)
    df=pd.concat([df_factor,df[['pchg','label']]],axis=1)
    return df

# Function to calculate the top and last index with the reserved_proportion
def top_last_index(num,reserved_proportion):
    top_index=round(num*reserved_proportion)
    last_index=round(num*(1-reserved_proportion))
    return top_index,last_index

# Split the training set into training set and dev set with the specified ratio
def train_test_split(df, split_size=0.2, random_state=None):
    df = shuffle(df, random_state=random_state)
    df_train = df[round(len(df) * split_size):]
    df_dev = df[:round(len(df) * split_size)]

    return df_train, df_dev

# Function to retrieve the cleaned dataset of the current date
def get_dataset_bydate_dev(current_date,date_list,interval=20,dev_set=False,year=2,reserved_proportion=0.3, dev_set_proportion=0.2,seed=27,drop=True):
    """
    :param dev_set: return the development set if yes
    :param year: Obtain the data of the length n years before the current_date
    :param reserved_proportion: the percentage of the data is retained before and after
    :param dev_set_proportion: the percentage of the data taken as the dev set
    :param drop: if drop the 30% of the data in the middle
    """
    # Retrieve data from function get_dataset_bydate
    df_train,df_test=get_dataset_bydate(current_date,date_list,interval=interval,dev_set=False,year=year,reserved_proportion=reserved_proportion,drop=drop)

    # check if the dev_set needed
    if dev_set:
        df_train,df_dev=train_test_split(df_train,split_size=dev_set_proportion,random_state=seed)
        return df_train,df_dev,df_test
    else:
        return df_train,df_test

# Function to retrieve the cleaned dataset of the current date
def get_dataset_bydate(current_date,date_list,interval=20,dev_set=False,year=2,reserved_proportion=0.3,drop=True):
    """
    :param dev_set: return the development set index if yes
    :param year: Obtain the data of the length n years before the current_date
    :param reserved_proportion: the percentage of the data is retained before and after
    :param drop: if drop the 30% of the data in the middle
    """
    # Check whether the local data exists
    if drop:
        file_name=current_date+'_drop'
    else:
        file_name=current_date
    path='Database/dataset/'+str(interval)+'/'+str(year)+'/'+file_name+'.pkl'

    if os.path.exists(path):
        df = read_from_file(file_name,'Database/dataset/'+str(interval)+'/'+str(year))
        df_train=df[0]
        df_test=df[1]
        dev_set_index=df[2]
        if dev_set:
            return df_train,dev_set_index,df_test
        else:
            return df_train,df_test

    # Calculate the data set length according to interval
    n=interval_dict[interval]
    total_n=year*n

    index = date_list.index(current_date)
    date_list_selected=date_list[(index-n):index]
    df_train=pd.DataFrame()
    for date in date_list_selected:
        df_temp=factor_read_from_file(date)
        df_temp['dev_index']=len(df_temp)*[dev_set_index_dict[date_list_selected.index(date)]]
        df_train=pd.concat([df_train,df_temp])

    # Obtain the full training set data
    index=date_list.index(current_date)
    date_list_selected=date_list[(index-total_n):index-n]
    for date in date_list_selected:
        df_temp=factor_read_from_file(date)
        df_temp['dev_index']=len(df_temp)*[-1]
        df_train=pd.concat([df_train,df_temp])
    # Remove the stocks with no next month's profit
    df_train.dropna(subset=['pchg'], inplace=True)
    # Sort the dataframe by next month's profit in descending order
    df_train=df_train.sort_values(by=['pchg'],ascending=False)

    if drop:
        # Calculate the index to be preserved
        top_index,last_index=top_last_index(len(df_train),reserved_proportion)
        df_train=pd.concat([df_train.iloc[:top_index,:],df_train.iloc[last_index:,:]])
        # Label The top 30% of next month's profit as 1, and the last 30% as 0, drop the middle
        mean=np.mean(list(df_train['pchg']))
        df_train['label']=list(df_train['pchg'].apply(lambda x:1 if x>mean else 0))
    else:
        # Calculate the index to be preserved
        top_index,last_index=top_last_index(len(df_train),reserved_proportion)
        # Label The top 30% of next month's profit as 1, and the last 70% as 0
        mean=df_train.iloc[top_index+1]['pchg']
        df_train['label']=list(df_train['pchg'].apply(lambda x:1 if x>mean else 0))

    dev_set_index=list(df_train['dev_index'])
    df_train.drop(['dev_index'],axis=1,inplace=True)
    # Data cleaning the training set
    df_train=preprocess(df_train,current_date)

    # Same operation with the test set
    df_test = factor_read_from_file(current_date)
    # Remove the stocks with no next month's profit
    df_test.dropna(subset=['pchg'], inplace=True)
    # Sort the dataframe by next month's profit in descending order
    df_test=df_test.sort_values(by=['pchg'],ascending=False)

    if drop:
        # Calculate the index to be preserved
        top_index,last_index=top_last_index(len(df_test),reserved_proportion)
        df_test=pd.concat([df_test.iloc[:top_index,:],df_test.iloc[last_index:,:]])
        # Label The top 30% of next month's profit as 1, and the last 30% as 0, drop the middle
        mean=np.mean(list(df_test['pchg']))
        df_test['label']=list(df_test['pchg'].apply(lambda x:1 if x>mean else 0))
    else:
        # Calculate the index to be preserved
        top_index,last_index=top_last_index(len(df_test),reserved_proportion)
        # Label The top 30% of next month's profit as 1, and the last 70% as 0
        mean=df_test.iloc[top_index+1]['pchg']
        df_test['label']=list(df_test['pchg'].apply(lambda x:1 if x>mean else 0))

    # Data cleaning the test set
    df_test=preprocess(df_test,current_date)

    # Save the cleaned dataset
    if drop:
        save_to_file([df_train,df_test,dev_set_index],current_date+'_drop','Database/dataset/'+str(interval)+'/'+str(year))
    else:
        save_to_file([df_train,df_test,dev_set_index],current_date,'Database/dataset/'+str(interval)+'/'+str(year))

    if dev_set:
        return df_train,dev_set_index,df_test
    else:
        return df_train,df_test


if __name__ == '__main__':
    pro=authorization(1)

    current_date='2015-12-31'

    df=factor_read_from_file(current_date)
    corr_heat_map(df)

    interval=20
    start_date = '2007-01-01'
    end_date = '2020-07-15'
    base_date='2015-12-31'

    date_list=get_date_list(start_date,end_date,base_date,interval=interval)

    training_set,dev_set_index,test_set=get_dataset_bydate(base_date,date_list,year=1,interval=interval,drop=True,dev_set=True)


