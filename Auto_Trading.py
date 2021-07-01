# Preliminary exploration of the automated trading

# It may not be run directly after installing easytrader,
# the source code of the easytrader has been modified a lot to adapt to the current version of the client.

import easytrader

# Predefined dictionary to translate the returned information from Chinese into English
defined_dict={
    '资金余额':'Fund balance',
    '可用金额':'Available balance',
    '可取金额':'Withdrawable balance',
    '股票市值':'Stock market value',
    '总资产':'Total balance',

    '股票余额':'Stock Balance',
    '可用余额':'Balance avaliable',
    '冻结数量':'Frozen quantity',
    '参考盈亏':'Reference profit',
    '成本价':'Cost price',
    '盈亏比例(%)':'Profit and loss (%)',
    '市价':'Market price',
    '市值':'Market value',
    '交易市场':'Trading market',
    '历史成交':'Historical transaction',
    '个股仓位(%)':'Stock position (%)',

    '成交时间':'Closing time',
    '证券代码':'Code',
    '证券名称':'Name',
    '操作':'Operation',
    '成交数量':'Transaction quantity',
    '成交均价':'Average transaction price',
    '成交金额':'Transaction amount',
    '合同编号':'Contract number',
    '成交编号':'Transaction number',

    '委托时间':'Entrusting time',
    '成交状态':'Closing status',
    '委托数量':'Entrusted quantity',
    '委托价格':'Entrusted price',
    '委托状态':'Entrusted status',
    '委托子业务':'Entrusted sub-business',
    '约定号':'Agreement number',
    '席位号':'Seat number',
    '备注':'Comments',
    '对方账号':'Front account',
    '申报编号':'Declaration number',

    '买':'Buy',
    '卖':'Sell',
    '华宝油气':'Huabao petroleum gas',
    '已成交':'Done',
    '场外撤单':'Off-site cancellation',
    '正常委托':'Normal',
    '深圳Ａ股':'Shenzhen A-shares'
}

# Function to translate the object from Chinese into English
def ch_to_eng(object):
    li=[]

    if isinstance(object,dict):
        object=[object]

    for dic in object:
        for key in list(defined_dict.keys()):
            if key in dic:
                if dic[key] in defined_dict.keys():
                    dic[key]=defined_dict[dic[key]]
                dic[defined_dict[key]]=dic.pop(key)

        dic_copy=dic.copy()
        for key in dic:
            if 'Unnamed' in key:
                dic_copy.pop(key)

        li.append(dic_copy)

    return li

# Connect to the Tonghuashun Trading Client
user = easytrader.use('ths')
# Path where trading client is located
client_path=r'D:\Soft\东方同花顺金融终端\xiadan.exe'
user.connect(client_path)
user.refresh()

# Query the account balance
print(ch_to_eng(user.balance))

# Query the account position
print(ch_to_eng(user.position))

# Buy the designated stock
print(user.buy('162411', price=0.20, amount=100))

# Sell the designated stock
print(user.sell('162411', price=0.20, amount=100))

# Cancel the entrustment that has been submitted with the specified entrust number
print(user.cancel_entrust('="220"'))

# Query today's trades
print(ch_to_eng(user.today_trades))

# Query today's entrusts
print(ch_to_eng(user.today_entrusts))



