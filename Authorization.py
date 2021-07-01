# Load the account and the password of each quantitative financial database for obtaining data

Tushare_token='your Tushare token'
Joinquant_account_lists=[
        ['your Joinquant username1','your Joinquant password1'],
        ['your Joinquant username2','your Joinquant password2']
    ]

import warnings
warnings.filterwarnings('ignore')

import tushare as ts
from jqdatasdk import *

# Function to get authorization from Joinquant and Tushare
def authorization(index=0):
    account_list=Joinquant_account_lists
    auth(account_list[index][0], account_list[index][1])
    token = Tushare_token
    pro = ts.pro_api(token)
    return pro

# Function to get authorization from Joinquant
def authorization_jq(index=0):
    account_list=Joinquant_account_lists
    auth(account_list[index][0], account_list[index][1])

if __name__ == '__main__':
    authorization()