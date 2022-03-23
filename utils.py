import pandas as pd
import yfinance as yf
import pandas_datareader as pdr
import statsmodels.api as sm
import urllib
import zipfile
import os


# def get_data(ticker, interval, period='30y'):
#     '''
#     Downloads data from Yahoo! finance.
#     :param symbol: (string, list of strings) containing Stock name.
#     :param period: Time period.
#     :param interval: Frequency.
#     :return: pandas dataframe of requested data.
#     '''
#     df = yf.download(str(ticker), period=str(period), interval=str(interval), progress=False)
#     pd.DataFrame(df)
#     return df

def get_data(ticker, start, end, interval='m', period='30y'):
    '''
    Downloads data from Yahoo! finance.
    :param ticker: (string, list of strings) of Stock names.
    :param start: starting period of download.
    :param end: end period of download.
    :param interval: Frequency (`d`: `daily`; `m`: `monthly`; `y`: `yearly`)
    :return: pandas dataframe of requested data.
    '''
    for label in ticker:
        df = pdr.get_data_yahoo(ticker, start, end, interval)
    return df


def get_monthly_prices(df1, df2, label1='df1', label2='df2', column='Adj Close'):
    '''
    Function to get monthly prices of a given ticker.
    :param df1: first dataframe
    :param df2: second dataframe
    :return: monthly prices of given stocks.
    '''
    # Join closing prices of given datasets
    monthly_prices = pd.concat([df1[str(column)], df2[str(column)]], axis=1)
    monthly_prices.columns = [label1.upper(), label2.upper()]
    clean_monthly_prices = monthly_prices.dropna(axis=0)
    return clean_monthly_prices


def get_monthly_returns(df1, df2, label1='df1', label2='df2', column='Adj Close'):
    '''
    Function to get monthly returns of a given ticker.
    :param df1: first dataframe
    :param df2: second dataframe
    :return: monthly returns of given stocks.
    '''
    # Join closing prices of given datasets
    monthly_prices = pd.concat([df1[str(column)], df2[str(column)]], axis=1)
    monthly_prices.columns = [label1.upper(), label2.upper()]

    # Calculate monthly returns
    monthly_returns = monthly_prices.pct_change(1)
    clean_monthly_returns = monthly_returns.dropna(axis=0)
    return clean_monthly_returns

def linear_regression(monthly_returns, label1='df1', label2='df2'):
    '''
    :param monthly_returns: monthly_returns database as output of get_monthly_returns function.
    :return: OLS model
    '''
    # Split dependent and independent vars
    X = monthly_returns[label2.upper()]
    y = monthly_returns[label1.upper()]

    # Constant definition
    X1 = sm.add_constant(X)

    # Regression Model
    model = sm.OLS(y, X1)
    return model

def get_risk_free (url, dst):
    '''
    Retreive data from given `url`, save and extract data to `dst`
    :param url: `string` of source url.
    :param dst: `string` of destination directory.
    :return: `None`
    '''
    urllib.request.urlretrieve(url, dst)
    # check for save_dir existence
    save_path = os.path.join(os.getcwd(), 'data')
    PATH_TO_DATA = save_path
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    # extract and delete source file
    with zipfile.ZipFile(dst, 'r') as zip_ref:
        zip_ref.extractall(save_path)
        os.remove(dst)
    return save_path

# LINEAR REGRESSION
def get_summary (budget, dataframe, x_label, y_label):
    '''
    Performs Linear Regression on a dataset (or list of datasets).
    :param dataframe: Dataset (or list of datasets).
    :param x_label (string): Column name to be used as dependent variable.
    :param y_label (string): Column name to be used as independent variable.
    :return: Returns a pandas dataframe.
    '''
    avg_label = (f'Avg_{y_label}%')
    labels = {  'Interval' : ['30Y','15Y','5Y'],
                    avg_label           : [],
                    'Sharpe Ratio'      : [],
                    'Alpha'             : [],
                    'Beta'              : [],
                    'R-squared'         : [],
                    't-value (alpha)'  : [],
                    'p-value (alpha)'  : [],
                    't-value (beta)'   : [],
                    'p-value (beta)'   : [],
                    'Excess Return'     : []}
    # OLS Linear Regression
    for df in dataframe:
        x = df[x_label]
        y = df[y_label]
        X = sm.add_constant(x)

        model = sm.OLS(y, X).fit()

        avg_excess_ret = y.mean()
        sharpe = y.mean() / y.std()

        labels[avg_label].append(f"{(avg_excess_ret * 100).round(4)}%")
        labels['Sharpe Ratio'].append(sharpe.round(4))
        labels['Alpha'].append(model.params['const'].round(4))
        labels['Beta'].append(model.params[1].round(4))
        labels['R-squared'].append(model.rsquared.round(4))
        labels['t-value (alpha)'].append(model.tvalues[0].round(4))
        labels['p-value (alpha)'].append(model.pvalues[0].round(4))
        labels['t-value (beta)'].append(model.tvalues[1].round(4))
        labels['p-value (beta)'].append(model.pvalues[1].round(4))
        labels['Excess Return'].append(f'{avg_excess_ret * budget:,.2f}$')
    summary = pd.DataFrame(labels)
    return summary