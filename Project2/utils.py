# IMPORTS
import pandas as pd
import statsmodels.formula.api as smf

# FUNCTIONS
def one_factor_model (datasets):
    '''
    :params datasets: (Pandas Dataframe) or (list of Pandas DataFrames)
    :return summary: (Pandas Dataframe) containing the summary of the ols regression analysis.
    '''
    labels = {
            'interval'  : ['20Y', '10Y', '5Y'],
            'r^2'       : [],
            'alpha'     : [],
            't-alpha'   : [],
            'p-alpha'   : [],
            'beta'      : [],
            't-beta'    : [],
            'p-beta'    : []
            }
    for df in datasets:
        ff_model = smf.ols(
            formula = 'pf_rets ~ spx_rets',
            data = df
        ).fit()
        
        labels['r^2'].append(ff_model.rsquared.round(4))
        labels['alpha'].append(ff_model.params[0].round(4))
        labels['t-alpha'].append(ff_model.tvalues[0].round(4))
        labels['p-alpha'].append(ff_model.pvalues[0].round(4))
        labels['beta'].append(ff_model.params[1].round(4))
        labels['t-beta'].append(ff_model.tvalues[1].round(4))
        labels['p-beta'].append(ff_model.pvalues[1].round(4))
        #print(results.summary())
    summary = pd.DataFrame(labels)
    return summary



def three_factor_model (datasets):
    '''
    Takes as input a database (or list of databases) and performs linear regression taking as variables:
        - `mkt`: excess return of the market.
        - `smb`: excess return of stocks with a small market cap over those with a large market cap.
        - `hml`: excess return of value stocks over growth stocks.
        - `excess_rtn`: (Ra - Rf) = asset return - risk-free return.
    :params datasets: (Pandas Dataframe) or (list of Pandas DataFrames)
    :return summary: (Pandas Dataframe) containing the summary of the ols regression analysis.
    '''
    labels = {
            'interval': ['30Y', '20Y', '10Y', '5Y'],
            'alpha': [],
            't-alpha': [],
            'p-alpha': [],
            'mkt': [],
            't-mkt': [],
            'p-mkt': [],
            'smb': [],
            't-smb': [],
            'p-smb': [],
            'hml': [],
            't-hml': [],
            'p-hml': [],
            }
    for df in datasets:
        ff_model = smf.ols(
            formula = 'excess_rtn ~ mkt + smb + hml',
            data = df
        ).fit()
        
        labels['alpha'].append(ff_model.params[0].round(4))
        labels['t-alpha'].append(ff_model.tvalues[0].round(4))
        labels['p-alpha'].append(ff_model.pvalues[0].round(4))
        labels['mkt'].append(ff_model.params[1].round(4))
        labels['t-mkt'].append(ff_model.tvalues[1].round(4))
        labels['p-mkt'].append(ff_model.pvalues[1].round(4))
        labels['smb'].append(ff_model.params[2].round(4))
        labels['t-smb'].append(ff_model.tvalues[2].round(4))
        labels['p-smb'].append(ff_model.pvalues[2].round(4))
        labels['hml'].append(ff_model.params[3].round(4))
        labels['t-hml'].append(ff_model.tvalues[3].round(4))
        labels['p-hml'].append(ff_model.pvalues[3].round(4))
        #print(results.summary())
    summary = pd.DataFrame(labels)
    return summary



def five_factor_model (datasets):
    '''
    Takes as input a database (or list of databases) and performs linear regression taking as variables:
        - `mkt`: excess return of the market.
        - `smb`: excess return of stocks with a small market cap over those with a large market cap.
        - `hml`: excess return of value stocks over growth stocks.
        - `excess_rtn`: (Ra - Rf) = asset return - risk-free return.
    :params datasets: (Pandas Dataframe) or (list of Pandas DataFrames)
    :return summary: (Pandas Dataframe) containing the summary of the ols regression analysis.
    '''
    labels = {
            'interval': ['30Y', '20Y', '10Y', '5Y'],
            'alpha': [],
            't-alpha': [],
            'p-alpha': [],
            'mkt': [],
            't-mkt': [],
            'p-mkt': [],
            'smb': [],
            't-smb': [],
            'p-smb': [],
            'hml': [],
            't-hml': [],
            'p-hml': [],
            'term': [],
            't-term': [],
            'p-term': [],
            'credit': [],
            't-credit': [],
            'p-credit': [],
            }
    for df in datasets:
        ff_model = smf.ols(
            formula = 'excess_rtn ~ mkt + smb + hml + term + credit',
            data = df
        ).fit()
        
        labels['alpha'].append(ff_model.params[0].round(4))
        labels['t-alpha'].append(ff_model.tvalues[0].round(4))
        labels['p-alpha'].append(ff_model.pvalues[0].round(4))
        labels['mkt'].append(ff_model.params[1].round(4))
        labels['t-mkt'].append(ff_model.tvalues[1].round(4))
        labels['p-mkt'].append(ff_model.pvalues[1].round(4))
        labels['smb'].append(ff_model.params[2].round(4))
        labels['t-smb'].append(ff_model.tvalues[2].round(4))
        labels['p-smb'].append(ff_model.pvalues[2].round(4))
        labels['hml'].append(ff_model.params[3].round(4))
        labels['t-hml'].append(ff_model.tvalues[3].round(4))
        labels['p-hml'].append(ff_model.pvalues[3].round(4))
        labels['term'].append(ff_model.params[4].round(4))
        labels['t-term'].append(ff_model.tvalues[4].round(4))
        labels['p-term'].append(ff_model.pvalues[4].round(4))
        labels['credit'].append(ff_model.params[5].round(4))
        labels['t-credit'].append(ff_model.tvalues[5].round(4))
        labels['p-credit'].append(ff_model.pvalues[5].round(4))
        #print(results.summary())
    summary = pd.DataFrame(labels)
    return summary
# summary = three_factor_model(datasets)
# summary

