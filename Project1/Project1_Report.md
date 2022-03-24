# Table of Content <>

- [Table of Content <>](#table-of-content-)
- [Some  Underlying Theory](#some--underlying-theory)
  - [Capital Asset Pricing Model (CAPM)](#capital-asset-pricing-model-capm)
    - [Formula](#formula)
    - [Additional Resources](#additional-resources)
- [1. Individual Stock](#1-individual-stock)
  - [1.1 Risk Free Return](#11-risk-free-return)
- [2. Multi-Stock Portfolio](#2-multi-stock-portfolio)
- [3. Portfolio Analysis](#3-portfolio-analysis)
  - [3.1. CAPM Analysis](#31-capm-analysis)
      - [Beta](#beta)
      - [R-Squared](#r-squared)
      - [P-value and t-value Interpretation](#p-value-and-t-value-interpretation)
  - [3.2. Single Stock Portfolio CAPM Analysis](#32-single-stock-portfolio-capm-analysis)
  - [3.3. Multi Stock Portfolio CAPM Analysis (benchmark: S&P500)](#33-multi-stock-portfolio-capm-analysis-benchmark-sp500)
  - [3.4. Multi Stock Portfolio CAPM Analysis (benchmark: Fama&French)](#34-multi-stock-portfolio-capm-analysis-benchmark-famafrench)
- [4. Trend Analysis during the Financial Crisis of 2008•2009](#4-trend-analysis-during-the-financial-crisis-of-20082009)



# Some  Underlying Theory
## Capital Asset Pricing Model (CAPM)
- Relationship between systematic risk and expected return
- There are several assumptions behinf CAPM formula that have shown not to always hold in reality
- CAPM formula is still widely used

### Formula
$ER_i = R_f + \beta_i\left(ER_m-R_f\right)$
- $ER_i$: expected return from investment
- $R_f$: risk-free return
- $\beta_i$: the beta of the investment (correlation between systematic risk and ____)
- $\left(ER_m-R_f\right)$: market risk premium

We assume as **risk free return** the 10 year Treasury Note available at this [link](https://www.treasury.gov/resource-center/data-center/interest-rates/pages/textview.aspx?data=yield).

**Market risk premium** 

### Additional Resources
- CAPM at https://www.investopedia.com/terms/c/capm.asp

# 1. Individual Stock

## 1.1 Risk Free Return

In this section data from [Fama&French](http://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html) will be used as benchmark.
If you are willing to run the original notebook, please download the csv file and save it at `./data/F-F_Research_Data_Factors_CSV.zip`.\
Here is the code used to gather the data:

```python
# RISK-FREE RETURN FRENCH
dateparse = lambda x: dt.datetime.strptime(x,'%Y%m')
fama_french_df = pd.read_csv('./data/F-F_Research_Data_Factors.CSV',
                        header=0,
                        names=['Date','Mkt-RF','SMB','HML','RF'],
                        parse_dates=['Date'], date_parser=dateparse,
                        index_col=0,
                        skipfooter=99,
                        skiprows=3,
                        engine='python')

# Rf
risk_free_rets = fama_french_df['RF'].loc[fama_french_df.index >= '1992-01-01']/100
# (Rm - Rf)
market_risk_premium = fama_french_df['Mkt-RF'].loc[fama_french_df.index >= '1992-01-01']/100
```
Answers to project questions can be found in the last section of this file.

# 2. Multi-Stock Portfolio

Let's consider the following Multi-Stocks portfolio containing $10$ separate stocks:

| **Ticker**  | Company         | Branch        |
| :---:       |    :----:       |    :---:      |
| **AAPL**    | Apple           | tech          |
| **MSFT**    | Microsoft       | tech          |
| **NFLX**    | Netflix         | entertainment |
| **IBM**     | IBM             | tech          |
| **BTC-USD** | Bitcoin         | crypto        |
| **2222.SR** | Aramco          | oil           |
| **BA**      | Boeing          | transports    |
| **DIS**     | WaltDisney      | entertainment |
| **DAL**     | DeltaAirlines   | transports    |
| **LYFT**    | Lyft            | transports    |


# 3. Portfolio Analysis

## 3.1. CAPM Analysis

The stock asset in analysis is **Apple Inc.**, with stock ticker `AAPL`.

---

**CAPM** (**C**apital **A**sset **P**ricing **M**odel) analysis attempts to analyze the existing relationship that exists between expected returns and risk. The model implies that the analysis is performed combining _at least_ two types of assets or securities: a _risk-free_ security as benchmark, and a _risky_ asset. CAPM further posits that investors expect to be rewarded for holding such the abovementioned risky asset(s) according to the risk inherited for holding on to such assets. After all, market-related risk (**systematic risk**) cannot be diversified.

#### Beta

---

The **beta** of a stock asset is a measure of the sensitivity of its returns relative to a market benchmark (in our case we consider the S&P500 index as our benchmark). Beta is computed as follows:
$$\beta=\frac{cov(R_s-R_f)}{var(R_f)}$$
At the nominator is the variance of the difference between stock return and risk-free return of the market ($R_s-R_f$); at the denominator is the variance of the fisk-free market return.\
The interpretation of $\beta$ should be as follows:
- $\beta=0$ _indicates that the analyzed stock has no correlation with the chosen benchmark_
- $\beta=1$ _indicates a stock having the very same volatility as the market_
- $\beta>1$ _indicates a stock that is more volatile than its benchmark_
- $\beta<1$ _indicates a stock that is less volatile than its benchmark_
- $\beta=1.5$ for instance, means that _the stock asset is exactly $50%$ more volatile than the benchmark_

#### R-Squared

---

**R-Squared** ($R^2$), also called _coefficient of determinaiton_, determines the proportion of variance in the dependent variable (in this case the _S&P500 index_) that can be explained by the independent variable (in this case the _stock asset_). Simply put, R-Squared shows how well the data fit the regression model.\
$$R^2 = 1 - \frac{\text{unexplained variation}}{\text{total variation}}$$

#### P-value and t-value Interpretation

---
Alpha and Beta's statistical significance is ensured by looking at the confidence velues of both _p-value_ and _t-value_.
- _p-value_ $\le 0.05$ means that there is strong evidence against the null hypothesis. Hence, results are random.
- _p-value_ $> 0.05$ means that there is strong evidence in favour of the null hypothesis. Simply put, the result of what we are trying to prove has a confidence of $95\%$.

Similarly, we would like the _t-value_ to be at least $\sim 1.6$

## 3.2. Single Stock Portfolio CAPM Analysis

---

Yield data is compared with the beta value available at the Yahoo! Finance [page](https://finance.yahoo.com/quote/AAPL/history?p=AAPL). Yahoo provides a beta value for Apple of $1.19$. Yahoo! does not say what is the period used to calculate $\beta$. However, the beta value we get from our own analysis is pretty much consistent with $1.19$ (see table below).\
The **alpha** value yield by the regression analysis ranges between $0.0139$ (for the 5Y interval), and $0.0142$ (for the 15Y interval). Unfortunately Yahoo! Finance provides no alpha value to use as proof correctness. 
Let's then get proof of what obtained by observing the indicators.\
The **R-squared** value for our regression analysis varies a lot depending on the considered interval (see table below). For considering a regression as reliable, the model should have $R^2\sim 0.40$. As a result, the $30Y$ interval is not as well explained as the most recent data.\
**T-values** for $\alpha$ range between $2.2685$ and $3.3718$.\
**P-values** for $\alpha$ range between $0.09\%$ and $2.58\%$. The highest values of both indicators point towards the 15Y interval.\




<center>

| Interval  | Alpha     | Beta      | R-Squared |
| :---:     | :---:     | :---:     | :---:     |
| 30Y       | $0.0142$  | $1.2700$  | $0.1876$  |
| 15Y       | $0.0160$  | $1.1935$  | $0.3512$  |
| 5Y        | $0.0139$  | $1.2170$  | $0.4109$  |

</center>



## 3.3. Multi Stock Portfolio CAPM Analysis (benchmark: S&P500)

---

As for the previous analysis, S&P500 benchmark is be used in the current portfolio analysis.\
\
Looking at the statistical indexes first, **R-Squared** values are higher for the multiple-stock analysis with respect to the single-stock analysis. This is mainly due to *diversification*. The term 'diversification' indicates the process of investing money in different asset classes and securities in order to minimize the overall risk of the portfolio.\
R-Squared also explains the **systematic risk** (computed as $\alpha + \beta \times x$). Systematic risk refers to the risk inherited to the entire market (or a market segment). $(1-R^2)$ is called **idiosyncratic risk**, and refers to the inherent factors that can negatively impact individual securities or a very specific group of assets. Idiosyncratic risk is also positively affected by diversification, and this is mainly due to the different sectors each stock asset operates in. \
\
_T-value_ shows greater statistical significance of beta than alpha. It is worth highlighting that t-values decrease whenever the analysis interval is shortened.\
Similarly, _p-value_ shows greater statistical significance for beta over alpha, especially for the 5Y interval. _P-values_ of alpha are $0$ for each and every interval of analysis.

<center>

| Interval  | Alpha     | Beta      | R-Squared |
| :---:     | :---:     | :---:     | :---:     |
| 30Y       | $0.0059$  | $0.7656$  | $0.5589$  |
| 15Y       | $0.0085$  | $0.8507$  | $0.5846$  |
| 5Y        | $0.0056$  | $1.1044$  | $0.6235$  |

</center>


## 3.4. Multi Stock Portfolio CAPM Analysis (benchmark: Fama&French)

---

In this second portfolio analysis, Fama&French benchmark is being used.\
\
$R^2$ values as well as its counterpart $(1-R^2)$ are consistent with the results yielded by the previous analysis.\
\
Both beta and alpha values are consistent with the previous analysis, as well as the confidence values that explain them.

<center>

| Interval  | Alpha     | Beta      | R-Squared |
| :---:     | :---:     | :---:     | :---:     |
| 30Y       | $0.0048$  | $0.7464$  | $0.5692$  |
| 15Y       | $0.0072$  | $0.8378$  | $0.6041$  |
| 5Y        | $0.0048$  | $1.0770$  | $0.6518$  |

</center>


# 4. Trend Analysis during the Financial Crisis of 2008•2009

Compared to the market, the _single-stock_ portfolio shows an average excess return (loss) of $-1.65$ percentage points. A return which, if compared to the overall market loss of $5.29$ percentage points, can be considered to be a good one. In particular, a portfolio of such type performed $3.64$% better than the market.\

<center>

| Asset                               | **Reward** / <span style="color:red"> **(Loss)** </span>  |
| :---:                               | :---:                                                     |
| Market Excess Return (avg)          | <span style="color:red"> $(5.285)$% </span>           |
| Return of Single Stock Portfolio (avg) | <span style="color:red"> $(1.647)$% </span>           |

</center>


I also carried out a parallel analysis considering the portfolio to be composed of initially $20\%$, and then $40\%$ of risk-free assets.\
\
Let's first consider the first setting ($20$% Risk-free):\
the reweighted portfolio shows a loss of $1.318$% compared to the non-reweighted portfolio of $(1.647)$%, which is not a massive difference (only $0.33$% better).\
Intuitively, a greater amount of risk-free assets in our portfolio, shows a lower loss during the financial crisis timeframe. (More precise statistics are listed in the table below).\

<center>

| Asset                               | **Reward** / <span style="color:red"> **(Loss)** </span>  |
| :---:                               | :---:                                                     |
| Market Excess Return (avg)          | <span style="color:red"> $(5.285)$% </span>           |
| $20\%$ Risk-free Portfolio (avg)    | <span style="color:red"> $(1.318)$% </span>           |
| $40\%$ Risk-free Portfolio (avg)    | <span style="color:red"> $(0.9882)$% </span>           |

</center>


![FC2008_2009](FinancialCrisis2008-2009.png)
*The graph shows the trend of the portfolio during the 2008•2009 crisis.*

\
Overall, the higher the percentage of non-risky assets in our portfolio, the lower the risk of loss. However, the lower the risk, the lower the potential reward.

