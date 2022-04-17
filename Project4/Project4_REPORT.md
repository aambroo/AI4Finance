# Modeling Volatility with GARCH Models
In the previous projects we used models based on linear regression to modeling time series. However, such models cannot account for volatility that is not constant over time (heteroskedasticity).

In this fourth project we focus on **conditional heteroskedasticity**, which is a phenomenon caused when an increase in volatility is correlated with a further increase in volatility.\
Understanding the meaning of **volatility** is essential to investors. It is synonimous with risk and has many other applications:
+ used in *option pricing*
+ has an impact on *risk management* $\rightarrow$ we used it to calculate a portfolio's *Sharpe ratio* in [Project 1](../Project1/CAPM%20copy.ipynb)
+ and many more applications in *trading* as well

---

ℹ️ with GARCH models there are additional constraints on coefficients with respect to 'traditional' ARCH models. In case of a GARCH(1,1) model, $\alpha_i + \beta$ must be **statistical significant**, otherwise the model is unstable. 

To assess $\alpha_i + \beta$ are statistically significant, we run an ARCH model and observe its statistical significance. In our case specifically (see *ARCH model results*) alpha is statistical significant, so we can safely run the GARCH model.

---

## 1) Plot the conditional variance and volatility of Market Excess Returns. What is the <u>mean</u> and <u>standard deviation</u> of volatility?
<!-- Conditional Volatility Plot -->
<img
    src='./outputs/conditional_volatility.png'
    alt='Conditional Volatility Plot'
    style='text-align: center'
/>

Statistics:
+ $\mu=14.455$
+ $\sigma=$ (conditional volatility) • ($\sqrt{12}$) $=3.356$

### ARCH model results:
<!-- ARCH Model Results -->
<img 
    tag='ARCH'
    src='./outputs/ARCH_model_results.png'
    style='text-align: center'
/>

### GARCH model results:
<!-- GARCH Model Results -->
<img 
    tag='GARCH'
    src='./outputs/GARCH_model_results.png'
    style='text-align: center'
/>
<!-- The result of both ARCH and GARCH models show that the alpha results are statistically significant. So, if alpha s statistical significant for the ARCH model
$$y=\alpha+\beta x + \varepsilon$$
$$y=\alpha+\beta x\varepsilon$$ -->

---

## 2) Do conditional volatility of `mkt_excess_ret` have any predictive power for recessions in a simple or dynamic probit model?
To answer this question we run a Probit model (see its definition in [Project 3](../Project3/Project3.ipynb)) using the greatest of the **conditional volatility** results for market excess return  as explanatory variable.

To answer this question we run a Probit model (defined in [Project 3](../Project3/Project3.ipynb)) using the conditional volatility value having the highest r-squared index amongst 12 different models. Like we did in Project 3, 12 probit models have been run each predicting 1 to 12 periods in advance (lags). Only the model having the highest pseudo r-squared value is kept for the analysis. In this case the model best fitting the data is the one predicting 11 quarters in advance and has a pseudo $r^2 = 0.01554$, which is not a great value.

<!-- Probit Model Output -->
<img 
    src='./outputs/3)excess_mkt_ret_summary.png'
    style='text-align: left'
/>
<!-- Probit Model Plot -->
<img 
    src='./outputs/3)excess_mkt_ret_plot.png'
    style='text-align: right'
/>

<!-- Examining the results obtained by using *conditional volatility* of market excess return as an explanatory variable, it is to be noticed the low correlation level of `cond_vol` as a variable (see $r^2=0.037$) and its p-value $= 0.812$ which further confirms it is not statistically significant. \
The plot above further shows the almost absent prediction power of market excess return's conditional volatility over recession prediction. -->

## 3) Do conditional volatility of `treasury term` have any predictive power for recessions in a simple or dynamic probit model?
To answer this question we run a Probit model (defined in [Project 3](../Project3/Project3.ipynb)) using the conditional volatility value having the highest r-squared index amongst 12 different models. 

In this case the model best fitting the data is the one predicting 1 quarter in advance and has a pseudo $r^2 = 0.1278$ which, despite being higher than the previous model, is still not a great value.

<!-- Probit Model Output -->
<img 
    src='./outputs/4)term_summary.png'
    style='text-align: left'
/>
<!-- Probit Model Plot -->
<img 
    src='./outputs/4)term_plot.png'
    style='text-align: right'
/>


## 4) Do conditional volatility of `federal funds` have any predictive power for recessions in a simple or dynamic probit model?
To answer this question we run a Probit model (defined in [Project 3](../Project3/Project3.ipynb)) using the conditional volatility value having the highest r-squared index amongst 12 different models. 

In this case the model best fitting the data is the one predicting 1 quarter in advance and has a pseudo $r^2 = 0.1625$ which, despite being higher than the previous model, is still not a great value.

<!-- Probit Model Output -->
<img 
    src='./outputs/5)fed_funds_summary.png'
    style='text-align: left'
/>
<!-- Probit Model Plot -->
<img 
    src='./outputs/5)fed_funds_plot.png'
    style='text-align: right'
/>

## Conclusions
Runs of analyses at points 3), 4) and 5) show that neither of the three variables have strong prediction power of market recession. \
However, the recipe showing the greatest prediction power is a Probit model with a 1 lag prediction (1 quarter prediction) using `federal funds` as the explanatory variable. \
Similar results might be obtained using `treasury term` as explanatory variable for the same prediction lag.