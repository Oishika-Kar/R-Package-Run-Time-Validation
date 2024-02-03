# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: title,-all
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
"""
# R Packages Run Time Validation Analysis
"""
# %% [markdown]
"""
**Economics 430: Project 2**
"""
# %% [markdown]
"""
**Fall 2023**
"""
# %% [markdown]
"""
**Group Members:**
- Mauricio Vargas Estrada
- Luis Alejandro Samayoa Alvarado
- Sultan Al Balushi
- Oishika Kar
"""
# %% [markdown]
# <h2 id="index">Index</h2>
# %% [markdown]
"""

1. <a href="#introduction">Introduction</a>
2. <a href="#source-of-the-data">Source of the Data</a>
3. <a href="#variable-selection">Variable Selection</a>
   - <a href="#boruta-algorithm">Boruta Algorithm</a>
4. <a href="#descriptive-analysis">Descriptive Analysis</a>
   - <a href="#unusual-observations-identification-and-removal">Unusual Observations Identification and Removal</a>
   - <a href="#transformation-of-variables">Transformation of Variables</a>
   - <a href="#descriptive-statistics">Descriptive Statistics</a>
   - <a href="#quantile-quantile-histograms-and-density-plots">Quantile-Quantile, Histograms, and Density Plots</a>
   - <a href="#correlation-and-pairplots">Correlation and Pairplots</a>
5. <a href="#model-selection">Model Selection</a>
   - <a href="#comparison-between-feasible-models">Comparison Between Feasible Models</a>
       - <a href="#model-with-untransformed-variables">Model with Untransformed Variables</a>
       - <a href="#model-with-optimal-yeo-johnson-transformation-for-each-variable">Model With Optimal Yeo-Johnson Transformation for each Variable</a>
       - <a href="#model-with-selected-transformation-for-each-variable">Model With Selected Transformation for each Variable</a>
6. <a href="#analysis-of-the-selected-model">Analysis of the Selected Model</a>
   - <a href="#robustness-analysis">Robustness Analysis</a>
   - <a href="#cross-validation">Cross-Validation</a>
   - <a href="#marginal-effects-analysis-and-parameter-interpretation">Marginal Effects Analysis and Parameter Interpretation</a>
"""
# %% [markdown]
"""
--- 

<h1 id="introduction">Introduction </h1>
<a href="#index">Return to Index</a>

When an R package developer wishes to publish their package on CRAN, they must undergo a validation process. This process involves a series of tests conducted on the package to verify that it complies with CRAN standards. One of these tests is cross-validation, which involves running the `R CMD check` command on different platforms. The time it takes to execute this command is recorded in the `check_times.csv` file, which contains data from over 13,000 packages, including the respective execution time on each platform and the characteristics of each package, such as the number of dependencies, the number of files in the `src` directory, the number of files in the `data` directory, etc.
"""
# %% [markdown]
"""
The objective of this project is to analyze the different characteristics of R packages and determine which of these characteristics influence the execution time of the `R CMD check` command. Additionally, it aims to determine if it is possible to predict the execution time of this command using the package's characteristics.    
"""
# %% [markdown]
"""
<h2 id="source-of-the-data">Source of the Data</h2>

<a href="#index">Return to Index</a>

The dataset is part of the R `tidymodels` package and can be obtained with the following code:    

```r
library(tidymodels)
data("check_times")
```
"""
# %% [markdown]
"""
The dataset is accessible in the following GitHub repository:  

- [https://github.com/vincentarelbundock/Rdatasets](https://github.com/vincentarelbundock/Rdatasets)

The complete list of variables can be found at the following link:  

-[https://vincentarelbundock.github.io/Rdatasets/doc/modeldata/check_times.html](https://vincentarelbundock.github.io/Rdatasets/doc/modeldata/check_times.html)  
"""
# %% [markdown]
"""
Throughout this project, the selected from the above mentioned list for our analysis are:  

- `check_time` (dependent): The time (in seconds) to run ⁠R CMD check⁠ using the "r-devel-windows-ix86+x86_64' flavor.
- `depends`: The number of hard dependencies.
- `imports`: The number of imported packages.
- `r_size`: The total disk size of the R files.
- `src_size`: The size on disk of files in the src directory.
- `doc_size`: The disk size of the Rmd or Rnw files.
- `data_size`: The size on disk of files in the data directory.
- `Roxygen`: a binary indicator for whether Roxygen was used for documentation.
- `gh`: a binary indicator for whether the URL field contained a GitHub link.
"""
# %% [markdown]
"""
> Arel-Bundock, V. (n.d.). *Rdatasets: A collection of datasets originally distributed in R packages*. GitHub. Retrieved November 18, 2023, from [https://github.com/vincentarelbundock/Rdatasets](https://github.com/vincentarelbundock/Rdatasets)
"""
# %% [markdown]
"""
> Clarification Note:
>
> Many of the functions used in this project are part of the `src` package located in the folder of the same name in this repository. The package is of our authorship and is used to facilitate the analysis throughout the project.
"""

# %%
# ----- Packages -----
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import os
import numpy as np

import statsmodels.api as sm
import statsmodels.formula.api as smf
# ----- Self-made Modules -----
from src.bootstrap_estimation import bootstrap_estimation
from src.bootstrap_yeojohnson import bootstrap_yeojohnson
from src.boruta_parallel import boruta_parallel
from src.cross_validation import cross_validation
from src.plot_diagnostic_plots import plot_diagnostic_plots
from src.FGLS_weights import FGLS_weights
from src.het_breuschpagan import het_breuschpagan
from src.influential_plot import influential_plot
from src.iterative_reset_omitted import iterative_reset_omitted
from src.mosaic_boot import mosaic_boot
from src.mosaic_hist import mosaic_hist
from src.mosaic_qq import mosaic_qq
from src.parallel_yeojohnson import parallel_yeojohnson
from src.ramsey_reset_test import ramsey_reset_test
from src.VIF_X import VIF_X
from src.generate_boxplots import generate_boxplots

# Turn off future warnings

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# %%
# ----- Loading dataset -----
data = pd.read_csv(
    os.path.join('data', 'check_times.csv')
)

# ----- Subset of variables -----
# Factor or category variables
col_cat = [
    'Roxygen',
    'gh',
]

# Dependent variables
col_dependent = ['check_time']

# Numerical variables
col_num = [
    col for col in data.columns if col not in col_cat + col_dependent
]
# %% [markdown]
"""

---

<h1 id="variable-selection">Variable Selection</h1>

<a href="#index">Return to Index</a>

<h2 id="boruta-algorithm">Boruta Algorithm</h2>
<a href="#index">Return to Index</a>
"""

# %%
# %config Completer.use_jedi = False

boruta_results = boruta_parallel(
    data = data,
    y_name = col_dependent[0]
)
# %%
boruta_df = boruta_results[0]
# %% [markdown]
"""
According to the Boruta analysis, the variables classified as category 2 are those that show greater importance with respect to the shadow characteristics.

To complement the variable selection, the absolute correlation matrix (percentage) is calculated. This allows for the selection of variables with high importance and low correlation between them.
"""
# %%
abs_corr = data[col_dependent + list(boruta_df.index)].corr().abs()
abs_corr = abs_corr * 100
abs_corr = abs_corr.round(2)

abs_corr
# %% [markdown]
"""
Under this criterion, the selected variables are:
- `check_time` (dependent): The time (in seconds) to run ⁠R CMD check⁠ using the "r-devel-windows-ix86+x86_64' flavor.
- `depends`: The number of hard dependencies.
- `imports`: The number of imported packages.
- `r_size`: The total disk size of the R files.
- `src_size`: The size on disk of files in the src directory.
- `doc_size`: The disk size of the Rmd or Rnw files.
- `data_size`: The size on disk of files in the data directory.
"""
# %% [markdown]
"""
<h2 id="factor-variables-selection">Factor Variables Selection</h2>

<a href="#index">Return to Index</a>

In the original dataset, the following categorical variables are found:
- `Roxygen`: a binary indicator for whether Roxygen was used for documentation.
- `gh`: a binary indicator for whether the URL field contained a GitHub link.
- `status`: An indicator for whether the tests completed.
- `rforge`: a binary indicator for whether the URL field contained a link to R-forge.

Of these categorical variables, `status` is used to eliminate those packages that did not complete the validation test. This means that this variable is removed from the dataset for the model because it would only contain values of 1.

Since all the remaining packages contain links to R-forge, this variable is also removed from the dataset.

Consequently, the dataset used in subsequent sections will contain only the categorical variables `Roxygen` and `gh`.
"""

# %%
# ----- Subset of variables -----
col_num = ['depends', 'imports', 'r_size', 'src_size', 'doc_size', 'data_size']
# %% [markdown]
"""

---

<h1 id="descriptive-analysis">Descriptive Analysis</h1>
<a href="#index">Return to Index</a>
"""
# %% [markdown]
"""
<h2 id="unusual-observations-identification-and-removal">Unusual Observations Identification and Removal</h2>

<a href="#index">Return to Index</a>



For the selection of unusual observations, an auxiliary model is defined using the variables selected in the previous section. Subsequently, studentized residuals, leverage, and Cook's distance are used to classify observations as unusual. Finally, the unusual observations are eliminated, and the model is re-estimated.
"""
# %%
# ----- Subsetting the dataset -----
data = data[col_dependent + col_num + col_cat]
# %%
# ----- Constructing the linear model string -----
col_independent = col_num + col_cat
lm_string = col_dependent[0] + '~'
for i in range(len(col_independent)):
    if i == 0:
        lm_string = lm_string + col_independent[i]
    else:
        lm_string = lm_string + '+' + col_independent[i]

# %%
# ----- Auxiliary model -----
model_aux = smf.ols(lm_string, data = data).fit()
# %%
influential_plot(model_aux)

# %%
# ----- Extracting unusual observations measures -----
residual_measures = data.copy()
residual_measures['studentized_residuals'] = model_aux.get_influence().resid_studentized_internal
residual_measures['abs_studentized_residuals'] = residual_measures['studentized_residuals'].abs()
residual_measures['leverage'] = model_aux.get_influence().hat_matrix_diag
residual_measures['cooks_distance'] = model_aux.get_influence().cooks_distance[0]
residual_measures['abs_leverage'] = residual_measures['leverage'].abs()
mean_leverage = residual_measures['leverage'].mean()
# %%
# Checking how many observations are not outliers.
residual_measures['not_outlier'] = residual_measures.abs_studentized_residuals < 2
# Checking how many observations are not leverage points.
residual_measures['not_leverage'] = residual_measures.leverage < 2 * mean_leverage
# Checking how many observations are not influential points.
residual_measures['not_influential'] = residual_measures.cooks_distance < 4 / len(data)
# Points that are not outliers, leverage points and influential points.
residual_measures['good_observation'] = residual_measures.not_outlier & residual_measures.not_leverage & residual_measures.not_influential
# %%
# ----- Filtering the dataset -----
data_filtered = data[residual_measures.good_observation]
data_filtered = data_filtered.reset_index(drop=True)

# %%
# ----- New auxiliary model without unusual observations -----
model_aux_filtered = smf.ols(lm_string, data = data_filtered).fit()
# %%
influential_plot(model_aux_filtered)
# %%
plot_diagnostic_plots(model_aux)
# %%
plot_diagnostic_plots(model_aux_filtered)

# %% [markdown]
# To find and eliminate unusual observations, we start by examining them with an influential plot, focusing on high leverage, extreme studentized residuals, and significant Cook's distances. We perform these steps to enhance the robustness and reliability of the regression model. This is because the outliers can compromise the predictability and reliability of statistical inferences. We use certain thresholds to determine unusual observations: 2 for studentized residuals, 2 times the mean for leverage, and 4 divided by the number of observations for Cook's Distance. After eliminating this unusual observation, the stability of the parameters of the model will increase.
#
# **Thresholds**
#
# * Studentized Residuals: $± 2$
# * Leverage: $2 * \overline{leverage}$
# * Cook' Distance: $\frac{4}{Nobs}$
# %% [markdown]
"""
<h2 id="transformation-of-variables">Transformation of Variables</h2>
<a href="#index">Return to Index</a>
"""
# %% [markdown]
r"""
As part of the procedures to analyze possible transformations, a Yeo-Johnson transformation was applied to each of the variables, calculating a bootstrap interval with a 95% confidence level for the transformation parameter $\lambda$.
"""
# %%
# Calculating the optimal yeo-johnson transformation for each numeric variable. Also, the lower and upper bounds at 95% confidence interval.
data_yeo_filtered, lambda_yeo = parallel_yeojohnson(
    data_filtered[col_dependent + col_num]
)
data_yeo_filtered[col_cat] = data_filtered[col_cat]
# %%
# ----- Analyzing the optimal yeo-johnson transformation for each variable -----
lambda_yeo.T
# %% [markdown]
r"""
In the previous table, the different values of lambda for each variable can be observed. From these estimations, the lambda values will be rounded in such a way that the value falls within the confidence interval.

In particular, `check_time` will be transformed with $\lambda = 0$ because this value is close to the confidence interval and provides properties that improve the interpretation of the parameters. This will be analyzed in more detail in the model selection section.

Additionally, the residuals are analyzed through an auxiliary regression, as was done in the previous section. The objective is to identify problems in a regression using the optimal Yeo-Johnson transformations.
"""
# %%
model_aux_filtered_yeo = smf.ols(lm_string, data = data_yeo_filtered).fit()

# %%
influential_plot(model_aux_filtered_yeo)

# %%
plot_diagnostic_plots(model_aux_filtered_yeo)

# %% [markdown]
"""
It can be observed that an appropriate transformation of variables following the filtering process results in a more robust dataset with stronger linear relationships. This is a clear indication of the need for variable transformation for our final model.
"""

# %% [markdown]
"""
<h2 id="descriptive-statistics">Descriptive Statistics</h2>

<a href="#index">Return to Index</a>
"""
# %% [markdown]
"""
As part of the procedures we check if there are any missing values in our variables.
"""
# %%
def missing_values(data_frame):
    # create a empty dataframe
    missing_values = pd.DataFrame()
    # create a column with missing values
    missing_values['Are missing values?'] = data_frame.isnull().any(axis=0)
    # create a column with number of missing values
    missing_values['Number of missing values'] = data_frame.isnull().sum(axis=0)
# print dataframe
    return missing_values

missing_values(data)

# %%
data.describe().round(2)
# %%
generate_boxplots(data)
# %%
generate_boxplots(data_filtered)
# %%
generate_boxplots(data_yeo_filtered)

# %% [markdown]
"""
The "Check_time" variable stands out from the other variables, with a mean above 109.39 seconds and a median close of 79 seconds. This shows a clear positive skewness in the distribution of this variable. The same occurs with variable depends, imports. This suggests that normalizing the variables is necessary to adjust for differences in magnitudes and expected variance in residuals before setting the final regression. 

After check the boxplots from numerical variable, we can see that the Yeo-Jonshon transformations proposed improve the distributions of almost all variables, mainly our dependent variable, "check_time".
"""
# %% [markdown]
"""
<h2 id="quantile-quantile,-histograms-and-density-plots">Quantile-Quantile, Histograms and Density Plots</h2>

<a href="#index">Return to Index</a>
"""
# %% [markdown]
'''
#### Q-Q plots, Histogram and Density plots of unfiltered and untransformed variables
'''
# %%
fig =  mosaic_qq(
            data=data,
            variables=col_dependent + col_num,
            figsize=(20,15),
            ncols= 4,
            title_fontsize=20
        )
fig[0].suptitle('Quantile-Quantile Plots', fontsize=30, fontweight='bold')
# %%
fig =  mosaic_hist(
            data=data,
            variables=col_dependent + col_num,
            figsize=(20,15),
            ncols= 4,
            title_fontsize=20
        )
fig[0].suptitle('Histograms', fontsize=30, fontweight='bold')
# %% [markdown]
'''
From the Q-Q plots we can see that most variable plotted do not have observations which follow a normal distribution. 
We also notice a number of outliers. From the histogram and density plots we observe that for most variables, the observations are highly concentrated around the mean. The observations also exhibit high levels of positive skewness.
It makes sense to choose only those variables selected using boruta and check the plots of the filtered variables.
'''
# %% [markdown]
'''
#### Q-Q plots, Histogram and Density plots of filtered and untransformed variables
'''
# %%
fig =  mosaic_qq(
            data=data_filtered,
            variables=col_dependent + col_num,
            figsize=(20,15),
            ncols= 4,
            title_fontsize=20
        )
fig[0].suptitle('Quantile-Quantile Plots', fontsize=30, fontweight='bold')
# %%
fig =  mosaic_hist(
            data=data_filtered,
            variables=col_dependent + col_num,
            figsize=(20,15),
            ncols= 4,
            title_fontsize=20
        )
fig[0].suptitle('Histograms', fontsize=30, fontweight='bold')
# %% [markdown]
'''
We plot the Q-Q plots for numeric variables selected using Boruta. 
Here we have fewer influential observations, which could be potential outliers. 
From the histogram and density plot we can see that although the problem of high positive skewness persists, at least now we have the data more or less uniformly distributed.
This is evident form the rugplots.
'''
# %% [markdown]
'''
#### Q-Q plots, Histogram and Density plots of filtered and transformed variables
'''
# %%
fig =  mosaic_qq(
            data=data_yeo_filtered,
            variables=col_dependent + col_num,
            figsize=(20,15),
            ncols= 4,
            title_fontsize=20
        )
fig[0].suptitle('Quantile-Quantile Plots', fontsize=30, fontweight='bold')
# %%
fig =  mosaic_hist(
            data=data_yeo_filtered,
            variables=col_dependent + col_num,
            figsize=(20,15),
            ncols= 4,
            title_fontsize=20
        )
fig[0].suptitle('Histograms', fontsize=30, fontweight='bold')
# %% [markdown]
'''
Now we solve the problem of high positive skewness by using optimal Yeo-Johnson transformation and the parameter (λ) for the variables is discussed in the respective transformation sections.
'''
# %% [markdown]
"""
<h2 id="correlation-and-pairplots">Correlation and Pairplots</h2>


<a href="#index">Return to Index</a>
"""
# %% [markdown]
# Classifying using `Roxygen`
# %%
sns.pairplot(data, hue='Roxygen')
plt.show()
# %% [markdown]
# Classifying using `gh`
# %%
sns.pairplot(data, hue='gh')
plt.show()
# %% [markdown]
"""
Considering both Pairplots with caegorical variables "Roxygen" and "gh". We noticed that Variables doc_size and src_size show a right-skewed distribution. A log transformation may be used to reduce skewness and make the data more normally distributed, which is important for our statistical analysis that assumes normality.
"""
# %%
corr_matrix = data[col_dependent + col_num].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.show()
# %% [markdown]
"""
Based on the Correlation matrix above, we notice that the strongest correlation with the dependent variable "Check_time" is with the independent variable "imports" as well as "r_size".

In addition, we notice that there is no evidence of mulitcollinearity as there are no strong correlations between independent variables. The somewhat suspicious correlation is r_size and imports.
"""
# %%
df = data.copy()
df['log_src_size'] = np.log(df['src_size'] + 1)
df['log_doc_size'] = np.log(df['doc_size'] + 1)
# %%
plt.figure(figsize=(10, 6))
plt.scatter(df['log_src_size'], df['check_time'], alpha=0.7) 
plt.title('Log Transformed src_size vs check_time')
plt.xlabel('Log of src_size')
plt.ylabel('check_time')
plt.grid(True)
plt.show()
# %% [markdown]
"""
There is a dense concentration of data points towards the lower end of the log_src_size axis, indicating that most of the src_size values are small to moderate in size after log transformation.
"""
# %% [markdown]
"""
some potential outliers, particularly in the check_time, where some values are much higher than the rest. These could represent unusually large check times that may need to be investigated further.
"""
# %% [markdown]
"""
There does not appear to be a clear linear relationship between log_src_size and check_time across the entire range of data, suggesting that the size of the source code may not be a direct or sole predictor of check time. 
"""
# %% [markdown]
"""
While there is a slight increase in check_time as log_src_size increases, especially for values of log_src_size less than 1, the correlation does not seem strong, and there's a lot of dispersion in the data points.
"""
# %%
plt.figure(figsize=(10, 6))
plt.scatter(df['log_doc_size'], df['check_time'], alpha=0.7) 
plt.title('Log Transformed doc_size vs check_time')
plt.xlabel('Log of doc_size')
plt.ylabel('check_time')
plt.grid(True)
plt.show()
# %% [markdown]
"""
Similar to the previous plot, there is a significant concentration of data points at the lower end of the log_doc_size axis, suggesting that most documentation sizes are small when transformed logarithmically.
"""
# %% [markdown]
"""
There are potential outliers in check_time, particularly for some values that are extremely high compared to the rest. These outliers could be due to special cases or errors in the data and may warrant further investigation.
"""
# %% [markdown]
"""
There doesn't appear to be a strong correlation between log_doc_size and check_time, as the points do not form a clear trend.
"""
# %%

sns.pairplot(data_yeo_filtered, hue='Roxygen')
plt.show()

# %%
sns.pairplot(data_yeo_filtered, hue='gh')
plt.show()

## %% [markdown]
"""
It can be noted that the Yeo-Johnson transformation on the data without unusual observations presents the strongest linear relationship with the dependent variable. Additionally, the correlations of the transformed independent variables are very weak, which minimizes the risk of multicollinearity.

"""
# %% [markdown]
"""

---

<h1 id="model-selection">Model Selection</h1>
<a href="#index">Return to Index</a>
"""
# %% [markdown]
"""
<h2 id="comparison-between-feasible-models">Comparison Between Feasible Models</h2>


<a href="#index">Return to Index</a>
"""
# %% [markdown]
"""
A comparison is proposed between three different models:
- Model with untransformed variables.
- Model with optimal Yeo-Johnson transformation for each variable.
- Model with selected transformation for each variable.

The models will be estimated using FGLS to control for heteroskedasticity, VIF values will be examined to control for multicollinearity, and the Ramsey RESET test will be performed to control for model misspecification. Subsequently, the addition of quadratic and interaction terms will be evaluated to improve the model specification. Finally, a model will be selected based on its explanatory power and information criteria. The adjusted R-squared will be calculated using bootstrapping in order to generate a confidence interval for it.
"""
# %% [markdown]
"""
<h3 id="model-with-untransformed-variables">Model with Untransformed Variables</h3>


<a href="#index">Return to Index</a>
"""
# %% [markdown]
"""
<h4 id="least-squares-regression">Least Squares Regression</h4>
"""
# %%
# ----- Design matrix and dependent variable -----
Y_1 = data_filtered[col_dependent].copy()
X_1 = data_filtered[col_independent].copy()
X_1 = sm.add_constant(X_1)
# %% ----- OLS regression -----
model_ols_1 = sm.OLS(
    Y_1, X_1
).fit(cov_type='HC1')
model_ols_1.summary()
# %% [markdown]
"""
<h4 id="feasible-generalized-least-squares-regression">Feasible Generalized Least Squares Regression</h4>
"""
# %% ----- FGLS weights -----
weights_1 = FGLS_weights(Y_1, X_1, cov_type='HC1')
# %% ----- FGLS regression -----
model_fgls_1 = sm.WLS(
    Y_1, X_1,
    weights=weights_1
).fit(cov_type='HC1')

model_fgls_1.summary()
# %% VIF
VIF_results_1 = VIF_X(X_1)
# %%
reset_results_1 = ramsey_reset_test(model_fgls_1, order=2)
# %% [markdown]
"""
It is noticeable that there is no substantial difference between the OLS and FGLS estimations. The latter procedure is preferable because it controls the problem of heteroscedasticity. We can observe that this model does not have a significant problem with multicollinearity, as the VIF values are less than 10. However, the model presents specification issues, as the Ramsey test indicates the need to include quadratic terms and interactions.

Next, the inclusion of all possible pairs of quadratic and interaction terms is tested to identify which of these terms are significant. For this purpose, the Ramsey test is used for each of the possible combinations of terms.
"""
# %%
# ----- Iterative test for omitted variables -----
omitted_results_1, X_extended_1 = iterative_reset_omitted(
    Y_1, X_1,
    degree = 2,
    subset_size = 2,
    verbose = True
)
omitted_results_1.loc[omitted_results_1.pvalue_pct > 5]
# %% [markdown]
"""
From the previous analysis, it is clear that the better-specified model is the one that includes the terms `imports^2` and `r_size^2`. Therefore, these terms are included in the model.
"""
# %%
# ----- Adding the variables to the design matrix -----
add_variables_1 = [
    'imports^2',
    'r_size^2'
]
X_with_omitted_1 = X_1.copy()
X_with_omitted_1[add_variables_1] = X_extended_1[add_variables_1]
# %% ----- OLS regression -----
model_ols_ommited_1 = sm.OLS(
    Y_1, X_with_omitted_1
).fit(cov_type='HC1')
# ----- FGLS regression -----
weights_with_omitted_1 = FGLS_weights(Y_1, X_with_omitted_1, cov_type='HC1')
model_fgls_with_omitted_1 = sm.WLS(
    Y_1, X_with_omitted_1,
    weights=weights_with_omitted_1
).fit(cov_type='HC1')
model_fgls_with_omitted_1.summary()
# %%
reset_results_1 = ramsey_reset_test(model_fgls_with_omitted_1, order=2)
# %%
VIF_results_1 = VIF_X(X_with_omitted_1)
# %%
het_results_1 = het_breuschpagan(model_fgls_with_omitted_1)
# %% [markdown]
"""
It can be observed that the inclusion of quadratic terms improves the model's specification. The new estimation with FGLS shows high significance for all parameters, but the quadratic terms have a high VIF value, indicating that the addition of these terms increases the possibility of multicollinearity. Regarding heteroscedasticity, the Breusch-Pagan test indicates that the model has heteroscedasticity issues, but this is corrected with the use of FGLS.
"""
# %%
# ----- Bootstrap Estimation -----
boot_results_1, b_conf_int_1 = bootstrap_estimation(X_with_omitted_1, Y_1, fgls=True, n_boots=1000, cov_type='HC1')
# %%
# ----- Cross Validation -----

scores_1 = cross_validation(X_with_omitted_1, Y_1, nfolds = 5, test_size = 0.3, verbose = True)
# %% [markdown]
"""
To enhance the comparison, the model is evaluated by calculating its respective parameters through bootstrapping, computing the 95% confidence intervals for information criteria and the adjusted R-squared. Additionally, the average RMSE is calculated using cross-validation.
"""

# %% [markdown]
"""
<h4 id="cooks-distance-and-residuals-analysis">Cooks Distance and Residuals Analysis</h4>
<a href="#index">Return to Index</a>
"""
# %%
plot_diagnostic_plots(model_ols_ommited_1)
# %%
influential_plot(model_ols_ommited_1)
# %% [markdown]
"""
From the analysis of the residuals, it can be observed that the relationship is linear. The residuals deviate from normality as a consequence of the increasing variance of the residuals.
"""
# %% [markdown]
"""
<h3 id="model-with-optimal-yeo-johnson-transformation-for-each-variable">Model With Optimal Yeo-Johnson Transformation for each Variable</h3>
<a href="#index">Return to Index</a>
"""
# %% [markdown]
"""
<h4 id="least-squares-regression">Least Squares Regression</h4>
"""
# %%
# ----- Design matrix and dependent variable -----
Y_2 = data_yeo_filtered[col_dependent].copy()
X_2 = data_yeo_filtered[col_independent].copy()
X_2 = sm.add_constant(X_2)
# %% ----- OLS regression -----
model_ols_2 = sm.OLS(
    Y_2, X_2
).fit(cov_type='HC1')
model_ols_2.summary()
# %% [markdown]
"""
<h4 id="feasible-generalized-least-squares-regression">Feasible Generalized Least Squares Regression</h4>
"""
# %% ----- FGLS weights -----
weights_2 = FGLS_weights(Y_2, X_2, cov_type='HC1')
# %% ----- FGLS regression -----
model_fgls_2 = sm.WLS(
    Y_2, X_2,
    weights=weights_2
).fit(cov_type='HC1')

model_fgls_2.summary()
# %% VIF
VIF_results_2 = VIF_X(X_2)
# %%
reset_results_2 = ramsey_reset_test(model_fgls_2, order=2)
# %% [markdown]
"""
It is noticeable that there is no substantial difference between the OLS and FGLS estimations. In this model, the categorical variables do not appear to have explanatory power.
"""
# %%
# ----- Iterative test for omitted variables -----
omitted_results_2, X_extended_2 = iterative_reset_omitted(
    Y_2, X_2,
    degree = 2,
    subset_size = 2,
    verbose = True
)
omitted_results_2.loc[omitted_results_2.pvalue_pct > 5]
# %% [markdown]
"""
From the previous analysis, it is clear that the better-specified model is the one that includes the terms of interaction `depends imports` and `_size src_size`. Therefore, these terms are included in the model. The terms make sense, as the inclusion of more dependencies should increase the number of imported packages. Similarly, the total size of the package increases when more files are included in the `src` directory.
"""
# %%
# ----- Adding the variables to the design matrix -----
add_variables_2 = [
    'depends imports',
    'r_size src_size'
]
X_with_imitted_2 = X_2.copy()
X_with_imitted_2[add_variables_2] = X_extended_2[add_variables_2]
# %% ----- OLS regression -----
model_ols_ommited_2 = sm.OLS(
    Y_2, X_with_imitted_2
).fit(cov_type='HC1')
# ----- FGLS regression -----
weights_with_omitted_2 = FGLS_weights(Y_2, X_with_imitted_2, cov_type='HC1')
models_fgls_with_ommited_2 = sm.WLS(
    Y_2, X_with_imitted_2,
    weights=weights_with_omitted_2
).fit(cov_type='HC1')
models_fgls_with_ommited_2.summary()
# %%
reset_results_2 = ramsey_reset_test(models_fgls_with_ommited_2, order=2)
# %%
VIF_results_2 = VIF_X(X_with_imitted_2)
# %%
het_results_2 = het_breuschpagan(models_fgls_with_ommited_2)
# %% [markdown]
"""
It can be observed that the inclusion of quadratic terms improves the model's specification. The new estimation with FGLS shows high significance for all parameters except for the categorical variables. The interaction terms appear to be significant and do not seem to have a high variance inflation factor. Regarding heteroscedasticity, the Breusch-Pagan test indicates that the model has heteroscedasticity issues, but this is corrected with the use of FGLS.
"""
# %%
# ----- Bootstrap Estimation -----
boot_results_2, b_conf_int_2 = bootstrap_estimation(X_with_imitted_2, Y_2, fgls=True, n_boots=1000, cov_type='HC1')
# %%
# ----- Cross Validation -----

scores_2 = cross_validation(X_with_imitted_2, Y_2, nfolds = 5, test_size = 0.3, verbose = True)
# %% [markdown]
"""
To enhance the comparison, the model is evaluated by calculating its respective parameters through bootstrapping, computing the 95% confidence intervals for information criteria and the adjusted R-squared. Additionally, the average RMSE is calculated using cross-validation.
"""

# %% [markdown]
"""
<h4 id="cooks-distance-and-residuals-analysis">Cooks Distance and Residuals Analysis</h4>
<a href="#index">Return to Index</a>
"""
# %%
plot_diagnostic_plots(model_ols_ommited_2)
# %%
influential_plot(model_ols_ommited_2)
# %% [markdown]
"""
The model using Yeo-Johnson transformations shows an improvement in the model specification, enhancing the properties of the residuals. In this model, there is still the presence of heteroscedasticity, but it is corrected with the use of FGLS. The model shows a better fit than the previous one, as indicated by the value of the adjusted R-squared. The information criteria also indicate that this model is better than the previous one.
"""
# %% [markdown]
r"""
<h3 id="model-with-selected-transformation-for-each-variable">Model With Selected Transformation for each Variable</h3>
<a href="#index">Return to Index</a>

For this model, values of $\lambda$ that fall within the 95% confidence interval for each variable are selected. This is done with the aim of improving the interpretation of the parameters. Below are the selected values of $\lambda$ for each variable.
"""
# %%
lmbd = pd.read_csv(
    os.path.join('data', 'lambda.csv'),
    index_col=0
)
lmbd.T
# %%
# ----- Transforming the variables -----
data_sy = data_filtered.copy()
# %%
for col in col_num + col_dependent:
    data_sy[col] = stats.yeojohnson(data_sy[col], lmbda=lmbd[col][0])
# %% [markdown]
"""
<h4 id="least-squares-regression">Least Squares Regression</h4>
"""
# %%
# ----- Design matrix and dependent variable -----
Y_3 = data_sy[col_dependent].copy()
X_3 = data_sy[col_independent].copy()
X_3 = sm.add_constant(X_3)
# %% ----- OLS regression -----
model_ols_3 = sm.OLS(
    Y_3, X_3
).fit(cov_type='HC1')
model_ols_3.summary()
# %% [markdown]
"""
<h4 id="feasible-generalized-least-squares-regression">Feasible Generalized Least Squares Regression</h4>
"""
# %% ----- FGLS weights -----
weights_3 = FGLS_weights(Y_3, X_3, cov_type='HC1')
# %% ----- FGLS regression -----
model_fgls_3 = sm.WLS(
    Y_3, X_3,
    weights=weights_3
).fit(cov_type='HC1')

model_fgls_3.summary()
# %% VIF
VIF_results_3 = VIF_X(X_3)
# %%
reset_results_3 = ramsey_reset_test(model_fgls_3, order=2)
# %% [markdown]
"""
It is noticeable that there is no substantial difference between the OLS and FGLS estimations. In this model, the categorical variables do not appear to have explanatory power.
"""
# %%
# ----- Iterative test for omitted variables -----
omitted_results_3, X_extended_3 = iterative_reset_omitted(
    Y_3, X_3,
    degree = 2,
    subset_size = 2,
    verbose = True
)
omitted_results_3.loc[omitted_results_3.pvalue_pct > 5]
# %% [markdown]
"""
From the previous analysis, it is clear that the better-specified model is the one that includes the terms of interaction `depends imports` and `_size src_size`. Therefore, these terms are included in the model. The terms make sense, as the inclusion of more dependencies should increase the number of imported packages. Similarly, the total size of the package increases when more files are included in the `src` directory.
"""
# %%
# ----- Adding the variables to the design matrix -----
add_variables_3 = [
    'depends imports',
    'src_size doc_size'
]
X_with_imitted_3 = X_3.copy()
X_with_imitted_3[add_variables_3] = X_extended_3[add_variables_3]
# %% ----- OLS regression -----
model_ols_ommited_3 = sm.OLS(
    Y_3, X_with_imitted_3
).fit(cov_type='HC1')
# ----- FGLS regression -----
weights_with_omitted_3 = FGLS_weights(Y_3, X_with_imitted_3, cov_type='HC1')
models_fgls_with_ommited_3 = sm.WLS(
    Y_3, X_with_imitted_3,
    weights=weights_with_omitted_3
).fit(cov_type='HC1')
models_fgls_with_ommited_3.summary()
# %%
reset_results_3 = ramsey_reset_test(models_fgls_with_ommited_3, order=2)
# %%
VIF_results_3 = VIF_X(X_with_imitted_3)
# %%
het_results_3 = het_breuschpagan(models_fgls_with_ommited_3)
# %% [markdown]
"""
It can be observed that the inclusion of interaction terms improves the model's specification. The new estimation with FGLS shows high significance for all parameters except for the categorical variables. The interaction terms appear to be significant and do not seem to have a high variance inflation factor. Regarding heteroscedasticity, the Breusch-Pagan test indicates that the model has heteroscedasticity issues, but this is corrected with the use of FGLS.
"""
# %%
# ----- Bootstrap Estimation -----
boot_results_3, b_conf_int_3 = bootstrap_estimation(X_with_imitted_3, Y_3, fgls=True, n_boots=1000, cov_type='HC1')
# %%
# ----- Cross Validation -----

scores_3 = cross_validation(X_with_imitted_3, Y_3, nfolds = 5, test_size = 0.3, verbose = True)
# %% [markdown]
"""
To enhance the comparison, the model is evaluated by calculating its respective parameters through bootstrapping, computing the 95% confidence intervals for information criteria and the adjusted R-squared. Additionally, the average RMSE is calculated using cross-validation.
"""

# %% [markdown]
"""
<h4 id="cooks-distance-and-residuals-analysis">Cooks Distance and Residuals Analysis</h4>
<a href="#index">Return to Index</a>
"""
# %%
plot_diagnostic_plots(model_ols_ommited_3)
# %%
influential_plot(model_ols_ommited_3)
# %% [markdown]
"""
The model using selected Yeo-Johnson transformations shows an improvement in the model specification, enhancing the properties of the residuals. In this model, there is still the presence of heteroscedasticity, but it is corrected with the use of FGLS. Regarding the previous model, by analyzing the confidence interval of the adjusted R-squared through bootstrapping, it can be observed that the model with selected transformations has a statistically similar fit to the previous model. This model is preferable because the dependent variable can be interpreted as a semi-elasticity.
"""
# %% [markdown]
"""
<h3 id="comparison-summary">Comparison Summary</h3>
<a href="#index">Return to Index</a>

Below are the different comparison statistics between the three proposed models, calculated through bootstrapping and showing the 95
"""
# %% [markdown]
"""
| Metric | Model 1 | Model 2 | Model 3 |
|---|---|---|---|
| R Squared | 0.425 | 0.59 | 0.592 |
| R Squared (Lower Bound) | 0.409 | 0.584 | 0.576 |
| R Squared (Upper Bound) | 0.441 | 0.614 | 0.607 |
| Adjusted R Squared | 0.425 | 0.599 | 0.592 |
| Adjusted R Squared (Lower Bound) | 0.409 | 0.583 | 0.576 |
| Adjusted R Squared (Upper Bound) | 0.44 | 0.614 | 0.607 |
| AIC | 123153.28 | -8616.17 | 10921.27 |
| AIC (Lower Bound) | 122606.58 | -8974.27 | 10565.77 |
| AIC (Upper Bound) | 123698.21 | -8244.03 | 11295.61 |
| BIC | 123234.29 | -8534.59 | 11002.85 |
| BIC (Lower Bound) | 122688.17 | -8892.69 | 10647.35 |
| BIC (Upper Bound) | 123779.8 | -8162.45 | 11377.19 |
| Standard Deviation | 4.04 | 3.39 | 3.45 |
| Standard Deviation (Lower Bound) | 3.99 | 3.28 | 3.34 |
| Standard Deviation (Upper Bound) | 4.21 | 3.51 | 3.57 |
| Mean RMSE CV | 40.3 | 0.17 | 0.38 |
| Mean RMSE CV over Standard Deviation | 9.98 | 0.05 | 0.11 |
"""
# %% [markdown]
"""
From the table, it can be seen that model 1 has the lowest explanatory power and also presents the highest values of AIC and BIC. On the other hand, models 2 and 3 have similar explanatory power, but model 2 has lower AIC and BIC. The metrics of average RMSE from cross-validation over the standard deviation allow us to compare between models, showing that model 2 performs better.
"""
# %% [markdown]
"""
Given that the loss of explanatory power between models 2 and 3 is minimal, model 3 is chosen as the final model because the marginal effects could be interpreted as semi-elasticities.
"""
# %% [markdown]
"""
<h2 id="analysis-of-the-selected-model">Analysis of the Selected Model</h2>
<a href="#index">Return to Index</a>

As explained in the previous section, the selected model includes the Yeo-Johnson transformations selected for each variable. From this model, the estimation using FGLS is as follows:
"""
# %%
models_fgls_with_ommited_3.summary()
# %% [markdown]
"""
<h3 id="robustness-analysis">Robustness Analysis</h3>
<a href="#index">Return to Index</a>

Below are the estimators with a 95% bootstrap confidence interval for the selected model. To improve the robustness of the estimates, FGLS estimators and 5000 repetitions are used.
"""
# %%
# ----- Bootstrap Estimation -----
boot_results_3, b_conf_int_3 = bootstrap_estimation(X_with_imitted_3, Y_3, fgls=True, n_boots=5000, cov_type='HC1')

# %%
_ = mosaic_boot(boot_results_3, boot_results_3.columns, figsize=(20, 25), ncols=3, title_fontsize=12)

# %% [markdown]
"""
From the robustness analysis, it can be observed that the explanatory power of the model fluctuates between 0.58 and 0.61. Additionally, it is observed that the categorical variables `Roxygen` and `gh` have a confidence interval that crosses zero, indicating that they are not statistically significant. Their presence in the model is justified by the need to control for different documentation frameworks and the need to control for the use of GitHub as a code repository, which are often used in development frameworks that require more validation time.
"""
# %% [markdown]
"""
<h3 id="cross-validation">Cross-Validation</h3>
<a href="#index">Return to Index</a>
"""
# %%
# ----- Cross Validation -----

scores_3 = cross_validation(X_with_imitted_3, Y_3, nfolds = 5, test_size = 0.3, verbose = True)

# %% [markdown]
"""
From the cross-validation procedure and estimation via bootstrap, it can be estimated that the average RMSE is 0.38. Given that the mean of the dependent variable is 4.35, it can be said that our model has an average error of 8.7%, which is a low value. Therefore, it can be concluded that the selected model is robust.
"""
# %% [markdown]
"""
<h3 id="marginal-effects-analysis-and-parameter-interpretation">Marginal Effects Analysis and Parameter Interpretation</h3>
<a href="#index">Return to Index</a>

Below are once again the results of the estimation of the selected model.
"""
# %%
models_fgls_with_ommited_3.summary()
# %% [markdown]
"""
Because our coefficients correspond to transformed variables, the marginal effects do not have a direct interpretation, except for those derived from the categorical variables.
"""
# %% [markdown]
"""
From the categorical variables, it can be observed that both `Roxygen` and `gh` have a positive effect on the dependent variable, indicating that including GitHub links and adding Roxygen documentation increases the validation time. The addition of Roxygen increases the validation time by 0.3%, while the addition of GitHub links increases the validation time by 1.2%. It can be noted that these increases in time are low and not significant.
"""
# %% [markdown]
r"""
To calculate the marginal effect of a transformed variable that does not have interaction effects, the following formula is used:
$$
\frac{1}{y}\frac{\partial y}{\partial x_i} = \beta_i \left(x_i + 1\right)^{\lambda_i - 1}
$$

This means that the percentage change in the dependent variable $y$ with a change in the independent variable $x_i$ depends on the level of the same. Because of this, the coefficient $\beta_i$ does not have a direct interpretation, but it is possible to analyze the percentage response of the dependent variable given the observed values of $x_i$.
"""
# %%
def margin_no_interactions(data, variable, estimation, lambdas):
    
    # ----- DataFrame with coefficients -----
    temp_params = pd.DataFrame(estimation.params).T
    # Variable coefficient estimation
    temp_beta = temp_params[variable][0] 
    # Transformation lambda yeo-johnson
    temp_lmbd = lambdas[variable][0]
    # ----- Range of the variables -----
    rng_1 = np.linspace(data[variable].min(), data[variable].max(), 1000)
    # ----- Marginal effects -----
    margin = temp_beta * (rng_1 + 1)**(temp_lmbd -1 )
    
    return rng_1, margin
# %%
rng_1, margin = margin_no_interactions(data_filtered, 'data_size', models_fgls_with_ommited_3, lmbd)
fig, ax = plt.subplots()
ax.plot(rng_1, margin)
ax.set_title('data_size', fontsize=12, fontweight='bold')
ax.set_xlabel('data_size Observed Range', fontsize=12)
ax.set_ylabel('Marginal Effect', fontsize=12)
fig.suptitle('Marginal Effect on check_time (as a Semielasticity)', fontsize=12, fontweight='bold')
# %% [markdown]
"""
It can be observed that as the size of the data loaded into the package increases, the validation time also increases. However, each additional unit of data size has a diminishing marginal effect on the validation time. This can be seen in the graph, where the slope of the curve is decreasing.
"""
# %% [markdown]
r"""
To calculate the marginal effect of a transformed variable with interaction effects, the following formula is used:
$$
\frac{1}{y}\frac{\partial y}{\partial x_i} = \beta_i \left(x_i + 1\right)^{\lambda_i - 1} + \beta_{i,j}  \frac{(x_j + 1)^{\lambda_j} -1}{\lambda_j} (x_i + 1)^{\lambda_i - 1}
$$

This implies that percentage change in the dependent variable $y$ with a change in the independent variable $x_i$ depends on the level of the same and the level of $x_j$. Because of this, the coefficient $\beta_i$ or $\beta_{i,j}$ don't have a direct interpretation, but it is possible to analyze the percentage response of the dependent variable given the observed values of $x_i$ and $x_j$.
"""
# %%
def margin_interactions(data, variable, interaction, estimation, lambdas):
    
    # ----- DataFrame with coefficients -----
    temp_params = pd.DataFrame(estimation.params).T
    # Variable coefficient estimation
    temp_beta = temp_params[variable][0]
    try:
        temp_gamma = temp_params[variable + ' ' + interaction][0]  
    except:
        temp_gamma = temp_params[interaction + ' ' + variable][0]  
    # Transformation lambda yeo-johnson
    temp_lmbd_1 = lambdas[variable][0]
    temp_lmbd_2 = lambdas[variable][0]
    # ----- Range of the variables -----
    rng_1 = np.linspace(data[variable].min(), data[variable].max(), 1000)
    rng_2 = np.linspace(data[interaction].min(), data[interaction].max(), 5)
    # ----- Marginal effects -----
    margin = [] 
    for i in range(len(rng_2)):
        margin_t0 = (rng_1 + 1)**(temp_lmbd_1 - 1)
        margin_t1 = temp_beta * margin_t0
        margin_t2 = temp_gamma * stats.yeojohnson(rng_2[i], lmbda=temp_lmbd_2) * margin_t0
        margin.append(margin_t1 + margin_t2)
    return rng_1, rng_2, margin
# %%
rng, inter, margin = margin_interactions(data_filtered, 'depends', 'imports', models_fgls_with_ommited_3, lmbd)
# %%
fig, ax = plt.subplots()
for i in range(len(margin)):
    ax.plot(rng, margin[i])
ax.legend(inter, title = 'imports')
ax.set_title('depends', fontsize=12, fontweight='bold')
ax.set_xlabel('depends Observed Range', fontsize=12)
ax.set_ylabel('Marginal Effect', fontsize=12)
fig.suptitle('Marginal Effect on check_time (as a Semielasticity)', fontsize=12, fontweight='bold')

# %%
rng, inter, margin = margin_interactions(data_filtered, 'imports', 'depends', models_fgls_with_ommited_3, lmbd)
# %%
fig, ax = plt.subplots()
for i in range(len(margin)):
    ax.plot(rng, margin[i])
ax.legend(inter, title = 'depends')
ax.set_title('imports', fontsize=12, fontweight='bold')
ax.set_xlabel('imports Observed Range', fontsize=12)
ax.set_ylabel('Marginal Effect', fontsize=12)
fig.suptitle('Marginal Effect on check_time (as a Semielasticity)', fontsize=12, fontweight='bold')
# %% [markdown]
"""
It can be observed that the semi-elasticity of the validation time with respect to the variable `depends` is positive but decreasing. This indicates that as the number of dependencies increases, the validation time increases, but each additional unit of dependencies has a diminishing marginal effect on the validation time. Additionally, it can be observed that as `imports` increases, the marginal effect of `depends` on the elasticity of the validation time reduces. This means that as the number of imported packages increases, adding extra dependencies has a smaller marginal effect on the validation time.
"""
# %% [markdown]
"""
This effect can also be observed when analyzing the marginal effects of `imports`. However, its marginal effects are not drastically altered when `depends` increases.
"""
# %%
rng, inter, margin = margin_interactions(data_filtered, 'src_size', 'doc_size', models_fgls_with_ommited_3, lmbd)
# %%
fig, ax = plt.subplots()
for i in range(len(margin)):
    ax.plot(rng, margin[i])
ax.legend(inter, title = 'doc_size')
ax.set_title('src_size', fontsize=12, fontweight='bold')
ax.set_xlabel('src_size Observed Range', fontsize=12)
ax.set_ylabel('Marginal Effect', fontsize=12)
fig.suptitle('Marginal Effect on check_time (as a Semielasticity)', fontsize=12, fontweight='bold')

# %%
rng, inter, margin = margin_interactions(data_filtered, 'doc_size', 'src_size', models_fgls_with_ommited_3, lmbd)
# %%
fig, ax = plt.subplots()
for i in range(len(margin)):
    ax.plot(rng, margin[i])
ax.legend(inter, title = 'src_size')
ax.set_title('doc_size', fontsize=12, fontweight='bold')
ax.set_xlabel('doc_size Observed Range', fontsize=12)
ax.set_ylabel('Marginal Effect', fontsize=12)
fig.suptitle('Marginal Effect on check_time (as a Semielasticity)', fontsize=12, fontweight='bold')
# %% [markdown]
"""
It can be observed that both `src_size` and `doc_size` have positive marginal effects, which means that as they increase, the percentage change of `check_times` is positive. This increase appears to decline as they increase, and additionally, as they increase, they also reduce the marginal effect of the variable with which they interact.
"""
# %% [markdown]
"""
It can also be observed that the marginal effect of `src_size` decreases rapidly, indicating that an increase in the size of the `src` folder does not significantly increase the validation time of the package.
"""
# %% [markdown]
"""
From the previous analysis, it can be noted that the highest semi-elasticity is found in the marginal effect of `doc_size`, indicating that when constructing a package, careful consideration should be given to the amount of documentation added, as it tends to increase the validation time more than the other variables.
"""

# %%
