#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
#%%
df=pd.read_csv(r"C:\Users\eglha\Downloads\Understanding Uncertainty\data\nhanes_data_17_18.csv")
df.head()


# %%
print(df.isnull().sum())
# %%
print(df.count())
# %%
df.head(100)
# %%
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)

summary=pd.DataFrame({"dtype":df.dtypes,
                      "n_unique":df.nunique(),
                      "n_missing":df.isnull().sum(), "total_rows":len(df)})
summary
# %%
gen_health_order=["Excellent", "Very good", "Good", "Fair or", "Poor?"]
df["GeneralHealthCondition"]=pd.Categorical(df["GeneralHealthCondition"], categories=gen_health_order, ordered=True)
contingency_table=pd.crosstab(df["GeneralHealthCondition"], df["MaritalStatus"], dropna=False)
contingency_table


# %%
uniq_conditions=df["GeneralHealthCondition"].unique()
# %%
ordered_conditions=[uniq_conditions[5], uniq_conditions[3], uniq_conditions[1], uniq_conditions[2], uniq_conditions[4]]
ordered_conditions
# %%
df["GeneralHealthCondition"]=pd.Categorical(df["GeneralHealthCondition"], categories=ordered_conditions, ordered=True)
contingency_table=pd.crosstab(df["GeneralHealthCondition"], df["MaritalStatus"], dropna=False)
contingency_table

# %%
col_percent=contingency_table.div(contingency_table.sum(axis=0), axis=1)*100
col_percent=col_percent.round(2)
col_percent
# %%
row_percent=contingency_table.div(contingency_table.sum(axis=1), axis=0)*100
row_percent=row_percent.round(2)
row_percent
# %%
"""
It appears that the respondents who are separated 
skewed lower on the general health condition rankings, 
especially when compared with married respondents.
"""

# %%
# QUESTION 2

# %%
# Categorical Variable = Gender
# Numeric Variable = Protein

df["Gender"].value_counts()
# %%
protein_descr=df["ProteinGm_DR1TOT"].describe()
# %%
df["log_protein"]=np.log(df["ProteinGm_DR1TOT"])
df["log_protein"].describe()
# %%
conditioner= "Gender"
sns.kdeplot(data=df, x="log_protein", hue=conditioner, common_norm=False).set(title="KDE")
plt.xlim(-2,8)
plt.show()
sns.ecdfplot(data=df, x="log_protein", hue=conditioner).set(title="ECDF")
plt.xlim(-2,8)
plt.show()
df.loc[:,["ProteinGm_DR1TOT", conditioner]].groupby(conditioner).describe()
# %%
"""
It appears that as I hypothesized, male respondents 
have a slightly higher protein measurement than female respondents. 
"""

# %%
""" 
$$MSE(\\hat{y}(z)) = \\dfrac{1}{N} \\sum_{i=1}^N \\left\\lbrace y_i - \\hat{y}(z) \\right\\rbrace^2 \\frac{1}{h}k\\left(\\frac{z-x_i}{h}\\right)$$
"""

# %% 
"""
LCLS estimator formula:

MSE= (1/N) * Σ [ (y_i - ŷ(z))^2 * (1/h) * k((z - x_i)/h) ]

The MSE calculates the difference between the actual values 
(y_i) and the estimated values (y_hat(z)) at a specific point z. 
This difference is then divided by the bandwidth (h).

It is then weighted with the kernel function (k), which is also then 
divided by the bandwidth. This "smooths" the estimates, with 
points closer to z having a higher weight.



LCLS Formula: constant a= Σ [ K subscript h (Z subscript i - z)(Y subscript i - a)^2 ]

"""

# %%



# %%
# QUESTION 4

y =df["TotalSugarsGm_DR2TOT"]
x= df["AnnualHouseholdIncome"]

def lcls(x,y, plot=True):

    n = len(x) 
    grid = np.sort(x.unique()) 


    iqr = np.quantile(x,.75) - np.quantile(x,.25)
    h = 0.9 * min(np.std(x), iqr/1.34) * n **(-0.2)
    # h represents a normal bandwith here

    # (k) kernel:
    I = -(x.to_numpy().reshape(-1,1)-grid.reshape(1,-1) )**2
    K = np.exp(I/(2*h**2) )/np.sqrt(2*np.pi*h**2 )

    # Compute LCLS estimator:
    numerator = y@K 
    denominator = np.sum(K,axis=0) 
    y_hat = numerator/denominator 

    # Plot results:
    if plot:
        sns.scatterplot(data=df, y="TotalSugarsGm_DR2TOT", x="AnnualHouseholdIncome", alpha=.05)
        sns.lineplot(x=grid,y=y_hat, color='blue')

    return y_hat, grid

y_hat, grid = lcls(x,y)

"""
Contrary to my hypothesis, the plot for grams of sugar and 
annual household income shows minimal correlation.
"""
# %%

# QUESTION 5

plt.scatter(df["AnnualHouseholdIncome"], df["TotalSugarsGm_DR2TOT"], alpha=0.05)
means = df.loc[:,['AnnualHouseholdIncome','TotalSugarsGm_DR2TOT']].groupby('TotalSugarsGm_DR2TOT').mean()
sns.scatterplot(data=df, y='AnnualHouseholdIncome', x='TotalSugarsGm_DR2TOT',alpha=.05, label='data')
sns.lineplot(data=means, x='TotalSugarsGm_DR2TOT',y='AnnualHouseholdIncome',color='orange')

"""
Again, we see little to no correlation.
"""
# %%
