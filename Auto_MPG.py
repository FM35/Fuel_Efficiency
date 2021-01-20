import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

Auto_MPG = pd.read_csv('Auto_MPG\Auto_MPG.csv')

Auto_MPG = Auto_MPG.drop(Auto_MPG[Auto_MPG.horsepower == '?'].index)
Auto_MPG['horsepower'] = Auto_MPG['horsepower'].astype(np.int64)
Auto_MPG['cylinders'] = Auto_MPG['cylinders'].astype(object)
Auto_MPG['origin'] = Auto_MPG['origin'].astype(object)

#print(Auto_MPG['horsepower'].dtypes)

#print(Auto_MPG.describe(include = 'all'))

Auto_MPG = Auto_MPG.drop(['car name'], axis = 1)

f, axes = plt.subplots(4, 2, figsize=(10, 10), sharex= False)

sns.distplot(Auto_MPG['mpg'] , ax = axes[0,0])
sns.distplot(Auto_MPG['cylinders'], ax = axes[0,1])
sns.distplot(Auto_MPG['displacement'], ax = axes[1,0])
sns.distplot(Auto_MPG['horsepower'], ax = axes[1,1])
sns.distplot(Auto_MPG['weight'] , ax = axes[2,0])
sns.distplot(Auto_MPG['acceleration'], ax = axes[2,1])
sns.distplot(Auto_MPG['model year'], ax = axes[3,0])
sns.distplot(Auto_MPG['origin'], ax = axes[3,1])

#plt.show()

log_mpg = np.log(Auto_MPG['mpg'])
Auto_MPG['log_mpg'] = log_mpg

Auto_MPG = Auto_MPG.drop(['mpg'], axis = 1)

f, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8)) = plt.subplots(2, 4, sharey = True, figsize =(15,10))

ax1.scatter(Auto_MPG['displacement'], Auto_MPG['log_mpg'])
ax1.set_title('mpg and displacement')

ax2.scatter(Auto_MPG['horsepower'], Auto_MPG['log_mpg'])
ax2.set_title('mpg and horsepower')

ax3.scatter(Auto_MPG['weight'], Auto_MPG['log_mpg'])
ax3.set_title('mpg and weight')

ax4.scatter(Auto_MPG['acceleration'], Auto_MPG['log_mpg'])
ax4.set_title('mpg and acceleration')

ax5.scatter(Auto_MPG['model year'], Auto_MPG['log_mpg'])
ax5.set_title('mpg and model year')

#plt.show()

from statsmodels.stats.outliers_influence import variance_inflation_factor

variables = Auto_MPG[['horsepower','acceleration']]
vif = pd.DataFrame()
vif["VIF"] = [variance_inflation_factor(variables.values, i) for i in range(variables.shape[1])]
vif['features'] = variables.columns
#print(vif)

Auto_MPG_with_dummies = pd.get_dummies(Auto_MPG, drop_first = True)
Auto_MPG_with_dummies = Auto_MPG_with_dummies.drop(['weight', 'model year', 'displacement'], axis = 1)

cols = ['log_mpg', 'horsepower', 'acceleration', 'cylinders_4', 'cylinders_5', 'cylinders_6', 'cylinders_8',
       'origin_2', 'origin_3']

Auto_MPG_preprocessed = Auto_MPG_with_dummies[cols]

print(Auto_MPG_preprocessed)