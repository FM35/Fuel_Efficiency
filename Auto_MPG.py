import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.linear_model import LinearRegression

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

log_mpg = np.log(Auto_MPG['mpg'])
Auto_MPG['log_mpg'] = log_mpg

Auto_MPG = Auto_MPG.drop(['mpg'], axis = 1)

f, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8)) = plt.subplots(2, 4, sharey = False, figsize =(15,10))

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

#targets and inputs
targets = Auto_MPG_preprocessed['log_mpg']
inputs = Auto_MPG_preprocessed.drop(['log_mpg'], axis = 1)

#Feature Scaling the inputs
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(inputs)
inputs_scaled = scaler.transform(inputs)

#Splitting data into training and testing
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(inputs_scaled,targets,test_size = 0.2, random_state = 365)

#making the regression model
reg = LinearRegression()
reg.fit(x_train, y_train)

#You can check accuracy of the models by either plotting predicted values against the training ones or plotting the residuals which is the difference of between 
#the actual ones - the predicted ones
y_hat = reg.predict(x_train)
#plt.scatter(y_train, y_hat)
sns.distplot(y_train - y_hat)
plt.title('Residuals PDF', size = 18)

print(reg.score(x_train, y_train))
print(reg.intercept_)

#Putting the weights and Feature names into a dataframe
reg_summary = pd.DataFrame(inputs.columns.values, columns = ['Features'])
reg_summary['Weights'] = reg.coef_

print(reg_summary)
plt.show()