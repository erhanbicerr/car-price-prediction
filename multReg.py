import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
scale = StandardScaler()

df = pd.read_excel('http://cdn.sundog-soft.com/Udemy/DataScience/cars.xls')

'''
df1 = df[['Mileage','Price']]
bins = np.arange(0, 50000, 10000)
groups = df1.groupby(pd.cut(df1['Mileage'],bins)).mean()
print(groups.head())
groups['Price'].plot.line()
plt.show()
'''

X = df[['Mileage', 'Cylinder', 'Doors']]
y = df['Price']

X[['Mileage', 'Cylinder', 'Doors']] = scale.fit_transform(X[['Mileage', 'Cylinder', 'Doors']].as_matrix())

print(X)
mileVar, cylVar, doorVar = input("Mileage, cylinder, doors ? ").split()

est = sm.OLS(y, X).fit()
print(est.summary())
scaled = scale.transform([[mileVar, cylVar, doorVar]]) #infos of car that we want to predict its price
print(scaled)
predicted = est.predict(scaled) #prediction of price
print("Your car is about $" + str(predicted) + ".")
#Price = B0 + B1 * Mileage + B2 * Cylinder + B3 * Doors 
