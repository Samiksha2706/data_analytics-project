import pandas as pd
from scipy.stats.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import QuantileTransformer
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
import seaborn as sns
# for GUI
import PySimpleGUI as sg
import os.path

#study the dataset and Predict the price of a house 
df = pd.read_excel(r'C:\it\task1.xlsx')

# converting into csv file
df.to_csv("task1.csv", index = None, header=True)

# rename the column
df.rename(columns = {'Distance from nearest Metro station (km)':'dis_from_metro'}, inplace = True)
df.rename(columns = {'Number of convenience stores':'No_con'}, inplace = True)
df.rename(columns = {'House price of unit area':'price'}, inplace = True)
df.rename(columns = {'Number of bedrooms':'br'}, inplace = True)
df.rename(columns = {'House size (sqft)':'size'}, inplace = True)
df.rename(columns = {'Transaction date' :'Transaction_date' },inplace=True)

# to find the dependency among variables
list1 = df['price']
list2 = df['House Age']
list3=df['dis_from_metro']
list4=df['No_con']
list5=df['longitude']
list6=df['br']
list7=df['size']
list8=df['latitude']


# Apply the pearson correlation 
corr1, _ = pearsonr(list2,list1)
corr2, _ = pearsonr(list3,list1)
corr3, _ = pearsonr(list4,list1)
corr4, _ = pearsonr(list5,list1)
corr5, _ = pearsonr(list6,list1)
corr6, _ = pearsonr(list7,list1)
corr7, _ = pearsonr(list8,list1)

print('Pearsons correlation between price and housing age is: %.3f' % corr1)
print('Pearsons correlation between price and distance from the metro is: %.3f' % corr2)
print('Pearsons correlation between price and Number of convenience stores is: %.3f' % corr3)
print('Pearsons correlation between price and longitude is: %.3f' % corr4)
print('Pearsons correlation between price and Number of bedrooms is: %.3f' % corr5)
print('Pearsons correlation between price and House size (sqft) is: %.3f' % corr6)
print('Pearsons correlation between price and latitude is: %.3f' % corr7)

# In the output we can see, 
# There is negetive correlation between price and housing age and price and distance from the metro,which
#means if more hosing age leading low prices and if the distance from the metro will more more, the pricewill drop
# while other variables shows positive correlation

# From the correlation value we can deduce that, distance from the metro,Number of convenience stores,
#longitude ,latitude are relatively high factor for determining price of the house.

# let's understand correlation of variable in a better way by scatter plot graph and
#let's fit the regression line on scatter plot to have better idea

sb.lmplot(x = "House Age",
            y = "price", 
            ci = None,
            data = df)

sb.lmplot(x = "dis_from_metro",
            y = "price", 
            ci = None,
            data = df)

sb.lmplot(x = "price",
            y = "No_con", 
            ci = None,
            data = df)
sb.lmplot(x = "price",
            y = "longitude", 
            ci = None,
            data = df)

sb.lmplot(x = "price",
            y = "br", 
            ci = None,
            data = df)

sb.lmplot(x = "price",
            y = "size", 
            ci = None,
            data = df)

sb.lmplot(x = "price",
            y = "latitude", 
            ci = None,
            data = df)


# After plotting regression line in scatter plot graph, we can say that data points placed linearly and
# therefore we can use linear regression
# but before that let us check the distribution of continous variables
#Continous variables are - House age , distance from metro,Number of covinience stores,

df.hist(column='No_con')
df.hist(column='longitude')
df.hist(column='latitude')
df.hist(column='dis_from_metro')
df.hist(column='size')
df.hist(column='House Age')
df.hist(column='price') 
# number of bedrooms(br) variable has categorical values
numerical_features = [feature for feature in df.columns if df[feature].dtypes != 'O']
continuous_feature=[feature for feature in numerical_features]

# logarithmic distribution of every feature
for feature in continuous_feature:
    data = df.copy()
    if 0 in data[feature].unique():
        pass    
    else:
        data[feature]=np.log(data[feature])
        data['price']=np.log(data['price'])
        plt.scatter(data[feature],data['price'])
        plt.xlabel(feature)
        plt.ylabel('price')
        plt.title(feature)
        plt.show()

# finding outliers in every feature
for feature in numerical_features:
    data=df.copy()
    if 0 in data[feature].unique():
        pass
    else:
        data[feature]=np.log(data[feature])
        data.boxplot(column=feature)
        plt.ylabel(feature)
        plt.title(feature)
        plt.show()
# we can see that longitude , latitude and price variables has outliers        

#  for categorical variable
df.dtypes




# chaging categorical value into object type
df['br'] = df.br.astype(object)
df['Transaction_date']=df.Transaction_date.astype(object)

categorical_features=[feature for feature in df.columns if df[feature].dtypes=='object']
print(categorical_features) 
# as expected , we only have br(number of bedrooms as a categorical feature)

# to know relation between relation between categorical value and price
for feature in categorical_features:
    data=df.copy()
    data.groupby(feature)['price'].median().plot.bar()
    plt.xlabel(feature)
    plt.ylabel('price')
    plt.title(feature)
    plt.show()  


# Preprocessing
# to remove outli'ers
cols = ['longitude', 'latitude']
Q1 = df[cols].quantile(0.25)
Q3 = df[cols].quantile(0.75)
IQR = Q3 - Q1

df = df[~((df[cols] < (Q1 - 1.5 * IQR)) |(df[cols] > (Q3 + 1.5 * IQR))).any(axis=1)]
     

# to find null vallues
print(df.isna().sum())
# there are no null values in a dataset

# to know the datatypes
print(df.dtypes)

#changing int to float
df['No_con'] = df['No_con'].astype(float)
df['size'] = df['size'].astype(float)
print(df.dtypes) 

# displaying year 

# MACHINE LEARNING

# Linear Regression
mdl = LinearRegression()
X = data.drop(['price'],axis=1) 
Y = data['price']

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)

mdl = LinearRegression()
mdl.fit( X, Y )
y_prediction =  mdl.predict(x_test)
score=r2_score(y_test,y_prediction)

print(score)
print( mean_squared_error(y_test,y_prediction))
print(np.sqrt(mean_squared_error(y_test,y_prediction)))


# prediction,it will return the predicted price
print("predicted price based on the parameters")

# passing the values
print(mdl.predict([[2013.5 , 13.3 , 561.9845 , 5.0 , 24.9874 ,  121.54 , 2 ,875.0 ]]))

# random forest regression

# after execution of this model , we cam see that it is more accurate then linear regression     
model = RandomForestRegressor()

#transforming target variable through quantile transformer
ttr = TransformedTargetRegressor(regressor=model, transformer=QuantileTransformer(output_distribution='normal'))
ttr.fit(X,Y)
yhat = ttr.predict(x_test)
r2_score(y_test, yhat), mean_absolute_error(y_test, yhat), np.sqrt(mean_squared_error(y_test, yhat))



# asking user for the input value for prediction
print("price can be approx")
num1 = input("Enter first number : ")
num2 = float(input("Enter first number : "))
num3 = float(input("Enter first number : "))
num4 = float(input("Enter first number : "))
num5 = float(input("Enter first number : "))
num6 = float(input("Enter first number : "))
num7 = input("Enter first number : ")
num8 = float(input("Enter first number : "))

res= ttr.predict([[num1,num2,num3,num4,num5,num6,num7,num8]])
print(res)

# more insights from the data
# Visualization

# 1) fluctuation in price in 2013 and 2014
plt.figure(figsize=(25, 10))
nyc_chart = sns.lineplot(
    x="Transaction date",
    y='price',
    
    data=df
).set_title('Level 2 on different parts of India on different months')
plt.show()

plt.figure(figsize=(25, 10))
nyc_chart = sns.lineplot(
    x="Transaction date",
    y='price',
    hue='br',
    data=df
).set_title('Level 2 on different parts of India on different months')
plt.show()

