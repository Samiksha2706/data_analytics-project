import pandas as pd
from scipy.stats.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sb
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import Normalizer
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

from sklearn.datasets import load_digits
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split



df = pd.read_excel(r'C:\it\task1.xlsx')

# converting into csv file
df.to_csv("task1.csv", index = None, header=True)

# rename the column
df.rename(columns = {'Distance from nearest Metro station (km)':'dis_from_metro'}, inplace = True)
df.rename(columns = {'Number of convenience stores':'No_con'}, inplace = True)
df.rename(columns = {'House price of unit area':'price'}, inplace = True)
df.rename(columns = {'Number of bedrooms':'br'}, inplace = True)
df.rename(columns = {'House size (sqft)':'size'}, inplace = True)

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
#means if more hosing age leading low prices and if the distance from the metro will more more, the price
# will drop
# while other variables shows positive correlation

# From the correlation value we can deduce that, distance from the metro,Number of convenience stores,
#longitude ,latitude are relatively high factor for determining price of the house.

# let's understand correlation of variable in a better way by scatter plot graph
#plt.scatter(list1,list8)
#plt.scatter(list1,list3)
#plt.scatter(list1,list4)
#plt.scatter(list1,list5)

#let's fit the regression line to have better idea
sb.lmplot(x = "price",
            y = "House Age", 
            ci = None,
            data = df)

sb.lmplot(x = "price",
            y = "dis_from_metro", 
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

# to find null vallues
print(df.isna().sum())
# there are no null values in a dataset

# to know the datatypes
print(df.dtypes)

#changing int to float
df['No_con'] = df['No_con'].astype(float)
df['br'] = df['br'].astype(float)
df['size'] = df['size'].astype(float)
print(df.dtypes)

y = df.price.values
X = df.drop(['price'],axis=1)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)
numeric=['Transaction date','House Age', 'dis_from_metro','No_con','latitude','longitude','br','size','price']

lr= LinearRegression()
lr.fit(X,y)
lr= LinearRegression()
ypred = lr.predict(X)
r2_score(y_test, ypred), mean_absolute_error(y_test, ypred), np.sqrt(mean_squared_error(y_test, ypred))

