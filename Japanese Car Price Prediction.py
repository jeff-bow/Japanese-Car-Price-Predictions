#!/usr/bin/env python
# coding: utf-8

# In[2]:


#Run this cell to automatically download the dataset to the same folder as this Jupyter Notebook file
from urllib.request import urlretrieve
#If this does not work simply open the below link in your browser to
#view the dataset

url = "https://ac-101708228-virtuoso-prod.s3.amazonaws.com/uploads/download/120/us_car_prices.csv"


# In[1]:


# importing libraries
import pandas as pd
#importing additional libraries
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn.linear_model as skl
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Handling the dataset
inpPath = r'/Users/jeffbowers/Documents/Jupyter Notebook Work/'
car_prices_us = pd.read_csv(inpPath + 'us_car_prices.csv', 
                    index_col = 0)
car_prices_us


# In[2]:


car_prices_us.head()


# In[3]:


# Assigning the 'price' column to the 'response' variable
response = car_prices_us[['price']]


# In[4]:


# Assigning all of the columns except 'price' to the 'predictors' variable
predictors = car_prices_us.drop(['price'], axis=1)


# In[5]:


# Number of cars in the dataset
count_cars = car_prices_us.shape[0]

# Printing the number of cars in the dataset
print(f"The dataset includes {count_cars} cars.")


# In[6]:


# Number of columns (features) in the 'predictors' variable
num_columns = car_prices_us.shape[1]

# Printing the number of predictor columns
print(f"There are {num_columns} columns in the dataframe.")


# In[7]:


#importing additional libraries
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[8]:


# Selecting numeric predictors for correlation analysis
numeric_columns = car_prices_us.iloc[:,10:]
numeric_columns = numeric_columns.drop(["compressionratio", "stroke", "symboling", "peakrpm",
                                        "horsepower", "carlength", "citympg", "carheight"], 
                                       axis=1)


# In[9]:


corr_matrix = numeric_columns.corr()


# In[10]:


# Generating the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()


# In[11]:


# Removing predictors that have a correlation above 0.8 and below 0.2 and updating variable
predictors = predictors.drop(["compressionratio", "stroke", "symboling", "peakrpm",
                                        "horsepower", "carlength", "citympg", "carwidth", "carheight"],
                            axis=1)


# In[12]:


# Changing the non-numeric data to numeric 0 or 1
predictors=pd.get_dummies(predictors, drop_first=True)


# In[13]:


# Number of columns in predictors variable after removing columns and adding dummy features
num_columns = predictors.shape[1]
print(f"There are {num_columns} columns in the 'predictors' variable after preprocessing.")


# In[14]:


# R score for these variables
r_score = car_prices_us['enginesize'].corr(car_prices_us['boreratio'])

print(f"R score: {r_score}")


# In[15]:


# Which predictors had a weak relationship with the price response?
predictors_Lst = ['stroke', 'highwaympg', 'horsepower', 'carlength']

# The correlation of each predictor with the response 'price'
for predictor in predictors_Lst:
    correlation = car_prices_us[predictor].corr(car_prices_us['price'])
    print(f"Correlation of '{predictor}' with price: {correlation}")


# In[16]:


# Identifying predictors with weak correlation with 'price'
weak_relationships = [predictor for predictor in predictors_Lst if abs(car_prices_us[predictor].corr(inpDf['price'])) < 0.2]
print("Predictors with a weak relationship with 'price':", weak_relationships)


# In[17]:


#Building a linear regression model
linear_regression = skl.LinearRegression()
linear_regression.fit(predictors,response)
print(linear_regression.coef_)
print(linear_regression.intercept_)


# In[18]:


# Evaluate the model
r_squared = linear_regression.score(predictors,response)
print("R_squared: ",r_squared)


# In[19]:


# Calculate the adjusted R-squared using formula below
n = len(response)
k = predictors.shape[1]
adjusted_r_squared = 1-((1-r_squared)*(n-1)/(n-k-1))
print("Adjusted R_squared: ",adjusted_r_squared)


# In[20]:


response_predictions = linear_regression.predict(predictors)
residuals = response - response_predictions
print(residuals.mean())
print(residuals.std())


# In[21]:


plt.hist(residuals,bins=100, color="m")
plt.title("Distribution of Residuals")
plt.show()


# In[22]:


plt.scatter(response_predictions,residuals, color="b")
plt.title("Homoscedasticity")
plt.xlabel("Response Predictions")
plt.ylabel("Residuals")
plt.show()


# In[23]:


# The results are not homoscedastic
#Creating our test and train splits to try other methods
predictors_train, predictors_test, response_train, response_test = train_test_split(
    predictors, response, test_size=0.2, random_state=42)


# In[24]:


#Building and testing decision tree
tree_regressor = DecisionTreeRegressor(random_state=0)
tree_regressor.fit(predictors_train,response_train)
tree_predictions = tree_regressor.predict(predictors_test)


# In[25]:


# mae stands for mean absolute error
# mse stands for squared absolute error
tree_mae = mean_absolute_error(response_test,tree_predictions)
tree_mse = mean_squared_error(response_test,tree_predictions)
tree_r_squared = r2_score(response_test,tree_predictions)
print("Decision Tree Mean Absolute Error: ",tree_mae)
print("Decision Tree Mean Squared Error:" ,tree_mse)
print("Decision Tree R squared: ",tree_r_squared)


# In[26]:


#Building and testing k nearest neighbours
knn_regressor = KNeighborsRegressor(n_neighbors=3)
knn_regressor.fit(predictors_train,response_train)
knn_predictions = knn_regressor.predict(predictors_test)


# In[27]:


# mae stands for mean absolute error
# mse stands for squared absolute error
knn_mae = mean_absolute_error(response_test,knn_predictions)
knn_mse = mean_squared_error(response_test,knn_predictions)
knn_r_squared = r2_score(response_test,knn_predictions)
print("KNN Mean Absolute Error: ",knn_mae)
print("KNN Mean Squared Error: ",knn_mse)
print("KNN R squared: ",knn_r_squared)


# In[28]:


#Building and testing support vector regression
svr_regression = SVR(kernel="rbf", C=100, gamma=0.1, epsilon=0.1)
svr_regression.fit(predictors_train, response_train["price"])
svr_predictions = svr_regression.predict(predictors_test)


# In[29]:


# mae stands for mean absolute error
# mse stands for squared absolute error
svr_mae = mean_absolute_error(response_test,svr_predictions)
svr_mse = mean_squared_error(response_test,svr_predictions)
svr_r_squared = r2_score(response_test,svr_predictions)
print("SVR Mean Absolute Error: ",svr_mae)
print("SVR Mean Squared Error: ",svr_mse)
print("SVR R squared: ",svr_r_squared)


# In[ ]:




