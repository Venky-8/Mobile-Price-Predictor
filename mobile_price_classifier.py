// Author: Venkatesh Mangnale

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 13:35:52 2019

@author: venky
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
from mlxtend.feature_selection import SequentialFeatureSelector as SFS

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LassoCV
from PIL import Image

st.title('Mobile price classification')

image = Image.open('battery-black.jpg')
st.image(image, caption='A mobile with its back open', use_column_width=True)

DATASET = 'new_train.csv'

@st.cache
def load_data(nrows):
    data = pd.read_csv(DATASET, nrows=nrows)
    lowercase = lambda x: str(x).lower()
    data.rename(lowercase, axis='columns', inplace=True)
    return data

@st.cache
def describe_data(data): 
    return data.describe()

data_load_state = st.text('Loading data...')
data = load_data(1000)
data_load_state.text('Loading data... done!')

if st.sidebar.checkbox('Show raw data'):
    st.subheader('Raw data')
    st.write(data)
        
describe_data = describe_data(data)
if st.sidebar.checkbox('Describe data'):
    st.subheader('Describe data')
    st.write(describe_data)

@st.cache
def get_code():
    code = '''import pandas as pd
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv('new_train.csv')
print(df)

# Input data
# X contains all the columns from battery power to wifi
X = df.iloc[:,1:21]
# y is price range column
y = df.iloc[:,-1]

# Using feature selection to select k best attributes by chi square method
k = 10
best_features = SelectKBest(score_func = chi2 , k = k)
fit = best_features.fit(X , y)
df_scores = pd.DataFrame(fit.scores_)
df_columns = pd.DataFrame(X.columns)
f_score = pd.concat([df_scores , df_columns], axis = 1)
f_score.columns = ['score' , 'features']
print (f_score.nlargest(20 , 'score'))

# These are the selected features
selected_features = f_score[f_score['score'] > 100]
print(selected_features)
selected_columns = selected_features.iloc[:,1:]
print(selected_columns)
sel_columns = selected_columns.values.T.tolist()[0]
new_columns = data[sel_columns]

new_X = new_columns.iloc[:,:]
new_y = df.iloc[:,-1]

X_train , X_test , y_train , y_test = train_test_split(new_X , new_y , test_size = .25 , train_size = .75)
    
# Use multiple linear regression to predict price of mobile
new_X = new_columns.iloc[:,:]
new_y = data.iloc[:,-1]
X_train , X_test , y_train , y_test = train_test_split(new_X , new_y , test_size = .25 , train_size = .75)
clf = LinearRegression()
clf.fit(X_train,y_train)
print (clf.score(X_test , y_test))
print (clf.score(X_test , y_test))'''
    return code
 
if st.sidebar.checkbox('Show code'):
    st.subheader('Code :')
    code = get_code()
    st.code(code, language='python')

# Input data
# X contains all the columns from battery power to wifi
X = data.iloc[:,1:21]
# y is price range column
y = data.iloc[:,-1]

# Using feature selection to select k best attributes by chi square method
k = 10
best_features = SelectKBest(score_func = chi2 , k = k)
fit = best_features.fit(X , y)
df_scores = pd.DataFrame(fit.scores_)
df_columns = pd.DataFrame(X.columns)
f_score = pd.concat([df_scores , df_columns], axis = 1)
f_score.columns = ['score' , 'features']

# These are the selected features
selected_features = f_score[f_score['score'] > 100]
selected_columns = selected_features.iloc[:,1:]
print('Filter Method using chi2 score')
print(selected_columns)
sel_columns = selected_columns.values.T.tolist()[0]
new_columns = data[sel_columns]
    
# Use multiple linear regression to predict price of mobile
new_X = new_columns.iloc[:,:]
new_y = data.iloc[:,-1]
X_train , X_test , y_train , y_test = train_test_split(new_X , new_y , test_size = .25 , train_size = .75)
clf1 = LinearRegression()
clf1.fit(X_train,y_train)
print ('Accuracy is ', clf1.score(X_test , y_test))
    
# Forward Selection Wrapper method
print('Forward Selection')
sfs = SFS(LinearRegression(),
           k_features=6,
           forward=True,
           floating=False,
           scoring = 'r2',
           cv = 0)
sfs.fit(X, y)
ffs = pd.DataFrame(sfs.k_feature_names_)
sel_columns2 = ffs.values.T.tolist()[0]
new_columns2 = data[sel_columns2]
new_X = new_columns2.iloc[:,:]
new_y = data.iloc[:,-1]
X_train2 , X_test2 , y_train2 , y_test2 = train_test_split(new_X , new_y , test_size = .25 , train_size = .75)
clf2 = LinearRegression()
clf2.fit(X_train,y_train)
print("--------Features-------")
print(ffs)
print('Accuracy is ', clf2.score(X_test , y_test))

# Lasso Regularization Embedded method
print('Lasso Regularization')
reg = LassoCV()
reg.fit(X, y)
print("Best alpha using built-in LassoCV: %f" % reg.alpha_)
print("Best score using built-in LassoCV: %f" %reg.score(X,y))
coef = pd.Series(reg.coef_, index = X.columns)
print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables") 
imp_coef = coef.sort_values()
import matplotlib
matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)
imp_coef.plot(kind = "barh")
plt.title("Feature importance using Lasso Model")

if st.sidebar.checkbox('Show Features Selected'):
    st.subheader('Filter method - chi2 score')
    st.write(f_score.nlargest(6 , 'score'))
    st.subheader('Wrapper method - Forward Selection')
    st.write(ffs)
    st.subheader('Embedded method - Lasso Regularization')
    st.pyplot()

# Predict price function
def predict_price(battery_power, int_memory, clock_speed, mobile_wt, px_width, px_height, ram):
    pred1 = [
            [battery_power,
            int_memory,
            mobile_wt,
            px_height,
            px_width,
            ram]
            ]
    st.write('Predicted price of mobile in ₹', )
    pred2 = [
            [battery_power,
            clock_speed,
            mobile_wt,
            px_height,
            px_width,
            ram]
            ]
    prices = [[clf1.predict(pred1)[0]], [clf2.predict(pred2)[0]], [clf1.predict(pred1)[0]]]
    index = ['Filter', 'Wrapper', 'Embedded']
    price_df = pd.DataFrame(prices, columns=['Price'], index=index)
    price_df.round(3)
    price_df['Price'] = '₹' + price_df['Price'].astype(str)
    price_df['Price'] = price_df['Price'].str[0:10]
    st.dataframe(price_df)

# Test your model
if st.sidebar.checkbox('Test Model'):
    st.subheader('Test the model')
    battery_power = st.slider('Battery Power in mAh',500,8000,1250)
    int_memory = st.slider('Internal Memory in GB',2,64,32)
    clock_speed = st.slider('Clock Speed in GHz',0.5,3.0,1.5)
    mobile_wt = st.slider('Mobile Weight in gram',80,200,150)
    px_height = st.slider('Pixel Resolution Height in px',500,4000,1250)
    px_width = st.slider('Pixel Resolution Width in px',1,2000,650)
    ram = st.slider('Ram in MB',500,8000,1250)

    st.write("Battery Power is ", battery_power)
    st.write("Internal Memory is ", int_memory)
    st.write("Clock speed is ", clock_speed)
    st.write("Pixel Resolution Height is ", px_height)
    st.write("Pixel Resolution Width is ", px_width)
    st.write("Ram in MB is ", ram)
    predict_price(battery_power, int_memory, clock_speed, mobile_wt, px_height, px_width, ram)

# Function to display mobiles
def display_mobile(name, image_name, battery_power_, int_memory_, clock_speed_, mobile_wt_, px_height_, px_width_, ram_):
    st.subheader(name)
    image = Image.open(image_name)
    st.image(image, caption=name, use_column_width=True, width=500)
    st.write('Battery Power: ', battery_power_, ' mAh')
    st.write('Internal Memory: ', int_memory_, ' GB')
    st.write('Mobile Weight: ', mobile_wt_, ' gm')
    st.write('Pixel Resolution Height: ', px_height_, ' px')
    st.write('Pixel Resolution Width: ', px_width_, ' px')
    st.write("Ram: ", ram_, " MB")
    predict_price(battery_power_, int_memory_, clock_speed_, mobile_wt_, px_height_, px_width_, ram_)
    
st.subheader('Mobiles')
# Redmi Note 7
display_mobile('Redmi Note 7', 'redmi-note-7.jpg', 3500, 32, 2.2, 196, 1920, 1080, 3072)
# Samsung Galaxy M30
display_mobile('Samsung Galaxy M30', 'samsung-galaxy-m30.jpg', 5000, 64, 1.8, 174, 2280, 1080, 4096)
# Vivo V9
display_mobile('Vivo V9', 'vivo-v9.jpg', 3260, 64, 2.2, 150, 2280, 1080, 4096)