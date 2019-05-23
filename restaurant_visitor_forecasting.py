#!/usr/bin/env python
# coding: utf-8

# In[174]:


'''
----------------------------------------------------------------------------
Project Name: Forecasting Restaurant Visitors
Rishika Kapoor
----------------------------------------------------------------------------

Restaurant Dataset Description:
-------------------------------
The restaurant reservation and visitor datasets are selected from Kaggle.com and 
it was submitted by the organization Recruit Holding Co., Ltd for the challenge 
mainly aimed to find a solution to predict restaurants visitors. The dataset is provided in 
multiple CSV files that have restaurants info, reservations info and visit info.    

The dataset provided includes the reservation data from Hot Pepper Gourmet application 
(online restaurant search and reservation application) and the restaurant visitor data 
retrieved from AirREGI reservation and cash registration system.  The data sets consisted of 
the data collected between 2016 to 2017 April as well as training data sets.  

Here are the data files provided:
air_store_info.csv:   The restaurant master file from the AirREGI system.
    air_store_id - Restaurant identifier 
    air_genre_name - Restaurant genre 
    air_area_name - Area where the restaurant is located 
    latitude -  Coordinates of restaurant location 
    longitude - Coordinates of restaurant location 

air_reserve.csv:  The reservation data retrieved from the AirREGI system.
    air_store_id - Restaurant identifier 
    visit_datetime - The date and time of the planned restaurant visit 
    reserve_datetime - The reservation date and time 
    reserve_visitors - The number of visitors associated with the reservation 

air_visit_data.csv: The actual visit data retrieved from the AirREGI reservation system. 
    air_store_id - Restaurant identifier 
    visit_date - The actual visit date time 
    visitors - The number of actual visitors 

hpg_store_info.csv:   The restaurant master file from the Hot Pepper Gourmet application. 
    hpg_store_id - Restaurant identifier 
    air_genre_name - Restaurant genre 
    air_area_name - Area where the restaurant is located 
    latitude -  Coordinates of restaurant location 
    longitude - Coordinates of restaurant location 

hpg_reserve.csv:  The reservation data retrieved from the Hot Pepper Gourmet application. 
    hpg_store_id - Restaurant identifier 
    visit_datetime - The date and time of the planned restaurant visit 
    reserve_datetime - The reservation date and time 
    reserve_visitors - The number of visitors associated with the reservation 

date_info.csv: Provides the date, day of the week and a boolean flag indicating if the day is a holiday or not. 

store_id_relation.csv: One-to-one mapping of store identifiers retrieved from AirREGI and Hot Pepper Gourmet application.  

X Features:
-----------
restaurant_id               object
visit_date          datetime64[ns]
reserve_visitors             int64
air_area_name               object
air_genre_name              object
latitude                   float64

Y = F(X):
---------
visitors    int64

Restaurant Dataset reference:
--------------------------
https://www.kaggle.com/c/recruit-restaurant-visitor-forecasting/overview

Dataset downloaded from:
------------------------
https://www.kaggle.com/c/recruit-restaurant-visitor-forecasting/data

'''


# In[175]:


# Ignores the warnings.
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Imports the required packages.
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns ;sns.set()
get_ipython().run_line_magic('matplotlib', 'inline')


# In[176]:


# --------------------------------------------------------------------------
# Preprocessing of  Restaurant master data from both HPG and AirREGI systems.
# Creates the consolidated Restaurant master dataframe.
# --------------------------------------------------------------------------

# Loads the HPG store master data to the pandas dataframe.
hpg_store_info_df = pd.read_csv("../restaurant-dataset/hpg_store_info.csv")
print(f'Number of Restaurants in HPG System: {len(hpg_store_info_df)}')

# Loads the AirREGI store master data to the pandas dataframe.
air_store_info_df = pd.read_csv("../restaurant-dataset/air_store_info.csv")
print(f'Number of Restaurants in AirREGI System: {len(air_store_info_df)}')

# Loads the AirREGI and HPG store mapping reference data to the pandas dataframe.
store_mapped_ref_df = pd.read_csv("../restaurant-dataset/store_id_relation.csv")

# Removes the store mapping data from HPG data. Performs left outer join.
# hpg_store_info_df_preprocessed has the restaurants that has ONLY HPG system.
hpg_store_info_preprocessed_df = hpg_store_info_df.merge(store_mapped_ref_df, on='hpg_store_id', how='left',indicator=True)
hpg_store_info_preprocessed_df = hpg_store_info_preprocessed_df.query("_merge == 'left_only'")
hpg_store_info_preprocessed_df = hpg_store_info_preprocessed_df.drop('_merge', axis=1)
print(f'Number of Restaurants with ONLY HPG System: {len(hpg_store_info_preprocessed_df)}')

# Removes the store mapping data from AIRREGI data. Performs left outer join.
# air_store_info_df_preprocessed has the restaurants that has ONLY AirREGI system.
air_store_info_preprocessed_df = air_store_info_df.merge(store_mapped_ref_df, on='air_store_id', how='left',indicator=True)
air_store_info_preprocessed_df = air_store_info_preprocessed_df.query("_merge == 'left_only'")
air_store_info_preprocessed_df = air_store_info_preprocessed_df.drop('_merge', axis=1)
print(f'Number of Restaurants with ONLY AirREGI System: {len(air_store_info_preprocessed_df)}')

# Concats the HPG and AirREGI data frames (this dataframe does not contain records found in mapping data). 
restaurant_store_master_df = pd.concat([hpg_store_info_preprocessed_df,air_store_info_preprocessed_df],ignore_index=True,sort=True).drop_duplicates().reset_index(drop=True)


# Concats the Restaurant data that has both HPG and AirREGI systems.
restaurant_store_master_df = restaurant_store_master_df.merge(store_mapped_ref_df, on='hpg_store_id', how='outer')

# Renames the column.
restaurant_store_master_df.rename(index=str, columns={"air_store_id_x": "air_store_id"},inplace=True)

# Creates the restaurant_id.
restaurant_store_master_df['restaurant_id'] = restaurant_store_master_df['air_store_id'].astype(str) + '_ID_' + restaurant_store_master_df['hpg_store_id'].astype(str)

# Deletes the unwanted columns.
del restaurant_store_master_df['air_store_id_y']
del restaurant_store_master_df['air_store_id']
del restaurant_store_master_df['hpg_store_id']

# Rearranges the columns.
restaurant_store_master_df = restaurant_store_master_df[['restaurant_id', 'air_area_name', 'hpg_area_name','air_genre_name','hpg_genre_name','latitude','longitude']]

print(f'Number of Restaurants with BOTH AirREGI and HPG Systems: {len(store_mapped_ref_df)}')
print(f'Total Number of Restaurants: {len(restaurant_store_master_df)}')
restaurant_store_master_df.to_csv('./temp/restaurant_master_consolidated.csv')
print(restaurant_store_master_df.head(5))


# In[177]:


# -------------------------------------------------------------------------------
# Preprocessing of  Restaurant reservation data from both HPG and AirREGI systems.
# Creates the consolidated Restaurant reservation dataframe.
# -------------------------------------------------------------------------------
# Loads the HPG reservation data to the pandas dataframe.
hpg_store_reservation_df = pd.read_csv("../restaurant-dataset/hpg_reserve.csv")

#print(hpg_store_reservation_df.head(5))
print(f'Number of HPG reservation records: {len(hpg_store_reservation_df)}')

# Loads the AirREGI reservation data to the pandas dataframe.
air_store_reservation_df = pd.read_csv("../restaurant-dataset/air_reserve.csv")

#print(air_store_reservation_df.head(5))
print(f'Number of AirREGI reservation records: {len(air_store_reservation_df)}')

# Preprocess HPG reservation data and add mapping store identifiers from mapping reference data.
hpg_store_reservation_preprocessed_df = hpg_store_reservation_df.merge(store_mapped_ref_df, on='hpg_store_id', how='left',indicator=True)
print(f'Number of HPG preprocessed reservation records: {len(hpg_store_reservation_preprocessed_df)}')

# Preprocess AirREGI reservation data and add mapping store identifiers from mapping reference data.
air_store_reservation_preprocessed_df = air_store_reservation_df.merge(store_mapped_ref_df, on='air_store_id', how='left',indicator=True)
print(f'Number of AirREGI preprocessed reservation records: {len(air_store_reservation_preprocessed_df)}')

# Concats the HPG and AirREGI reservation data frames. 
restaurant_reservation_df = pd.concat([hpg_store_reservation_preprocessed_df,air_store_reservation_preprocessed_df],ignore_index=True,sort=True).reset_index(drop=True)

# Creates the restaurant_id.
restaurant_reservation_df['restaurant_id'] = restaurant_reservation_df['air_store_id'].astype(str) + '_ID_' + restaurant_reservation_df['hpg_store_id'].astype(str)

# Retain only dates in reserve_datetime and visit_datetime as the visits and forecasting is at date level.
restaurant_reservation_df['reserve_datetime'] = restaurant_reservation_df['reserve_datetime'].astype('datetime64[ns]').dt.date 
restaurant_reservation_df['visit_datetime'] = restaurant_reservation_df['visit_datetime'].astype('datetime64[ns]').dt.date 
restaurant_reservation_df.rename(index=str, columns={"reserve_datetime": "reserve_date", "visit_datetime": "visit_date"},inplace=True)
print(f'Total Number of reservation records: {len(restaurant_reservation_df)}')


restaurant_reservation_df = restaurant_reservation_df[['restaurant_id','air_store_id', 'hpg_store_id', 'reserve_date', 'reserve_visitors','visit_date']]
restaurant_reservation_df = restaurant_reservation_df.groupby(['restaurant_id','visit_date']).sum()
restaurant_reservation_df.reset_index(inplace=True)

print(f'Total Number of reservation records grouped by restaurant_id and visit_date): {len(restaurant_reservation_df)}')
print(restaurant_reservation_df.head(5))
restaurant_reservation_df.to_csv('./temp/restaurant_reservation_aggregated.csv')


# In[178]:


# --------------------------------------------------------------------------
# Preprocessing of  Restaurant visits data from AirREGI systems.
# Creates the consolidated Restaurant visits dataframe.
# --------------------------------------------------------------------------
# Loads the AirREGI visits data to the pandas dataframe.
air_store_visit_df = pd.read_csv("../restaurant-dataset/air_visit_data.csv")

#print(hpg_store_reservation_df.head(5))
print(f'Number of AirREGI visit records: {len(air_store_visit_df)}')

# Preprocess AirREGI visits data and add mapping store identifiers from mapping reference data.
air_store_visit_preprocessed_df = air_store_visit_df.merge(store_mapped_ref_df, on='air_store_id', how='left',indicator=True)

# Creates the restaurant_id.
air_store_visit_preprocessed_df['restaurant_id'] = air_store_visit_preprocessed_df['air_store_id'].astype(str) + '_ID_'  + air_store_visit_preprocessed_df['hpg_store_id'].astype(str)
air_store_visit_preprocessed_df = air_store_visit_preprocessed_df[['restaurant_id','air_store_id', 'hpg_store_id', 'visit_date', 'visitors']]

# Deletes the unwanted columns.
del air_store_visit_preprocessed_df['air_store_id']
del air_store_visit_preprocessed_df['hpg_store_id']

print(f'Number of AirREGI preprocessed visit records: {len(air_store_visit_preprocessed_df)}')
print(air_store_visit_preprocessed_df.head(5))
air_store_visit_preprocessed_df.to_csv('./temp/air_restaurant_visit_consolidated.csv')


# In[179]:


# --------------------------------------------------------------------------------
# Builds the TRAINING DATASETS 
# Input Xs from reservation data: restaurand_id, visit_date,reserve_visitors; 
# Input Xs from restaurant master data:  air_area_name, hpg_area_name, air_genre_name,hpg_genre_name, latitude, longitude
# Y = visitors
# --------------------------------------------------------------------------------

# Step-1: Merge the visits and reservation data frames.
# Convert data types so that the merge works properly.
air_store_visit_preprocessed_df.visit_date = pd.to_datetime(air_store_visit_preprocessed_df.visit_date)
restaurant_reservation_df.visit_date = pd.to_datetime(restaurant_reservation_df.visit_date)
air_store_visit_preprocessed_df.restaurant_id = air_store_visit_preprocessed_df.restaurant_id.astype(str)
restaurant_reservation_df.restaurant_id = restaurant_reservation_df.restaurant_id.astype(str)

# Merge the visits and reservation data frames.
restaurant_visit_training_df = air_store_visit_preprocessed_df.merge(restaurant_reservation_df, on=['restaurant_id','visit_date'], how='left',indicator=True)
restaurant_visit_training_df = restaurant_visit_training_df.drop('_merge', axis=1)

# Removes the emtpy reserve_vistors as those records can't be used by training.
restaurant_visit_training_df.dropna(subset=['reserve_visitors'],inplace = True)

# Fix the data type.
restaurant_visit_training_df['reserve_visitors'] = restaurant_visit_training_df['reserve_visitors'].astype(np.int64)
# Removing the redundant area name and genre name columns.
del restaurant_store_master_df['hpg_area_name'] 
del restaurant_store_master_df['hpg_genre_name'] 

# Removing the longitude column as it has strong correlation with latitude for the given dataset.
del restaurant_store_master_df['longitude'] 


# Step-2: Merge the Step-1 dataframe with the restaurant store master data. 
# Merge the visits and reservation data frames.
restaurant_visit_training_df = restaurant_visit_training_df.merge(restaurant_store_master_df, on=['restaurant_id'], how='left',indicator=True)
restaurant_visit_training_df = restaurant_visit_training_df.drop('_merge', axis=1)

print(restaurant_visit_training_df.head(5))
print(f'Total number of records in training dataset: {len(restaurant_visit_training_df)}')
restaurant_visit_training_df.to_csv('./temp/restaurant_visit_training_df.csv')


# In[180]:


# Extract the input features.
# restaurant_id,visit_date,reserve_visitors,air_area_name,hpg_area_name,air_genre_name,hpg_genre_name, latitude,longitude
# Extract the output feature.(visitors)
X_restaurant_visit_training_df = restaurant_visit_training_df.drop('visitors',axis=1)
print("FEATURES X:")
print("-----------")
print(X_restaurant_visit_training_df.dtypes)
print(X_restaurant_visit_training_df.head(10))
X_restaurant_visit_training_df.to_csv('./temp/X_restaurant_visit_training.csv')
print("Y = F(X)")
print("-----------")
Y_restaurant_visit_training_df = restaurant_visit_training_df[['visitors']]
print(Y_restaurant_visit_training_df.dtypes)
print(Y_restaurant_visit_training_df.head(10))


# In[181]:


# Function that preprocess the training and testing datasets to convert non-numeric values to numeric classes
# using Label Encoder.
from sklearn import preprocessing
def convert_to_numeric_classes(data):

    label_encoder = preprocessing.LabelEncoder()
    data['restaurant_id'] = label_encoder.fit_transform(data.restaurant_id.astype(str))
    data['visit_date'] = label_encoder.fit_transform(data.visit_date.astype(str))
    data['reserve_visitors'] = label_encoder.fit_transform(data.reserve_visitors.astype(str))
    data['air_genre_name'] = label_encoder.fit_transform(data.air_genre_name.astype(str))
    data['air_area_name'] = label_encoder.fit_transform(data.air_area_name.astype(str))
    data['latitude'] = label_encoder.fit_transform(data.latitude.astype(str))
    return data


# In[182]:


# Invokes the function to perform the pre-processing to replace all strings with numeric values. 
X_restaurant_visit_training_df = convert_to_numeric_classes(X_restaurant_visit_training_df)


# In[183]:


# -----------------------------------------------------------
# Visualizes the Restaurant visits data from AirREGI systems.
# -----------------------------------------------------------

sns.set(style="darkgrid")
plt.figure(figsize=(16, 6))
g = sns.lineplot(x="visit_date", y="visitors",linewidth=1.5,color='blue', data=restaurant_visit_training_df)
g.set(xticklabels=[])
plt.title('Restaurant Visit - Training Data')
plt.xlabel('Days (01/13/2016 to 04/22/2017)')
plt.ylabel('Total number of visits')


# In[184]:


# Gets the correlation coefficient matrix for training dataset.
training_corrmat = X_restaurant_visit_training_df.corr()

# Plots the heatmap for correlation coefficient.
# As the heatmap indicates, the selected features have low correlations amongst themselves.
# Changes the size of the figure to 10X10(inch X inch).
f, ax = plt.subplots(figsize=(10,10))

# Draws the heatmap using seaborn.
sns.heatmap(training_corrmat, vmax=.8, square=True, annot=True,  cmap="YlGnBu")
f.tight_layout()
plt.title("Training Dataset - Feature correlation",fontsize=16,fontweight='bold',color='black')


# In[185]:


# Trains the model with Stochastic Gradient Descent.
# Creates the SGDC classifier
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)

scaler = StandardScaler()
scaler.fit(X_restaurant_visit_training_df)  
X_restaurant_visit_training_df = scaler.transform(X_restaurant_visit_training_df)

stochastic_gradient_regressor = SGDRegressor(max_iter=50, penalty=None, eta0=0.1,random_state=0)

# Fits the classifier with the training data (features X and target Y = attack)
stochastic_gradient_regressor.fit(X_restaurant_visit_training_df, Y_restaurant_visit_training_df.values.ravel())

# Prints learned coeficients and intercept.
print(f'Coefficients: {stochastic_gradient_regressor.coef_}')
print(f'Intercept: {stochastic_gradient_regressor.intercept_}')
print(f'Best Score: {stochastic_gradient_regressor.score}')


# In[186]:


# Loads the input dataset into a Pandas dataframe.
X_restaurant_input_original_df = pd.read_csv("../restaurant-dataset/restaurant_input_data.csv")
X_restaurant_input_original_df = X_restaurant_input_original_df.drop([X_restaurant_input_original_df.columns[0]] ,  axis='columns')
X_restaurant_input_df = X_restaurant_input_original_df.copy(deep=True)
X_restaurant_input_df = X_restaurant_input_df.drop([X_restaurant_input_original_df.columns[0]] ,  axis='columns')
X_restaurant_input_df = convert_to_numeric_classes(X_restaurant_input_df)


# Transforms the data.
scaler.fit(X_restaurant_input_df)  
X_restaurant_input_df = scaler.transform(X_restaurant_input_df)

# Performs the forecasting.
Y_restaurant_visit_forecast_nparray = stochastic_gradient_regressor.predict(X_restaurant_input_df)
Y_restaurant_visit_forecast_nparray = np.int64(Y_restaurant_visit_forecast_nparray)
Y_restaurant_visit_forecast_df = pd.DataFrame(Y_restaurant_visit_forecast_nparray)

# Creates the results dataframe and writes to csv file.
restaurant_visit_forecast_df = pd.merge(X_restaurant_input_original_df, Y_restaurant_visit_forecast_df, left_index=True, right_index=True)
restaurant_visit_forecast_df.rename(columns={ restaurant_visit_forecast_df.columns[7]: 'forecasted_visits' },inplace=True)

# Removes the unwanted columns.
del restaurant_visit_forecast_df['air_area_name']
del restaurant_visit_forecast_df['reserve_visitors']
del restaurant_visit_forecast_df['air_genre_name']
del restaurant_visit_forecast_df['latitude']
restaurant_visit_forecast_df.drop([restaurant_visit_forecast_df.columns[0]] ,  axis='columns',inplace=True)




print(restaurant_visit_forecast_df.head(10))
restaurant_visit_forecast_df.to_csv('./results/restaurant_reservation_forecasting_results.csv')


# In[187]:


# -------------------------------------------------
# Visualizes the forecasted Restaurant visits data 
# -------------------------------------------------

sns.set(style="darkgrid")
plt.figure(figsize=(16, 6))
g = sns.lineplot(x="visit_date", y="forecasted_visits",linewidth=1.5,color='blue', data=restaurant_visit_forecast_df)
g.set(xticklabels=[])
plt.title('Restaurant Visit - Forecasted Data')
plt.xlabel('Days (01/13/2019 to 04/22/2020)')
plt.ylabel('Total number of visits')

