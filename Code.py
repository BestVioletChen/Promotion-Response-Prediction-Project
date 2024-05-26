#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


# # Load Data
# 
# Upload the files to your Google Drive and mount it in the Colab notebook using the code below.
# 
# Files: `promos.csv.gz`, `test_history.csv.gz`, `train_history.csv.gz`, and `transactions.csv.gz`.

# In[2]:


# This will ask you to give Colab permission to access your Google Drive
# and enter an authorization code.
# See: https://colab.research.google.com/notebooks/io.ipynb#scrollTo=u22w3BFiOveA
from google.colab import drive
drive.mount('/content/drive')


# In[3]:


# MODIFY THIS LINE with the path to where you saved the datafiles on your Google drive
path = '/content/drive/My Drive/Colab Notebooks/RSM8521-Assignment 3'


# In[4]:


promos = pd.read_csv(path + '/promos.csv.gz')
promos.head()


# In[5]:


train_history = pd.read_csv(path + '/train_history.csv.gz', parse_dates=['promodate'])
train_history.head()


# In[6]:


test_history = pd.read_csv(path + '/test_history.csv.gz', parse_dates=['promodate'])
test_history.head()


# In[7]:


# load productsize and amt columns as float32 to reduce memory footprint
transactions = pd.read_csv(path + '/transactions.csv.gz',
                           parse_dates=['date'],
                           dtype={'productsize': 'float32',
                                  'amt': 'float32'})
transactions.head()


# ## Compute RFM Features

# In[8]:


max_date = transactions.date.max()
transactions['last_purchase'] = (max_date - transactions['date']) / np.timedelta64(1, "D")

# Agg transaction data to id
trans_features = transactions.groupby('id').agg({
    'last_purchase': 'min',
    'date': ['nunique', 'min'],
    'amt': 'sum',
})

# Compute RFM columns
trans_features['recency'] = trans_features['last_purchase']['min']
trans_features['frequency'] = (trans_features['date']['nunique']
                               / ((max_date - trans_features['date']['min']) / np.timedelta64(1, "D")))
trans_features['monetary'] = trans_features['amt']['sum']

# Select out required features
trans_features = trans_features[['recency', 'frequency', 'monetary']].reset_index()
trans_features.columns = trans_features.columns.get_level_values(0)
trans_features.head()


# In[9]:


train_history_rfm = pd.merge(train_history, trans_features, on = 'id', how = 'left')
train_history_rfm_time = pd.merge(train_history_rfm, promos, on = 'promo', how = 'left')
train_history_rfm_time.head()


# ## Transaction Feature

# In[10]:


transactions['date'] = pd.to_datetime(transactions['date'])

trans_agg = transactions.groupby('id').agg({
    'date': 'count',  # Total transactions time
    'category': lambda x: x.nunique(),  # Number of diff category
    'amt': 'sum' # Total Amount
}).rename(columns={
    'date': 'total_transactions',
    'category': 'unique_categories',
    'amt': 'total_spent'
})



# In[11]:


train_history_rfm_time_trans = pd.merge(train_history_rfm_time, trans_agg, on='id', how='left')
train_history_rfm_time_trans.head()


# In[89]:


df = train_history_rfm_time_trans
df.head()


# ### Correlation Check

# In[13]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[19]:


df1 = df.drop(columns = ['id','active'], axis = 1)


# In[21]:


corr = df1.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', cbar=True, square=True)
plt.title('Correlation Matrix Heatmap')
plt.show()


# # Data balance check

# In[27]:


# Create a bar plot to visualize the balance of the 'active' column
plt.figure(figsize=(5, 4))
sns.kdeplot(df['active'], fill=True)
plt.title('Density Plot of Active')
plt.xlabel('Value')
plt.ylabel('Density')
plt.show()


# In[28]:


active_counts = df['active'].value_counts()

plt.figure(figsize=(5, 4))
sns.barplot(x=active_counts.index, y=active_counts.values, palette='viridis')
plt.title('Distribution of Active Column')
plt.xlabel('Active Status')
plt.ylabel('Count')
plt.show()


# # Generate train/test features
# 

# In[90]:


df.recency.fillna(365, inplace=True)
df.frequency.fillna(0, inplace=True)
df.monetary.fillna(0, inplace=True)


# In[91]:


df = df.drop(columns = ['total_spent', 'total_transactions','promo','store'],axis = 1)


# In[92]:


df['promodate'] = (df['promodate'] - df['promodate'].min()).dt.days

# Generate dummy variables for 'category', 'brand', and 'region'
df_with_dummies = pd.get_dummies(df, columns=['category', 'brand', 'region', 'manufacturer'])

# Drop the 'id' column as it's not a feature for training
df_with_dummies.drop('id', axis=1, inplace=True)


# In[93]:


# Now, split the dataset into train and test sets
X = df_with_dummies.drop(['active'], axis=1)  # Features
y = df_with_dummies['active']  # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# # Build Model

# In[94]:


from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score


# In[95]:


rf = RandomForestClassifier(n_estimators=100, random_state=64)
rf.fit(X_train, y_train)


# ## Feature Importance Score

# In[96]:


# Get feature importance
feature_importances = rf.feature_importances_

# Create a pandas DataFrame to hold the feature names and their importance
importances_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': feature_importances
})

# Sort the DataFrame by importance in descending order
importances_df.sort_values(by='Importance', ascending=False, inplace=True)

# Print the DataFrame
print(importances_df)


# ### Formal Model Build

# In[97]:


# decide to drop 'promoqty'
X = df_with_dummies.drop(['active','promoqty'], axis=1)  # Features
y = df_with_dummies['active']  # Target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[98]:


rf = RandomForestClassifier(n_estimators=100, random_state=64)
rf.fit(X_train, y_train)


# In[102]:


param_dist = {
    'n_estimators': [150,200,250],
    'max_depth': [6, 8, 10],
    'min_samples_split': [7, 9, 11],
    'max_features': [12,14,16]
}

# Initialize RandomizedSearchCV object
random_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=128),
    param_distributions=param_dist,
    n_iter=10,
    cv=5,
    scoring='roc_auc',
    verbose=1,
    random_state=57,
    n_jobs=-1
)

# Perform the random search
random_search.fit(X_train, y_train)

# Print the best parameters found
print("Best parameters:", random_search.best_params_)



# In[103]:


best_rf = random_search.best_estimator_


# In[104]:


from sklearn.impute import SimpleImputer

# Impute missing values with, for example, the mean
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
X_test_imputed = imputer.fit_transform(X_test)

# Now predict using the imputed test set
y_test_pred = best_rf.predict(X_test_imputed)
y_pred_proba = best_rf.predict_proba(X_test_imputed)[:, 1]

# Calculate performance metrics
test_accuracy = accuracy_score(y_test, y_test_pred)
auc_score = roc_auc_score(y_test, y_pred_proba)

print(f'Test Accuracy Score: {test_accuracy:.2f}')
print(f'Test AUC Score: {auc_score:.2f}')


# # Generate Prediction

# In[105]:


predict_features = test_history.merge(trans_features, on='id', how='left')

test_history_rfm_time = pd.merge(predict_features, promos, on = 'promo', how = 'left')

transactions['date'] = pd.to_datetime(transactions['date'])

trans_agg = transactions.groupby('id').agg({
    'date': 'count',  # Total transactions time
    'category': lambda x: x.nunique(),  # Number of diff category
    'amt': 'sum' # Total Amount
}).rename(columns={
    'date': 'total_transactions',
    'category': 'unique_categories',
    'amt': 'total_spent'
})

test_history_rfm_time_trans = pd.merge(test_history_rfm_time, trans_agg, on='id', how='left')


# In[106]:


df_predict = test_history_rfm_time_trans


# In[107]:


df['promodate'] = pd.to_datetime(df['promodate'])
df_predict['promodate'] = pd.to_datetime(df_predict['promodate'])

# Calculate the number of days since the earliest 'promodate' in the original dataframe
min_promo_date = df['promodate'].min()
df_predict['promodate'] = (df_predict['promodate'] - min_promo_date).dt.days

# Align dummy variables in 'df_predict' with those in 'X_train'
df_predict_with_dummies = pd.get_dummies(df_predict, columns=['category', 'brand', 'region', 'manufacturer'])
df_predict_with_dummies = df_predict_with_dummies.reindex(columns=X_train.columns, fill_value=0)


# In[108]:


predict_out = df_predict[['id','active']].copy()
predict_out.head()


# In[109]:


from sklearn.impute import SimpleImputer

# Initialize the imputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')  # You can choose another strategy like 'median' or 'most_frequent'

# Fit the imputer and transform the dataset to fill in NaNs
df_predict_imputed = imputer.fit_transform(df_predict_with_dummies)

# Use the imputed dataset to make predictions
predict_out['active'] = best_rf.predict_proba(df_predict_imputed)[:, 1]


# In[110]:


predict_out.to_csv('predict.csv', header=True, index=False)
predict_out.head()


# In[111]:


# This will download your prediction files
from google.colab import files
files.download('predict.csv')

