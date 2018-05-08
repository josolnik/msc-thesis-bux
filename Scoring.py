
# coding: utf-8

# In[1]:

import pandas as pd
import utils
import utils_bux
import featuretools as ft
from sklearn.externals import joblib


### DEFINE PIPELINE PARAMETERS

# In[2]:

load_to_vertica = False

# the timeframe of extracted users
users_from = '2016-01-01'
# make relative
users_till = '2017-01-01'
cohort_size = 2000

# the timeframe of extracted behavioral data
interval = '1 week'

# the type of the prediction problem
# 'regression', 'binary classification', 'multiclass classification'
prediction_problem_type = 'multiclass classification'

# multiclass values
medium_value = 5
high_value = 50

print("Pipeline parameters defined")


# ### CONNECT TO THE DATABASE

# In[3]:

# connect to the vertica database, create a cursor
cur = utils.connect_to_db()
print("Connected to the database")


# ### BUILD ENTITIES

# #### Cohorts entity

# In[4]:

cohorts = utils_bux.build_cohorts_entity(cur=cur,
                                         users_from=users_from,
                                         users_till=users_till)
print("Cohorts entity built")


# #### Users entity

# In[ ]:

user_details = utils_bux.build_users_entity(cur=cur,
                                            users_from=users_from,
                                            users_till=users_till,
                                            interval=interval,
                                            cohorts=cohorts,
                                            cohort_size=cohort_size)
print("Users entity built")


# In[ ]:

user_details.head()


# #### Transactions entity

# In[ ]:

daily_transactions = utils_bux.build_transactions_entity(cur=cur,
                                                         interval=interval)
print("Transactions entity built")


# In[ ]:

daily_transactions.head()


# #### Labels

# In[ ]:

labels = utils_bux.build_target_values(cur=cur,
                                       medium_value=medium_value,
                                       high_value=high_value)
print("Target values built")


# In[ ]:

labels.head()


# ### CREATE THE ENTITY SET

# In[11]:

es = utils_bux.create_bux_entity_set(cohorts, user_details, daily_transactions)
es


# ### FEATURE ENGINEERING (DFS)

# In[12]:

top_features = ft.load_features("top_features", es)
fm = utils.calculate_feature_matrix_top_features(es, top_features)
# X = fm.reset_index().merge(labels)
# X.to_csv("production_features.csv")
print("Features built")


# ### LOADING THE MODEL

# In[13]:

model = joblib.load('models/model.pkl')
print("Model loaded")


# ### SCORING

# In[15]:

X = fm.reset_index().merge(labels)


# In[16]:

X, y = utils.make_labels(X, prediction_problem_type)
X_train, X_test, y_train, y_test = utils.train_test_splitting(X, y)
model = utils.xgboost_train(X_train, y_train, prediction_problem_type)
y_pred = utils.xgboost_predict(model, X_test, prediction_problem_type)
print("Prediction done")


# ### LOAD RESULTS INTO VERTICA

# In[13]:

print("Scoring loaded to vertica")

