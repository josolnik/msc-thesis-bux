
# coding: utf-8

# In[7]:

import sys
import pandas as pd
import utils
import utils_bux
import featuretools as ft
from sklearn.externals import joblib

def main(users_from, users_till):

    # ### DEFINE PIPELINE PARAMETERS

    # In[8]:

    load_to_vertica = False

    # the timeframe of extracted users
    # users_from = '2016-01-01'
    # users_till = '2016-01-07'
    cohort_size = 1000000

    # the timeframe of extracted behavioral data
    interval = '6 days'

    # the type of the prediction problem
    # 'regression', 'binary classification', 'multiclass classification'
    prediction_problem_type = 'multiclass classification'

    # # multiclass values
    # medium_value = 5
    # high_value = 50

    print("Pipeline parameters defined")
    print("Extraction of scoring for users from", users_from, "till", users_till)


    # ### CONNECT TO THE DATABASE

    # In[3]:

    # connect to the vertica database, create a cursor
    conn, cur = utils.connect_to_db()


    # ### BUILD ENTITIES

    # #### Cohorts entity

    # In[4]:

    cohorts = utils_bux.build_cohorts_entity(cur=cur,
                                             users_from=users_from,
                                             users_till=users_till)


    # #### Users entity

    # In[5]:

    user_details = utils_bux.build_users_entity(cur=cur,
                                                users_from=users_from,
                                                users_till=users_till,
                                                interval=interval,
                                                cohorts=cohorts,
                                                cohort_size=cohort_size)


    # #### Transactions entity

    # In[6]:

    daily_transactions = utils_bux.build_transactions_entity(cur=cur,
                                                             interval=interval)


    # In[5]:

    ### no need for labels


    # ### CREATE THE ENTITY SET

    # In[6]:

    es = utils_bux.create_bux_entity_set(cohorts, user_details, daily_transactions)
    es


    # ### FEATURE ENGINEERING (DFS)

    # In[7]:

    top_features = ft.load_features("top_features", es)
    fm = utils.calculate_feature_matrix_top_features(es, top_features)
    X = fm.reset_index(drop=True)
    # X = fm.reset_index().merge(labels)
    # X.to_csv("production_features.csv")
    print("Features built:\n", list(fm.columns))


    # ### LOADING THE MODEL

    # In[8]:

    model = joblib.load('models/model.pkl')
    print("Model loaded")


    # ### SCORING

    # In[18]:

    # X = fm.reset_index().merge(labels)
    # X, y = utils.make_labels(X, prediction_problem_type)
    # X_train, X_test, y_train, y_test = utils.train_test_splitting(X, y)
    # model = utils.xgboost_train(X_train, y_train, prediction_problem_type)
    # y_pred = utils.xgboost_predict(model, X_test, prediction_problem_type)
    y_pred = utils.xgboost_predict(model, X, prediction_problem_type)
    print("Prediction done")


    # ### SAVE RESULTS AS A CSV

    # In[19]:

    # utils.sql_query(cur, "SELECT * FROM analytics.model_scoring_predictions LIMIT 10")


    # In[20]:

    # topic_type	report_date	model_type	user_id	class_prediction	prob
    predictions = pd.DataFrame()
    predictions["user_id"] = user_details["user_id"]
    predictions["topic_type"] = "clv_prediction"
    predictions['report_date'] = pd.to_datetime('today').strftime("%Y-%m-%d")
    predictions["model_type"] = "xgboost"
    predictions["class_prediction"] = y_pred
    predictions["prob"] = 0
    predictions = predictions[["topic_type", "report_date", "model_type", "user_id", "class_prediction", "prob"]]
    predictions.head()

# In[37]:

    predictions.to_csv("scoring/clv_prediction_" + users_from + "-" + users_till + ".csv", index=False)


# ### LOAD RESULTS INTO VERTICA

# In[ ]:

# print("Scoring loaded to vertica")


# In[ ]:

# when running as a script
if __name__ == "__main__":
    users_from = sys.argv[1]
    users_till = sys.argv[2]
    main(users_from, users_till)

