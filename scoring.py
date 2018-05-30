
# coding: utf-8

# In[10]:

import sys
import pandas as pd
import utils
import utils_bux
import featuretools as ft
from sklearn.externals import joblib



def main(users_from, users_till):
    
    # ### DEFINE PIPELINE PARAMETERS

    # In[11]:

    load_to_database = False
    save_as_csv = False

    # the timeframe of extracted users
    # users_from = '2018-04-01'
    # users_till = '2018-04-30'

    # include all users in each of the cohorts
    cohort_size = 1000000000

    # the timeframe of extracted behavioral data
    interval = '3 weeks'

    # the type of the prediction problem
    # 'regression', 'binary classification', 'multiclass classification'
    prediction_problem_type = 'binary classification'

    print("Pipeline parameters defined")
    print("Extraction of scoring for users from", users_from, "till", users_till)


    # ### CONNECT TO THE DATABASE

    # In[12]:

    conn, cur = utils.connect_to_db()


    # ### BUILD ENTITIES

    # #### Cohorts entity

    # In[13]:

    cohorts = utils_bux.build_cohorts_entity(cur=cur,
                                             users_from=users_from,
                                             users_till=users_till)


    # #### Users entity

    # In[14]:

    user_details = utils_bux.build_users_entity(cur=cur,
                                                users_from=users_from,
                                                users_till=users_till,
                                                interval=interval,
                                                cohorts=cohorts,
                                                cohort_size=cohort_size)


    # #### Transactions entity

    # In[15]:

    daily_transactions = utils_bux.build_transactions_entity(cur=cur,
                                                             interval=interval)


    # ### CREATE THE ENTITY SET

    # In[16]:

    es = utils_bux.create_bux_entity_set(cohorts, user_details, daily_transactions)
    es


    # ### FEATURE ENGINEERING (DFS)

    # In[17]:

    top_features = ft.load_features("top_features", es)
    fm = utils.calculate_feature_matrix_top_features(es, top_features)
    X = fm.reset_index(drop=True).fillna(0)
    print("Features built:\n", list(fm.columns))


    # ### LOADING THE MODEL

    # In[18]:

    model = joblib.load('models/model.pkl')
    print("Model loaded")


    # ### SCORING

    # In[19]:

    y_pred = utils.rf_predict(model, X, prediction_problem_type)
    print("Prediction done")


    # In[20]:

    # save predictions in a csv
    predictions = pd.DataFrame()
    predictions["user_id"] = user_details["user_id"]
    predictions["topic_type"] = "clv_prediction"
    predictions['report_date'] = pd.to_datetime('today').strftime("%Y-%m-%d")
    predictions["model_type"] = "randomforest"
    predictions["class_prediction"] = y_pred
    predictions["prob"] = 0
    predictions = predictions[["topic_type", "report_date", "model_type", "user_id", "class_prediction", "prob"]]
    predictions.head()


    # ### SAVE AS CSV AND/OR LOAD RESULTS INTO THE THE DATABASE

    # In[21]:

    if save_as_csv:
        predictions.to_csv("scoring/results" + users_from + "-" + users_till, index=False)


    # In[22]:

    if load_to_database:
        utils_bux.copy_to_database(predictions, 'db_table_name', conn)


    # In[ ]:

# when running as a script
if __name__ == "__main__":
    users_from = sys.argv[1]
    users_till = sys.argv[2]
    # embed all the code above in the main function
    main(users_from, users_till)

