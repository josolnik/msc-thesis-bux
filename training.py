
# coding: utf-8

# In[1]:

import sys
import pandas as pd
import utils
import utils_bux
import featuretools as ft
from sklearn.externals import joblib
from dateutil.relativedelta import relativedelta



def main(users_from, users_till):

    # ### DEFINE THE PIPELINE PARAMETERS

    # In[2]:

    def stringify(date):
        return str(date)[0:10]


    # In[3]:

    show_report = True
    save_model = True

    # the timeframe of extracted users

    starting_date = '2016-01-01'

    users_till = stringify(pd.to_datetime(starting_date) - relativedelta(days=1)) 
    users_from = stringify(pd.to_datetime(starting_date) + relativedelta(months=-12))

    cohort_size = 2000

    # the timeframe of extracted behavioral data
    interval = '6 days'

    # the type of the prediction problem
    # 'regression', 'binary classification', 'multiclass classification'
    prediction_problem_type = 'multiclass classification'

    # multiclass values
    medium_value = 5
    high_value = 50

    # number of the most important features to extract
    number_of_features = 20

    print("Pipeline parameters defined")


    # In[4]:

    # starting_date = '2016-01-01'

    # users_till = pd.to_datetime(starting_date) - relativedelta(days=1)
    # users_from = users_till_train + relativedelta(months=-12)

    # users_from, users_till


    # ### CONNECT TO THE DATABASE

    # In[5]:

    # connect to the vertica database, create a cursor
    conn, cur = utils.connect_to_db()


    # ### BUILD ENTITY SET AND LABELS

    # #### Cohorts entity

    # In[6]:

    cohorts = utils_bux.build_cohorts_entity(cur=cur,
                                             users_from=users_from,
                                             users_till=users_till)


    # #### Users entity

    # In[7]:

    user_details = utils_bux.build_users_entity(cur=cur,
                                                users_from=users_from,
                                                users_till=users_till,
                                                interval=interval,
                                                cohorts=cohorts,
                                                cohort_size=cohort_size)


    # #### Transactions entity

    # In[8]:

    daily_transactions = utils_bux.build_transactions_entity(cur=cur,
                                                             interval=interval)


    # #### Labels

    # In[9]:

    labels = utils_bux.build_target_values(cur=cur,
                                           medium_value=medium_value,
                                           high_value=high_value)


    # ### CREATE THE ENTITY SET

    # In[10]:

    # entities
    # cohorts = pd.read_csv("data/cohorts.csv")
    # user_details = pd.read_csv("data/users_1y_6mCustomerValue_2000_3w.csv")
    # daily_transactions = pd.read_csv('data/cube_1y_6mCustomerValue_2000_3w.csv')

    # target values
    # labels = pd.read_csv('data/curcv_1y_6mCustomerValue_2000_3w.csv')


    # In[11]:

    # def create_entity_set(entityset_name, entityset_quads, entity_relationships):
        
    #     es = ft.EntitySet(entityset_name)
        
    #     for es_quad in entityset_quads:
    #         es.entity_from_dataframe(entity_id=es_quad[0],
    #                         dataframe=es_quad[1],
    #                         index=es_quad[2],
    #                         time_index=es_quad[3])
        
        
    #     if len(entityset_quads) > 2:
    #         for rel in entity_relationships:
    #             es.add_relationship(ft.Relationship(es[rel[0]][rel[2]], es[rel[1]][rel[2]]))
    #     elif len(entityset_quads) == 2:
    #         er = entity_relationships
    #         es.add_relationship(ft.Relationship(es[er[0]][er[2]], es[er[1]][er[2]]))
    #     return es


    # In[12]:

    # create_entity_set('bux_clv', entityset_quads, entity_relationships)


    # In[13]:

    # problem with the fillna (initial deposit lim and days to initial deposit)
    es = utils_bux.create_bux_entity_set(cohorts, user_details, daily_transactions)
    es


    # ### FEATURE ENGINEERING (DFS) FOR ALL FEATURES

    # In[ ]:

    from featuretools.primitives import (Sum, Std, Max, Min, Mean,
                                     Count, PercentTrue, NUnique, 
                                     Day, Week, Month, Weekday, Weekend)


    trans_primitives = [Day, Week, Month, Weekday, Weekend]
    agg_primitives = [Sum, Std, Max, Min, Mean, Count, PercentTrue, NUnique]


    fm_encoded, features_encoded = utils.calculate_feature_matrix_unparallel(es,
                                                                             "users",
                                                                             trans_primitives=trans_primitives,
                                                                             agg_primitives=agg_primitives,
                                                                             max_depth=2)
    X = fm_encoded.reset_index().merge(labels)


    # ### TRAINING  ON ALL FEATURES

    # In[ ]:

    # define the labels based on the prediction problem type
    X, y = utils.make_labels(X, prediction_problem_type)
    # split the data into training and testing
    X_train, X_test, y_train, y_test = utils.train_test_splitting(X, y)
    # fit the model
    model = utils.xgboost_train(X_train, y_train, prediction_problem_type)
    # predict on the testing set
    # y_pred = utils.xgboost_predict(model, X_test, prediction_problem_type)
    # extract the most important features
    top_features = utils.feature_importances(model, features_encoded, n=number_of_features)
    # save the top features
    ft.save_features(top_features, "top_features")
    print("All features built and the most important features saved")


    # ### FEATURE ENGINEERING (DFS) FOR TOP FEATURES

    # In[ ]:

    fm = utils.calculate_feature_matrix_top_features(es, top_features)
    X = fm.reset_index().merge(labels)
    print("Top features built")


    # ### TRAINING AND PREDICTION ON TOP FEATURES

    # In[ ]:

    # define the labels based on the prediction problem type
    X, y = utils.make_labels(X, prediction_problem_type)
    # split the data into training and testing
    X_train, X_test, y_train, y_test = utils.train_test_splitting(X, y)
    # fit the model
    model = utils.xgboost_train(X_train, y_train, prediction_problem_type)
    print("Model trained on top features")


    # In[ ]:

    # len(X.columns)


    # ### SAVE THE MODEL

    # In[ ]:

    if save_model == True:
        joblib.dump(model, 'models/model.pkl')
        print("Model saved")
    else:
        print("Model not saved")


    # In[ ]:

# when running as a script
if __name__ == "__main__":
    users_from = sys.argv[1]
    users_till = sys.argv[2]
    main(users_from, users_till)


# ### REPORT

# In[2]:

# if show_report == True:
#     # execute the report
#     print("Report shown")


# In[1]:

# y_pred_round_xgb = [1 if value > 0.5 else 0 for value in y_pred]

# from sklearn import metrics

# def plot_roc_curve(y_test, y_pred):
#     auc = metrics.roc_auc_score(y_test, y_pred)
#     fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)

#     plt.plot(fpr, tpr)
#     plt.xlim([0.0, 1.0])
#     plt.ylim([0.0, 1.0])
#     plt.rcParams['font.size'] = 12
#     plt.title('ROC curve, AUC: ' + str(auc))
#     plt.xlabel('False Positive Rate (1 - Specificity)')
#     plt.ylabel('True Positive Rate (Sensitivity)')
#     plt.grid(True)
    
# plot_roc_curve(y_test, y_pred)


# In[3]:

# cm = confusion_matrix(y_test, y_pred.round(0))
# utils.plot_confusion_matrix(cm, ['Non-whale', 'Whale'], title='Customer lifetime value prediction (Confusion matrix)')


# In[4]:

# scores = cross_val_score(model, X, y, cv=5, scoring='f1')
# print("F1: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


# #### LIME

# In[5]:

# import lime
# import lime.lime_tabular

