
# coding: utf-8

# In[1]:

import sys
import pandas as pd
import utils
import utils_bux
import featuretools as ft
from sklearn.externals import joblib
from dateutil.relativedelta import relativedelta
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import r2_score
import lime
import lime.lime_tabular

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("white")


def main(users_from, users_till):
    # ### DEFINE THE PIPELINE PARAMETERS

    # In[2]:

    show_report = False
    save_model = True

    # the timeframe of extracted users

    # users_from = '2016-10-01'
    # users_till = '2017-09-30'
    cohort_size = 500

    # the timeframe of extracted behavioral data
    interval = '3 weeks'

    # the type of the prediction problem
    # 'regression', 'binary classification', 'multiclass classification'
    prediction_problem_type = 'binary classification'

    # multiclass values
    medium_value = 5
    high_value = 50

    # number of the most important features to extract
    number_of_features = 20

    print("Pipeline parameters defined")


    # ### CONNECT TO THE DATABASE

    # In[3]:

    conn, cur = utils.connect_to_db()


    # ### BUILD ENTITY TABLES AND LABELS

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


    # #### Labels

    # In[7]:

    labels = utils_bux.build_target_values(cur=cur,
                                           medium_value=medium_value,
                                           high_value=high_value)


    # ### CREATE THE ENTITY SET

    # In[8]:

    es = utils_bux.create_bux_entity_set(cohorts, user_details, daily_transactions)
    es


    # ### FEATURE ENGINEERING (DFS) FOR ALL FEATURES

    # In[9]:

    from featuretools.primitives import (Sum, Std, Max, Min, Mean,
                                     Count, PercentTrue, NUnique, 
                                     Day, Week, Month, Weekday, Weekend)


    trans_primitives = [Day, Week, Month, Weekday, Weekend]
    agg_primitives = [Sum, Std, Max, Min, Mean, Count, PercentTrue, NUnique]


    fm_encoded, features_encoded = utils.calculate_feature_matrix(es,
                                                                 "users",
                                                                 trans_primitives=trans_primitives,
                                                                 agg_primitives=agg_primitives,
                                                                 max_depth=2)
    X = fm_encoded.reset_index().merge(labels)


    # ### TRAINING  ON ALL FEATURES

    # In[10]:

    # define the labels based on the prediction problem type
    X, y = utils.make_labels(X, prediction_problem_type)
    # split the data into training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    # train the model
    model = utils.rf_train(X_train, y_train, prediction_problem_type)
    # extract the most important features
    top_features = utils.feature_importances(model, features_encoded, n=number_of_features)
    # save the top features
    ft.save_features(top_features, "top_features")
    print("All features built and the most important features saved")


    # ### FEATURE ENGINEERING (DFS) FOR TOP FEATURES

    # In[11]:

    fm = utils.calculate_feature_matrix_top_features(es, top_features)
    X = fm.reset_index().merge(labels)
    print("Top features built")


    # ### TRAINING AND PREDICTION ON TOP FEATURES

    # In[12]:

    # define the labels based on the prediction problem type
    X, y = utils.make_labels(X, prediction_problem_type)
    # split the data into training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    # fit the model
    model = utils.rf_train(X_train, y_train, prediction_problem_type)
    print("Model trained on top features")


    # ### SAVE THE MODEL

    # In[13]:

    if save_model == True:
        joblib.dump(model, 'models/model.pkl')
        print("Model saved")
    else:
        print("Model not saved")


    # ### REPORT

    # In[ ]:

    if show_report:
        
        print("REPORT: \n \n \n")
        y_pred = utils.rf_predict(model, X_test, prediction_problem_type)
        
        print("Top features:\n")
        top_features_print = pd.DataFrame([str(feature).split(":")[1].split(">")[0] for feature in top_features])
        top_features_print.columns = ['Feature name']
        print(top_features_print)
        print("\n")
        
        if prediction_problem_type == "binary classification":
            
            
            # CONFUSION MATRIX WITHOUT THRESHOLDING
        
    #         print("Confusion matrix before thresholding (threhold = 0.5): \n")
    #         y_pred_round = y_pred.round(0)
    #         cm = confusion_matrix(y_test, y_pred_round)
    #         # title = 'Customer lifetime value prediction (Confusion matrix)'
    #         utils.plot_confusion_matrix(cm, ['Non-whale', 'Whale'], title="")
    #         print("\n")
        
    #         # THRESHOLDING 
    #         # profit of nudge > cost of nudge significancy -> recall more important than precision
    #         # thresholding (impact of the decision)

    #         nudge_revenue = 15
    #         nudge_cost = 3

    #         max_value_threshold = utils.calculate_threshold_maximum_value(y_pred, nudge_revenue, nudge_cost)
    #         print("\n")


    #         pd.Series(y_pred_round_rf).value_counts()

    #         CONFUSION MATRIX AFTER THRESHOLDING

    #         print("Confusion matrix after thresholding (threshold = " + str(max_value_threshold) + "): \n")
            
    #       y_pred_round = [1 if value > max_value_threshold else 0 for value in y_pred]
            y_pred_round = [1 if value > 0.1 else 0 for value in y_pred]
            cm = confusion_matrix(y_test, y_pred_round)
            # title = 'Customer lifetime value prediction (Confusion matrix)'
            utils.plot_confusion_matrix(cm, ['Non-whale', 'Whale'], title="")
            print("\n")
            
            
            # PERFORMANCE METRICS
            
            # AUC (with ROC curve)
            
            with sns.axes_style("dark"):
                utils.plot_roc_curve(y_test, y_pred_round)
            
            # cross-validation accuracy
            scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
            print("Accuracy: %0.2f (+/- %0.2f) \n" % (scores.mean(), scores.std()))
        
            # cross-validation F1 score
            scores = cross_val_score(model, X, y, cv=5, scoring='f1')
            print("F1: %0.2f (+/- %0.2f) \n" % (scores.mean(), scores.std()))

            # cross-validation Precision score
            scores = cross_val_score(model, X, y, cv=5, scoring='precision')
            print("Precision: %0.2f (+/- %0.2f) \n" % (scores.mean(), scores.std()))

            # cross-validation Recall score
            scores = cross_val_score(model, X, y, cv=5, scoring='recall')
            print("Recall: %0.2f (+/- %0.2f) \n" % (scores.mean(), scores.std()))
            
            
            # LIME - users with 5 highest and lowest values of the most relevant feature
            print("Explanation of predictions of 10 users, 5 with the highest values of the most relevant feature, 5 with the lowest value of the most relevant feature: \n")
            utils.lime_explain_n_users(model, X_train, X_test, y_train, y_test, mapper={0: 'non_whale', 1: 'whale'}, n=10)

            
        
        elif prediction_problem_type == "multiclass classification":
            
            print("Confusion matrix: \n")
            # y_pred_round = y_pred.round(0)
            cm = confusion_matrix(y_test, y_pred)
            # title = 'Customer lifetime value prediction (Confusion matrix)'
            utils.plot_confusion_matrix(cm, ['Low value', 'Medium value', 'High value'], title="")
            
            print(metrics.classification_report(y_test, y_pred))
            
            # LIME - users with 5 highest and lowest values of the most relevant feature
            print("Explanation of predictions of 10 users, 5 with the highest values of the most relevant feature, 5 with the lowest value of the most relevant feature: \n")
            utils.lime_explain_n_users(model, X_train, X_test, y_train, y_test, mapper={0: 'low', 1: 'medium', 2: 'high'}, n=10)

            
            
        elif prediction_problem_type == "regression":
            scores = cross_val_score(model, X, y, cv=5, scoring='r2')
            print("R2 score: %0.2f (+/- %0.2f) \n" % (scores.mean(), scores.std()))
        else:
            print("The prediction problem type not found, choose 'binary classification', 'multiclass classification' or 'regression'. ")


    # In[ ]:

# when running as a script
if __name__ == "__main__":
    users_from = sys.argv[1]
    users_till = sys.argv[2]
    # embed all the code above in the main function
    main(users_from, users_till)

