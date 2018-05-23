# ignore the deprecation warnings for XGBOOST

import warnings

def fxn():
    warnings.warn("deprecated", DeprecationWarning)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import os
import featuretools as ft
import itertools

# training a classifier / booster
import xgboost as xgb
from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
# from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn import metrics



from featuretools.primitives import (Sum, Std, Max, Min, Mean,
                                 Count, PercentTrue, NUnique, 
                                 Day, Week, Month, Weekday, Weekend)


trans_primitives = [Day, Week, Month, Weekday, Weekend]
agg_primitives = [Sum, Std, Max, Min, Mean, Count, PercentTrue, NUnique]






# extraction of entities

import vertica_python
import configparser

def connect_to_db():
    config = configparser.ConfigParser()
    config.read("config.txt")

    conn_info = {'host': config.get("databases","vertica.host"),
                 'port': config.getint("databases","vertica.port"),
                 'user': config.get("databases","vertica.username"),
                 'password': config.get("databases","vertica.password"),
                 'database': config.get("databases","vertica.database"),
                 'read_timeout': 600,
                 'unicode_error': 'strict',
                 'ssl': False}
    conn = vertica_python.connect(**conn_info)
    cur = conn.cursor('dict')   
    print("Connected to the database")
    return conn, cur

def sql_query(cur, query_string):
    cur.execute(query_string)
    df = pd.DataFrame(cur.fetchall())
    return df


def create_entity_set(entityset_name, entityset_quads, entity_relationships):
    
    es = ft.EntitySet(entityset_name)
    
    for es_quad in entityset_quads:
        es.entity_from_dataframe(entity_id=es_quad[0],
                        dataframe=es_quad[1],
                        index=es_quad[2],
                        time_index=es_quad[3])
    
    
    if len(entityset_quads) > 2:
        for rel in entity_relationships:
            es.add_relationship(ft.Relationship(es[rel[0]][rel[2]], es[rel[1]][rel[2]]))
    elif len(entityset_quads) == 2:
        er = entity_relationships
        es.add_relationship(ft.Relationship(es[er[0]][er[2]], es[er[1]][er[2]]))
    return es

# interesting values -> in transactions entity
# es['users']['trading_segment'].interesting_values = ['ICC Trader', 'Crypto Trader']
# example 1: "COUNT(sessions WHERE device = tablet)"
# example 2: "AVG_TIME_BETWEEN(sessions.session_start WHERE device = tablet)"


def calculate_feature_matrix_unparallel(es, target_entity, trans_primitives, agg_primitives, max_depth):
    
    feature_matrix, features = ft.dfs(
    entityset=es,
    target_entity=target_entity,
    trans_primitives=trans_primitives,
    agg_primitives=agg_primitives,
    # variable_types=variable_types,
    #  where_primitives=["count", "avg_time_between"],
    max_depth=max_depth,
    verbose=True
    )
        
    print("{} features generated".format(len(features)))

    fm_encoded, features_encoded = ft.encode_features(feature_matrix, features)
    fm_encoded = fm_encoded.fillna(0)
        
    return fm_encoded, features_encoded



# PARALLELIZED PIPELINE

def load_entity_set(data_dir):


    # cohorts entity
    cohorts = pd.read_csv(os.path.join(data_dir, "cohorts.csv"))

    # users entity
    user_details = pd.read_csv(os.path.join(data_dir, "user_details.csv"))
    user_details['bux_account_created_dts'] = pd.to_datetime(user_details['bux_account_created_dts'])

    # transactions entity
    daily_transactions = pd.read_csv(os.path.join(data_dir, "daily_transactions.csv"))
    
    entityset_name = "bux_clv"

    entityset_quads = (
        # entity name, entity dataframe, entity index, time index
        ['cohorts', cohorts, 'cohort_id', None],
        ['users', user_details, 'user_id', 'bux_account_created_dts'],
        ['transactions', daily_transactions, 'transaction_id', 'date']
        )

    entity_relationships = (
        # parent entity, child entity, key
        ['cohorts', 'users', 'cohort_id'],
        ['users', 'transactions', 'user_id']
    )

    es = create_entity_set(entityset_name, entityset_quads, entity_relationships)
 
    return es

def calculate_feature_matrix_top_features(es, features):
    # label_times, es = labels
    fm = ft.calculate_feature_matrix(features,
                                     entityset=es,
                                     # cutoff_time_in_index=False,
                                     verbose=False)



    # X = fm.reset_index().merge(label_times)

    return fm

def make_labels(X, prediction_problem_type, file_path=None):
    
    if file_path != None:
        labels = pd.read_csv(file_path)
        X = X.reset_index().merge(labels)
        
    # change the labels based on the prediction problem type
    # reg_label binaryclass_label   multiclass_label
    if prediction_problem_type == 'binary classification':
        X.drop(['user_id', 'reg_label', 'multiclass_label'], axis=1, inplace=True)
        X = X.fillna(0)
        y = X.pop('binaryclass_label').astype('int')
    # change the labels based on the prediction problem type
    elif prediction_problem_type == 'multiclass classification':
        X.drop(['user_id', 'reg_label', 'binaryclass_label'], axis=1, inplace=True)
        X = X.fillna(0)
        y = X.pop('multiclass_label').astype('int')

    elif prediction_problem_type == 'regression':
        X.drop(['user_id', 'binaryclass_label', 'multiclass_label'], axis=1, inplace=True)
        X = X.fillna(0)
        y = X.pop('reg_label').astype('int')
    else:
        print("The prediction problem type not found, choose 'binary classification', 'multiclass classification' or 'regression'. ")
        return 0,0

    if "time" in X.columns:
        X.drop("time", axis=1, inplace=True)

    return X, y

def train_test_splitting(X, y, test_size=0.3):
    
    X_train, X_test, y_train, y_test = train_test_split(X,
                                        y,
                                        test_size=test_size)
    
    return X_train, X_test, y_train, y_test

def xgboost_train(X_train, y_train, prediction_problem_type):
    if (prediction_problem_type == "binary classification") or (prediction_problem_type == "multiclass classification"):
        model = xgb.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05).fit(X_train, y_train)
        return model
    elif prediction_problem_type == "regression":
        model = xgb.XGBRegressor(max_depth=3, n_estimators=300, learning_rate=0.05).fit(X_train, y_train)
        return model
    else:
        return "The prediction problem type not found, choose 'classification' or 'regression'"

def xgboost_predict(model, X_test, prediction_problem_type):
    if prediction_problem_type == "binary classification":
        y_pred = model.predict_proba(X_test)
        y_pred = pd.Series([value[1] for value in y_pred])
    elif prediction_problem_type == "multiclass classification":
        y_pred = model.predict(X_test)
    elif prediction_problem_type == "regression":
        y_pred = model.predict(X_test)
    else:
        return("Unknown prediction problem type specified: ", prediction_problem_type)
    
    return y_pred

# REPORT


def plot_roc_curve(y_test, y_pred):
    auc = metrics.roc_auc_score(y_test, y_pred)
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)

    plt.plot(fpr, tpr)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.rcParams['font.size'] = 12
    plt.title('ROC curve, AUC: ' + str(auc))
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.grid(True)


def plot_confusion_matrix(cm,
                          classes=[0, 1],
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Source:
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')


def evaluate_performance(y_pred, y_test):

    precision, recall, fscore, support = score(y_test, y_pred)

    d = pd.DataFrame({'Precision': precision, 'Recall': recall, 'Fscore': fscore, 'Support': support}).T
    d = d[d.columns[1:]]
    d.columns = ['Score']
    d[:3].plot(kind='barh', xlim=(0,1))
    print(d[:3])


# calculate feature importance
# https://github.com/Featuretools/predict-remaining-useful-life/blob/master/utils.py
def feature_importances(model, features, n=10):
    importances = model.feature_importances_
    zipped = sorted(zip(features, importances), key=lambda x: -x[1])
    for i, f in enumerate(zipped[:n]):
        print("%d: Feature: %s, %.3f" % (i+1, f[0], f[1]))

    return([f[0] for f in zipped[:n]])

def feature_importances_xgb(model, feature_names):
    feature_importance_dict = model.get_fscore()
    fs = ['f%i' % i for i in range(len(feature_names))]
    f1 = pd.DataFrame({'f': list(feature_importance_dict.keys()),
                       'importance': list(feature_importance_dict.values())})
    f2 = pd.DataFrame({'f': fs, 'feature_name': feature_names})
    feature_importance = pd.merge(f1, f2, how='right', on='f')
    feature_importance = feature_importance.fillna(0)
    return feature_importance[['feature_name', 'importance']].sort_values(by='importance',
                                                                          ascending=False)



#####################################################################################

# UNUSED CODE




# def load_entity_set(data_dir):


#     # users entity
#     user_details = pd.read_csv(os.path.join(data_dir, "user_details.csv"))
#     user_details['bux_account_created_dts'] = pd.to_datetime(user_details['bux_account_created_dts'])
#     user_details['month_year'] = user_details['bux_account_created_dts'].apply(lambda x: x.strftime('%B-%Y'))

#     # cohort_limit_df = calculate_cohort_limit(user_details, sample_user_ratio)

#     # user_details_temp = pd.DataFrame()
#     # for month_year in cohort_limit_df["month_year"]:
#     #     user_details_temp = user_details_temp.append(limit_users(cohort_limit_df, user_details, month_year))

#     # user_details = user_details_temp

#     # transactions entity
#     daily_transactions = pd.read_csv(os.path.join(data_dir, "daily_transactions.csv"))
#     daily_transactions['date'] = pd.to_datetime(daily_transactions['date'])
#     daily_transactions = daily_transactions[daily_transactions.columns[1:]]
#     daily_transactions.reset_index(inplace=True,drop=True)
#     daily_transactions.reset_index(inplace=True)
#     daily_transactions.rename(columns={'index': 'transaction_id'}, inplace=True)
#     # daily_transactions = daily_transactions[daily_transactions['user_id'].isin(distinct_users)]   
    
        
#     es = ft.EntitySet("bux_cltv")
    
#     es.entity_from_dataframe(entity_id='users',
#                         dataframe=user_details,
#                         index='user_id',
#                         time_index='bux_account_created_dts')
    
#     es.entity_from_dataframe(entity_id='transactions',
#                         dataframe=daily_transactions,
#                         index='transaction_id',
#                         time_index='date')
    
#     es.add_relationship(ft.Relationship(es['users']['user_id'], es['transactions']['user_id']))
    
    
#     return es


# parallelization
# def make_labels(es):
#     # user_details = es["users"].df 
#     # distinct_users = user_details["user_id"].unique()
    
#     label_data = pd.read_csv("data/curcv_1y_6mCustomerValue_2000_3w.csv")

#     # label_data = label_data[label_data['user_id'].isin(distinct_users)]
#     label_data = label_data[label_data.columns[1:]]
#     label_data["label"] = label_data["com"] + label_data["ff"]
#     label_data = label_data[['user_id', 'label']]
#     label_data = label_data.fillna(0)

#     # whale_threshold = label_data["label"].quantile(0.99)
#     whale_threshold = 50
#     label_data["curcv"] = label_data["label"]
#     label_data["label"] = (label_data['curcv'] > whale_threshold).astype(int)

#     return label_data, es


# no used (in the DFS notebook)
# def calculate_feature_matrix(label_data):
#     labels, es = label_data

#     feature_matrix, features = ft.dfs(
#         entityset=es,
#         target_entity="users",
#         trans_primitives=trans_primitives,
#         agg_primitives=agg_primitives,
#         max_depth=2,
#         verbose=True
#     )

#     print("{} features generated".format(len(features)))

#     fm_encoded, features_encoded = ft.encode_features(feature_matrix, features)

#     X = fm_encoded.reset_index().merge(labels)

#     # feature matrix and the encoded features needed to calculate top features
#     return X


# def evaluate_feature_set(X):

#     X.drop(['user_id', 'curcv'], axis=1, inplace=True)
#     X = X.fillna(0)
#     y = X.pop('label').astype('int')
#     y.value_counts()


#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
#     rf_clf = RandomForestClassifier(n_estimators=400, n_jobs=-1)
#     rf_clf.fit(X_train,y_train)
#     y_pred = rf_clf.predict(X_test)
#     f1_score(y_test, y_pred, average=None)[1]

#     cm = confusion_matrix(y_test, y_pred)
#     plot_confusion_matrix(cm, ['Non-whale', 'Whale'], title='Customer lifetime value prediction')


#     # calculate the top 20 features based on feature importance
#     top_features = feature_importances(rf_clf, features_encoded, n=20)
#     print(top_features)

#     # random forest classifer, 5-fold cross-validation
#     scores = cross_val_score(estimator=rf_clf, X=X, y=y, scoring='f1', verbose=True, cv=5, n_jobs=-1)
#     return "F1 %.2f +/- %.2f" % (scores.mean(), scores.std())


# def train_xgb(X_train, labels, params):
#     Xtr, Xv, ytr, yv = train_test_split(X_train.values,
#                                         labels,
#                                         test_size=0.2,
#                                         random_state=0)

#     dtrain = xgb.DMatrix(Xtr, label=ytr)
#     dvalid = xgb.DMatrix(Xv, label=yv)

#     evals = [(dtrain, 'train'), (dvalid, 'valid')]

#     model = xgb.train(params=params, dtrain=dtrain,
#         num_boost_round=227, evals=evals, early_stopping_rounds=60,
#         maximize=False, verbose_eval=10)

#     print('Modeling AUC %.5f' % model.best_score)
#     return model


# def predict_xgb(model, X_test):

#     dtest = xgb.DMatrix(X_test.values)
#     y_test = model.predict(dtest)
#     return y_test


# def feature_importances(X, clf, feats=10):
#     feature_imps = [(imp, X.columns[i]) 
#                     for i, imp in enumerate(clf.feature_importances_)]
#     feature_imps.sort()
#     feature_imps.reverse()

#     for i, f in enumerate(feature_imps[0:feats]):
#         print('{}: {} [{:.3f}]'.format(i + 1, f[1], f[0]))
#     print('-----\n')
#     return [f[1] for f in feature_imps[:feats]]

#  sample of users func1
# def calculate_cohort_limit(user_details, ratio):
#     cohort_limit_df = pd.DataFrame(user_details.groupby("month_year").count()["user_id"]).reset_index()
#     cohort_limit_df.columns = [["month_year", "count"]]
#     cohort_limit_df["count/n"] = (cohort_limit_df["count"] / ratio).astype(int)
#     return cohort_limit_df

# #  sample of users func2
# def limit_users(cohort_limit_df, user_details, month_year):
#     limited_users = int(cohort_limit_df[cohort_limit_df["month_year"] == month_year]["count/n"])
#     limited_df = user_details[user_details["month_year"] == month_year][:limited_users]
#     return limited_df


# # fill the calendar data
# def create_date_range(row):
#     temp_df = pd.DataFrame()
#     temp_df['date'] = pd.Series(pd.date_range(row['date'], pd.to_datetime(row['date']) + pd.DateOffset(days=20)).strftime('%Y-%m-%d'))
#     temp_df['user_id'] = pd.Series([row['user_id']]*len(temp_df))
#     return temp_df

# cohort_limit_df = utils.calculate_cohort_limit(user_details, sample_user_ratio)

# user_details_temp = pd.DataFrame()
# for month_year in cohort_limit_df["month_year"]:
#     user_details_temp = user_details_temp.append(utils.limit_users(cohort_limit_df, user_details, month_year))

# user_details = user_details_temp



# In Notebook


# user_details = pd.read_csv("data/users_1y_6mCustomerValue.csv")
# user_segments_query = """
#         SELECT a.user_id, b.bux_account_created_dts::date, a.segment_value, a.valid_from_date, a.valid_to_date
#         FROM reporting.user_segments a
#         LEFT JOIN reporting.user_details b USING (user_id)
#         WHERE segment_type = 'Trading Segment'
#         AND b.bux_account_created_dts::date BETWEEN '2016-10-01' AND '2017-09-30'

# """
# user_segments = sql_query(user_segments_query)
# user_segments.to_csv("data/user_segments.csv")
# len(user_segments)

# user_segments = pd.read_csv("data/user_segments.csv")
# # user_segments = user_segments[user_segments['user_id'].isin(distinct_users)]
# user_segments = user_segments[user_segments.columns[1:]]
# user_segments.reset_index(drop=True,inplace=True)

# last_valid_before = '9999-12-31'
# last_valid_after = '2100-12-31'
# user_segments['valid_to_date'] = user_segments['valid_to_date'].astype(str)
# user_segments['valid_to_date'] = pd.Series([last_valid_after if value ==  last_valid_before else value for value in user_segments['valid_to_date']])

# user_segments['bux_account_created_dts'] = pd.to_datetime(user_segments['bux_account_created_dts'])
# user_segments['valid_from_date'] = pd.to_datetime(user_segments['valid_from_date'])
# user_segments['valid_to_date'] = pd.to_datetime(user_segments['valid_to_date'])
# user_segments['3w_usage'] = user_segments['bux_account_created_dts'] + pd.DateOffset(days=20)

# user_segments.head()

# # users that didn't change their segment
# # the last value for users that have the last update before 3 weeks after usage
# temp_df_1 = user_segments[(user_segments['valid_to_date'] == last_valid_after) & (user_segments['valid_from_date'] < user_segments['3w_usage'])]
# temp_df_1_uniques = temp_df_1['user_id'].unique()

# # users that changed their segment
# # the last value of users that aren't the first segment
# temp_df_2 = user_segments[~user_segments['user_id'].isin(temp_df_1_uniques)].groupby('user_id').max().reset_index()

# temp_df = temp_df_1[['user_id', 'segment_value']].append(temp_df_2[['user_id', 'segment_value']])
# len(temp_df) == len(user_segments['user_id'].unique())

# user_details = pd.merge(user_details, temp_df, on='user_id', how='left')
# user_details['segment_value'].value_counts().plot(kind='bar');
# # % of users have a segment
# len(temp_df) / len(user_details)
# mode_value = user_details['segment_value'].mode()[0]
# user_details['segment_value'] = user_details['segment_value'].astype(str)
# imputed_values = pd.Series([mode_value if value == 'nan' else value for value in user_details['segment_value']])
# user_details['segment_value'] = imputed_value

# calendar_input = pd.DataFrame(daily_transactions.groupby('user_id').min()['date']).reset_index()
# calendar_input.head()
# calendar_df = pd.DataFrame()

# for index, row in calendar_input.iterrows():
#     temp_df = utils.create_date_range(row)
#     calendar_df = calendar_df.append(temp_df)

# calendar_df['date'] = pd.to_datetime(calendar_df['date'])
# len(calendar_df) / len(calendar_df['user_id'].unique())
# daily_transactions_merged = pd.merge(calendar_df, daily_transactions, on=['user_id', 'date'], how='left')
# len(daily_transactions_merged) / len(daily_transactions)
# daily_transactions = daily_transactions_merged