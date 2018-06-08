# importing the dependencies
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("white")
import os
import featuretools as ft
import itertools
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn import metrics
import lime
import lime.lime_tabular
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
    
    # if cohorts entity is included
    if len(entityset_quads) > 2:
        for rel in entity_relationships:
            es.add_relationship(ft.Relationship(es[rel[0]][rel[2]], es[rel[1]][rel[2]]))
    # if cohorts entity is not included
    elif len(entityset_quads) == 2:
        er = entity_relationships
        es.add_relationship(ft.Relationship(es[er[0]][er[2]], es[er[1]][er[2]]))
    return es

# for training and scoring on all features
def calculate_feature_matrix(es, target_entity, trans_primitives, agg_primitives, max_depth):
    
    feature_matrix, features = ft.dfs(
    entityset=es,
    target_entity=target_entity,
    trans_primitives=trans_primitives,
    agg_primitives=agg_primitives,
    max_depth=max_depth,
    verbose=True
    )
        
    print("{} features generated".format(len(features)))

    fm_encoded, features_encoded = ft.encode_features(feature_matrix, features)
    fm_encoded = fm_encoded.fillna(0)
        
    return fm_encoded, features_encoded


# for training and scoring on the most relevant features
def calculate_feature_matrix_top_features(es, features):

    fm = ft.calculate_feature_matrix(features,
                                     entityset=es,
                                     cutoff_time_in_index=False,
                                     verbose=False)



    # X = fm.reset_index().merge(label_times)

    return fm

# create labels based on the prediction problem type
# (conditional on having all values available, otherwise a change is needed)
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

# training the algorithm based on the prediction problem type
def rf_train(X_train, y_train, prediction_problem_type):
    if (prediction_problem_type == "binary classification") or (prediction_problem_type == "multiclass classification"):
        model = RandomForestClassifier(n_estimators=50, oob_score=True).fit(X_train, y_train)
        # model = xgb.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05).fit(X_train, y_train)
        return model
    elif prediction_problem_type == "regression":
        model = RandomForestRegressor(n_estimators=50, oob_score=True).fit(X_train, y_train)
        # model = xgb.XGBRegressor(max_depth=3, n_estimators=300, learning_rate=0.05).fit(X_train, y_train)
        return model
    else:
        return "The prediction problem type not found, choose 'classification' or 'regression'"

# predicting on the testing set based on the prediction problem type
def rf_predict(model, X_test, prediction_problem_type):
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

    plt.figure(figsize=(6,4))
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
    plt.figure(figsize=(6,4))
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
    plt.savefig("confusion_matrix.png", bbox_inches="tight")
    plt.show()



def calculate_threshold_maximum_value(y_pred, y_test, nudge_revenue, nudge_cost):
    # TP-> by nudging a whale spends 25 euros extra for 5 euros, profit = 20
    # FP -> by nudging a non-whale who does nothing for 5 euros, profit = -5
    # TN -> by not nudging a non-whale nothing happens -> profit = 0
    # FN -> by not nudging a whale they don't spend anything extra, profit = -20 (opportunity cost)


    thresholds = [i/10 for i in range(1,10)]

    tp_value = nudge_revenue - nudge_cost
    fp_value = - nudge_cost
    tn_value = 0
    fn_value = - tp_value

    tp_value, fp_value, tn_value, fn_value

    max_value = {'threshold': 0, 'value':0}

    for cur_threshold in thresholds:
        y_pred_round_rf = [1 if value > cur_threshold else 0 for value in y_pred]
        cm = confusion_matrix(y_test, y_pred_round_rf)
        value = (cm[0][0] * tn_value) + (cm[0][1] * fp_value) + (cm[1][0] * fn_value) + (cm[1][1] * tp_value)
        if value > max_value['value']:
            max_value['threshold'] = cur_threshold
            max_value['value'] = value

    print("Maximum value: " + str(max_value['value']) + ", threshold: " + str(max_value['threshold']))
    return max_value['threshold']


# calculate precision, recall, fscore and support
def evaluate_binary_classification_performance(y_pred, y_test):

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


def lime_explain_n_users(model, X_train, X_test, y_train, y_test, mapper, n):
    explainer = lime.lime_tabular.LimeTabularExplainer(X_train.values,
                                                   feature_names=X_train.columns.values,
                                                   class_names=y_train.map(mapper).unique(),
                                                   discretize_continuous=True)
    
    slice = int(n / 2)
    
    feature_name = X_test.columns[0]
    low_value_ids = list(pd.DataFrame(X_test[feature_name].sort_values(ascending=False)[0:slice]).reset_index()['index'])
    high_value_ids = list(pd.DataFrame(X_test[feature_name].sort_values(ascending=False)[0:slice]).reset_index()['index'])

    high_df = X_test.reset_index()[(X_test.reset_index()['index'].isin(high_value_ids))].drop('index', axis=1).reset_index(drop=True)
    low_df = X_test.reset_index()[(X_test.reset_index()['index'].isin(low_value_ids))].drop('index', axis=1).reset_index(drop=True)

    df = high_df.append(low_df)


    for index, row in df.iterrows():
        exp = explainer.explain_instance(row, model.predict_proba, num_features=5, top_labels=1)
        exp.show_in_notebook(show_table=True, show_all=False)


def show_report(model, X_test, prediction_problem_type):
    print("REPORT: \n \n \n")
    y_pred = rf_predict(model, X_test, prediction_problem_type)
    
    print("Top features:\n")
    top_features_print = pd.DataFrame([str(feature).split(":")[1].split(">")[0] for feature in top_features])
    top_features_print.columns = ['Feature name']
    print(top_features_print)
    print("\n")
    
    if prediction_problem_type == "binary classification":
        
        
        # CONFUSION MATRIX WITHOUT THRESHOLDING
    
        print("Confusion matrix before thresholding (threhold = 0.5): \n")
        y_pred_round = y_pred.round(0)
        cm = confusion_matrix(y_test, y_pred_round)
        # title = 'Customer lifetime value prediction (Confusion matrix)'
        plot_confusion_matrix(cm, ['Non-whale', 'Whale'], title="")
        print("\n")

        # THRESHOLDING 
        # profit of nudge >> cost of nudge -> recall more important than precision
        # thresholding (impact of the decision)

        nudge_revenue = 25
        nudge_cost = 5

        max_value_threshold = calculate_threshold_maximum_value(y_pred, y_test, nudge_revenue, nudge_cost)
        print("\n")

        # CONFUSION MATRIX AFTER THRESHOLDING

        print("Confusion matrix after thresholding (threshold = " + str(max_value_threshold) + "): \n")
        y_pred_round = [1 if value > max_value_threshold else 0 for value in y_pred]

        cm = confusion_matrix(y_test, y_pred_round)
        # title = 'Customer lifetime value prediction (Confusion matrix)'
        plot_confusion_matrix(cm, ['Non-whale', 'Whale'], title="")
        print("\n")
        
        
        # PERFORMANCE METRICS
        
        # AUC (with ROC curve)
        
        with sns.axes_style("dark"):
            plot_roc_curve(y_test, y_pred_round)
        
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
        lime_explain_n_users(model, X_train, X_test, y_train, y_test, mapper={0: 'non_whale', 1: 'whale'}, n=10)

        
    
    elif prediction_problem_type == "multiclass classification":
        
        print("Confusion matrix: \n")
        # y_pred_round = y_pred.round(0)
        cm = confusion_matrix(y_test, y_pred)
        # title = 'Customer lifetime value prediction (Confusion matrix)'
        plot_confusion_matrix(cm, ['Low value', 'Medium value', 'High value'], title="")
        
        print(metrics.classification_report(y_test, y_pred))
        
        # LIME - users with 5 highest and lowest values of the most relevant feature
        print("Explanation of predictions of 10 users, 5 with the highest values of the most relevant feature, 5 with the lowest value of the most relevant feature: \n")
        lime_explain_n_users(model, X_train, X_test, y_train, y_test, mapper={0: 'low', 1: 'medium', 2: 'high'}, n=10)

        
        
    elif prediction_problem_type == "regression":
        scores = cross_val_score(model, X, y, cv=5, scoring='r2')
        print("R2 score: %0.2f (+/- %0.2f) \n" % (scores.mean(), scores.std()))
        scores = cross_val_score(model, X, y, cv=5, scoring='rmse')
        print("RMSE score: %0.2f (+/- %0.2f) \n" % (scores.mean(), scores.std()))

    else:
        print("The prediction problem type not found, choose 'binary classification', 'multiclass classification' or 'regression'. ")

# load the predicted values in the database
def copy_to_database(source_df, destination_table, connection, include_index=False, truncate=False, verbose=False):

    if truncate == True:
        print("Truncating table " + destination_table)
        cur = connection.cursor()
        cur.execute("truncate table " + destination_table)

    columns = list(source_df)
        
    if include_index == True:
        columns.insert(0, source_df.index.name)
    
    columns = " " + str(tuple(source_df)).replace("'","")
        
    tmp_file = tempfile.NamedTemporaryFile()
    source_df.to_csv(tmp_file.name, sep = ',', index = include_index)
    cur = connection.cursor()
    sql = "copy " + destination_table + columns + " from stdin delimiter ',' skip 1"
    
    if verbose == True:
        print(sql)
    
    with open(tmp_file.name, "rb") as fs:
        cur.copy(sql, fs)
        connection.commit()
    tmp_file.close()
    
    return str(len(source_df)) + ' row(s) written to table ' + destination_table