import os
from datetime import datetime, timedelta
import pandas as pd
from dateutil.relativedelta import relativedelta
import utils
import time
from pandas.tseries.offsets import MonthEnd


# execute the training script
def train(users_from_train, users_till_train):
	os.system("time python training.py" + " '" + users_from_train + "' '" + users_till_train + "'")

# execute the scoring script
def score(users_from_score, users_till_score):
	os.system("time python scoring.py" + " '" + users_from_score + "' '" + users_till_score + "'")
	# os.system("time python scoring_number_users.py" + " '" + users_from_score + "' '" + users_till_score + "'")


# convert the timestamp into a string
def stringify(date):
	return str(date)[0:10]


# define the starting date of training and scoring
starting_date = '2016-01-01'

# first iteration
users_till_train = pd.to_datetime(starting_date) - relativedelta(days=1)
users_from_train = users_till_train + relativedelta(months=-12) + relativedelta(months=-6)
print("Training on from: " + stringify(users_from_train) + " to " + stringify(users_till_train))
train(stringify(users_from_train), stringify(users_till_train))

users_from_score = users_till_train + relativedelta(days=+1)
users_till_score = users_from_score + relativedelta(months=1) - relativedelta(days=1)
print("Scoring from: " + stringify(users_from_score) + " to " + stringify(users_till_score))
score(stringify(users_from_score), stringify(users_till_score))


last_full_month = pd.to_datetime((pd.Timestamp.now() - pd.offsets.MonthBegin(1)).strftime('%Y-%m-%d')) - relativedelta(months=1)


# 2nd to end iteration
while users_from_score <= last_full_month:
	users_till_train = users_till_train + MonthEnd(1)
	users_from_train = users_from_train + relativedelta(months=1)

	users_from_score = users_from_score + relativedelta(months=1)
	users_till_score = users_till_score + MonthEnd(1)

	# train, save the model, save the features
	# print("Training on from: " + stringify(users_from_train) + " to " + stringify(users_till_train))
	# train(stringify(users_from_train), stringify(users_till_train))

	# score on the saved model and features, save scores in a csv
	print("Scoring from:" + stringify(users_from_score) + " to " + stringify(users_till_score))
	score(stringify(users_from_score), stringify(users_till_score))


# train(stringify(users_till_train), stringify(users_from_train))
# score(stringify(users_from_score), stringify(users_till_score))



##############################################################################################

# UNUSED CODE


# query = """
#         SELECT DISTINCT(date_trunc('week', report_date)::date) as report_week
#         FROM reporting.product_volatility
#         ORDER BY 1 ASC

#     """

# cur = utils.connect_to_db()
# report_weeks = utils.sql_query(cur, query)
# report_weeks = report_weeks.rename(index=str, columns={'report_week': 'report_week_before'})
# report_weeks["report_week_till"] = report_weeks["report_week_before"] + timedelta(days=6)


# users_from = '2016-01-01'
# users_till = '2016-01-07'


# train for a month, score for each of the weeks
# for index, row in report_weeks[6:50].iterrows():
#     score(str(row["report_week_before"]), str(row["report_week_till"]))