import os
from datetime import datetime, timedelta
import pandas as pd
from dateutil.relativedelta import relativedelta
import utils
import time
from pandas.tseries.offsets import MonthEnd


# convert a timestamp into a string to run a window function on
def stringify_date(date):
    return str(date)[0:10]

# execute the training script
def train(users_from_train, users_till_train):
	os.system("time python training.py" + " '" + users_from_train + "' '" + users_till_train + "'")

# execute the scoring script
def score(users_from_score, users_till_score):
	os.system("time python scoring.py" + " '" + users_from_score + "' '" + users_till_score + "'")


users_from_score = '2016-01-01'


users_from_score = pd.to_datetime(users_from_score)
users_till_score = users_from_score + MonthEnd(1)
users_till_train = users_from_score + relativedelta(months=-6) + MonthEnd(1)
users_from_train = users_from_score + relativedelta(months=-18) + MonthEnd(1)
print("Training on from: " + stringify_date(users_from_train) + " to " + stringify_date(users_till_train))
print("Scoring from: " + stringify_date(users_from_score) + " to " + stringify_date(users_till_score))

train(stringify_date(users_from_train), stringify_date(users_till_train))
score(stringify_date(users_from_score), stringify_date(users_till_score))


last_month = pd.to_datetime((pd.Timestamp.now() - pd.offsets.MonthBegin(1)).strftime('%Y-%m-%d'))

# 2nd to final iteration
while users_from_score <= last_month:

	users_from_score = users_from_score + relativedelta(months=1)
	users_till_score = users_till_score + MonthEnd(1)


	users_from_train = users_from_score + relativedelta(months=-18)
	users_till_train = users_from_score - relativedelta(months=-6)


	# train, save the model, save the features
	print("Training on from: " + stringify_date(users_from_train) + " to " + stringify_date(users_till_train))
	train(stringify_date(users_from_train), stringify_date(users_till_train))

	# score on the saved model and features, save scores in a csv
	print("Scoring from: " + stringify_date(users_from_score) + " to " + stringify_date(users_till_score))
	score(stringify_date(users_from_score), stringify_date(users_till_score))