import utils
import pandas as pd


### Build features


# The SQL and Python code is not present for confidentiality reasons
# The data was wrangled using Pandas, used utils script as a helper script
# the main function that was used was to query the database
# e.g. df = utils.sql_query(cur, query_name)

# COHORTS

def build_vix_features(cur, cohort_query_vix_normalized, cohort_query_vix):


	"""

	The data munging of the vix features, returning the cohorts table with the wrangle features

	"""


def build_cohorts_entity(cur, users_from, users_till):



	"""

	Query the data for the cohorts entity (vix features) based on the users_from and users_till parameters
	
	"""
    

# USERS

def build_time_to_features(cur, time_to_event_query):



	"""

	Query the data for the Users entity - time to a specific event features
	
	"""
 

def merge_users_features(user_details, users_initial_deposit_replace, cohorts, time_to_event_features):



	"""

	Merge all the Users entity features into one dataframe

	"""


def build_users_entity(cur, users_from, users_till, interval, cohorts, cohort_size):



	"""
	
	1. Create the temporary table
	2. Create the querries - query_time_to_event, query_get_initial_deposit, query_get_users
	3. Execute all queries and merge the features using merge_users_features function

	"""


# TRANSACTIONS

def mungle_transactions(cur, query_transactions):


	"""

	Wrangling of the transactions entity

	"""

def build_transactions_entity(cur,interval):



	"""

	Extract the transactions entity using the query_transactions query, 

	"""



### Mungle target values

def mungle_curv_cv(cur, query_curcv, medium_value, high_value):


	"""

	Create the target values using the query_curcv and medium_value and high_value - for the regression and classification task



	"""


def build_target_values(cur, medium_value, high_value):


	"""

	Extract the target values using query_curcv, followed by wrangling of the entity table using mungle_curv_cv function.

	Return the labels for all of the prediction problem types


	"""


### CREATE ENTITY SET

def create_bux_entity_set(cohorts, user_details, daily_transactions):


	"""

	Create the entity sets using triplets - entity name, entity table and time index

	"""