import utils
import pandas as pd


### Build features


# The SQL and Python code is not present for confidentiality reasons
# The data was wrangled using Pandas and used utils script as a helper script
# the main function that was used was to query the database
# e.g. df = utils.sql_query(cur, query_name)

# COHORTS

def build_vix_features(cur, cohort_query_vix_normalized, cohort_query_vix):


	"""

	1. Data munging of the vix features, returning the cohorts table with the wrangled features

	"""


def build_cohorts_entity(cur, users_from, users_till):



	"""

	1. Query the data for the cohorts entity (vix features) based on the users_from and users_till parameters
	2. Wrangle the features calling build_vix_features
	3. Return the wrangled cohorts table
	 
	
	"""
    

# USERS

def build_time_to_features(cur, time_to_event_query):



	"""

	1. Query the data for the Users entity - time to a specific event features
	
	"""
 

def merge_users_features(user_details, users_initial_deposit_replace, cohorts, time_to_event_features):



	"""

	1. Merge all the Users entity features into one dataframe

	"""


def build_users_entity(cur, users_from, users_till, interval, cohorts, cohort_size):



	"""
	
	1. Create the temporary table
	2. Create the queries - with query_time_to_event, query_get_initial_deposit and query_get_users functions
	3. Execute all queries and merge the features using merge_users_features function

	"""


# TRANSACTIONS

def mungle_transactions(cur, query_transactions):


	"""

	1. Wrangling of the transactions entity features

	"""

def build_transactions_entity(cur,interval):



	"""

	1. Extract the transactions entity using the query_transactions query
	2. Wrangle the features using mungle_transactions
	3. Return the transactions entity


	"""



### Mungle target values

def mungle_curv_cv(cur, query_curcv, medium_value, high_value):


	"""

	1. Wrangle the target values
	2. Return the result

	"""


def build_target_values(cur, medium_value, high_value):


	"""

	1. Extract the target values using query_curcv
	2. Wrangle the entity table using mungle_curv_cv function
	3. Return the wrangled labels for all of the prediction problem types


	"""


### CREATE ENTITY SET

def create_bux_entity_set(cohorts, user_details, daily_transactions):


	"""

	1. Create the entity sets using triplets - entity name, entity table and time index

	"""