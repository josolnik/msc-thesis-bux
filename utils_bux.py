import utils
import pandas as pd


### Mungle features

# COHORTS

def build_vix_features(cur, cohort_query_vix_normalized, cohort_query_vix):
    cohort_vix_normalized = utils.sql_query(cur, cohort_query_vix_normalized)
    # fix missing crypto vix values, hard-coded non-normalized vix
    cohort_vix_normalized = cohort_vix_normalized.append({'product_type': 'Crypto', 'average_vix_norm': 100}, ignore_index=True)
    cohorts = utils.sql_query(cur,cohort_query_vix)
    
    cohorts = cohorts.pivot(index='report_week', columns='product_type', values='average_vix').reset_index()

    # normalize by the average value in 2 years
    def normalize_vix(column_name):
        return cohorts[column_name] / int(cohort_vix_normalized[cohort_vix_normalized['product_type'] == column_name]['average_vix_norm'])

    # apply to each of the columns
    for column in cohorts.columns[2:]:
        cohorts[column + "_norm"] = normalize_vix(column)

    cohorts.reset_index(inplace=True)
    cohorts.rename(columns={'index': 'cohort_id'}, inplace=True)
    return cohorts

def build_cohorts_entity(cur, users_from, users_till):


	if pd.to_datetime(users_from) >= pd.to_datetime('2016-01-01'):


	    cohort_query_vix_normalized = """ 
	    
	            SELECT product_type, AVG(vix) as average_vix_norm
	            FROM reporting.product_volatility
	            WHERE report_date BETWEEN '2016-01-01' AND '2017-12-31'
	            GROUP BY 1
	            ORDER BY 1
	        """
	    
	    cohort_query_vix = """ 
	    
	            SELECT date_trunc('week', report_date)::date as report_week, product_type, AVG(vix) as average_vix
	            FROM reporting.product_volatility
	            WHERE report_date BETWEEN '{users_from}' AND '{users_till}'
	            GROUP BY 1,2
	            ORDER BY 1
	        """.format(users_from=users_from,users_till=users_till)
	    
	    print("Cohorts entity built")
	    cohorts = build_vix_features(cur, cohort_query_vix_normalized, cohort_query_vix)
	    return cohorts
	else:
		print("Cohorts entity not built - data not available")
		return ["Empty cohorts entity"]
    

# USERS

def build_time_to_features(cur, time_to_event_query):
    
    # query the data
    time_to_event = utils.sql_query(cur, time_to_event_query)
    
    binary_features = pd.pivot_table(time_to_event, values="did_event", index="user_id", columns="event_name").reset_index()
    binary_features.columns = ["user_id"] + [column + "_did_event" for column in binary_features.columns[1:]]
    
    
    continuous_features = pd.pivot_table(time_to_event, values="hours_till_event", index="user_id", columns="event_name").reset_index()
    continuous_features.columns = ["user_id"] + [column + "_hours_till_event" for column in continuous_features.columns[1:]]
    # replace the features that weren't completed by a user with a high value
    # continuous_features.fillna(500, inplace=True)

    time_to_event_df = pd.merge(binary_features, continuous_features, on='user_id')
    
    return time_to_event_df

def merge_users_features(user_details, users_initial_deposit_replace, cohorts, time_to_event_features):
    
    # merge initial deposit feature
    user_details = pd.merge(user_details, users_initial_deposit_replace, how='left', on='user_id')
    # user_details['initial_deposit_amount_lim'].fillna(0,inplace=True)
    # user_details['days_to_initial_deposit'].fillna(10000, inplace=True)
    
    # merge cohort data
    if len(cohorts) > 1:
    	user_details = pd.merge(user_details, cohorts[['cohort_id', 'report_week']], how='left', on='report_week')
    
    # merge time to event features
    user_details = pd.merge(user_details, time_to_event_features, how='left', on='user_id')
    
    return user_details



def build_users_entity(cur, users_from, users_till, interval, cohorts, cohort_size):
    ### Users basic features
    # nationality

	query_users = """ 

	            CREATE LOCAL TEMPORARY TABLE temp_users ON COMMIT PRESERVE ROWS AS
	            SELECT user_id, platform_type_name, trading_experience, title, network, bux_account_created_dts
	            FROM reporting.user_details
	            WHERE bux_account_created_dts::date BETWEEN '{users_from}' AND '{users_till}'
	            LIMIT {cohort_size} OVER (PARTITION BY date_trunc('month', bux_account_created_dts) ORDER BY RANDOMINT(1000000000));
	        """.format(users_from=users_from,users_till=users_till,cohort_size=cohort_size)

	utils.sql_query(cur, query_users)


	### Time to features

	query_time_to_event = """

	     WITH b AS (SELECT * from meta.bux_events WHERE event_type_id IN (1900024676472176506,
	                                                                     6343323308049849851,
	                                                                     260687741711542992,
	                                                                     2193673344032322170,
	                                                                     617792645229736175,
	                                                                     1280889395537804445,
	                                                                     1498292874234664683,
	                                                                     1793263894043445928,
	                                                                     2092633507578741654,
	                                                                     2389031425774675520,
	                                                                     3784994946525179991,
	                                                                     4665563951835963528,
	                                                                     5283878399315578611,
	                                                                     6070753512604390982,
	                                                                     8843571341358969389))
	   SELECT
	         a.user_id
	       , b.event_name
	       , case
	              when c.event_type_id is not null
	              then 1
	              else 0
	         end                                                       as did_event
	       , datediff ('hour', a.bux_account_created_dts, c.event_dts) as hours_till_event
	    FROM
	         temp_users a
	    CROSS JOIN
	         b
	    LEFT JOIN
	         reporting.user_session_events_first_occurence c
	      ON
	         c.user_id = a.user_id
	     AND c.event_type_id = b.event_type_id
	     AND c.event_dts::date <= a.bux_account_created_dts::date + interval '{interval}'

	""".format(interval=interval)

	time_to_event_features = build_time_to_features(cur, query_time_to_event)



    ### Initial deposit features

	query_get_initial_deposit = """
	    SELECT a.user_id, (amount * c.exchange_rate)::numeric(20,2) as initial_deposit_amount_lim, DATEDIFF(DAY, bux_account_created_dts, created_dts) as days_to_initial_deposit
	    FROM temp_users a
	    JOIN reporting.transactions b 
	    ON a.user_id = b.user_id 
	    AND b.created_dts < a.bux_account_created_dts + interval '{interval}'
	    JOIN reporting.exchange_rates_eur c
	    ON c.currency = b.currency
	    AND c.report_date = b.created_dts::date    
	    WHERE b.transaction_type = 'DEPOSIT' OR (b.transaction_type = 'CASH_TRANSFER' AND b.amount * c.exchange_rate >= 50)
	    LIMIT 1 OVER (PARTITION BY a.user_id ORDER BY b.created_dts asc)

	""".format(interval=interval)

	users_initial_deposit_replace = utils.sql_query(cur, query_get_initial_deposit)


	### Trading segment feature

	query_get_users = """
	        SELECT a.*, nvl(b.segment_value, 'No Trades') as trading_segment, date_trunc('week', bux_account_created_dts)::date as report_week
	        FROM temp_users a
	        LEFT JOIN reporting.user_segments b
	        ON b.segment_type = 'Trading Segment'
	        AND b.user_id = a.user_id
	        AND a.bux_account_created_dts::date + interval '{interval}' between b.valid_from_date and b.valid_to_date
	        LIMIT 1 OVER (PARTITION BY a.user_id ORDER BY valid_to_date desc)
	""".format(interval=interval)

	user_details = utils.sql_query(cur, query_get_users)

	### Merge all users features

	user_details = merge_users_features(user_details, users_initial_deposit_replace, cohorts, time_to_event_features)
	print("Users entity built with", len(user_details), "users")
	return user_details


# TRANSACTIONS

def mungle_transactions(cur, query_transactions):
    daily_transactions = utils.sql_query(cur, query_transactions)
    daily_transactions['date'] = pd.to_datetime(daily_transactions['date'])
    daily_transactions.reset_index(inplace=True,drop=True)
    daily_transactions.reset_index(inplace=True)
    daily_transactions.rename(columns={'index': 'transaction_id'}, inplace=True)
    return daily_transactions


def build_transactions_entity(cur,interval):
    query_transactions= """

        SELECT a.user_id,
        a.date,
        trades_sb_invested_amount,
        financing_deposits_amount,
        trades_sb_short,
        trades_sb_long,
        view_position,
        trades_sb_open_positions,
        total_session_duration,
        education_topic_read,
        trades_sb_commission,
        trades_fb_forex_open,
        trades_sb_forex_open,
        conversion_to_sb,
        trades_sb_forex_average_leverage,
        trades_sb_forex_average_leverage,
        (trades_fb_forex_average_leverage + trades_sb_forex_average_leverage) as trades_fbsb_forex_average_leverage,
        (trades_fb_forex_average_leverage + trades_sb_forex_average_leverage*10) as trades_fbsb10_forex_average_leverage
        FROM temp_users b
        JOIN calendar c ON c.date BETWEEN b.bux_account_created_dts::date and b.bux_account_created_dts::date + interval '{interval}'
        LEFT JOIN reporting.cube_daily_user a ON a.user_id = b.user_id AND a.date = c.date


    """.format(interval=interval)

    daily_transactions = mungle_transactions(cur, query_transactions)
    print("Transactions entity built with", len(daily_transactions), "transactions")
    return daily_transactions


### Mungle target values

def mungle_curv_cv(cur, query_curcv, medium_value, high_value):
    df = utils.sql_query(cur, query_curcv)
    df["reg_label"] = (df["com"] + df["ff"]).astype(float)
    df = df[['user_id', 'reg_label']].fillna(0)
    df["binaryclass_label"] = (df['reg_label'] > high_value).astype(int)
    df['multiclass_label'] = [2 if value > high_value else 1 if value > medium_value and value < high_value else 0 for value in df['reg_label']]
    return df


def build_target_values(cur, medium_value, high_value):
    query_curcv= """ 
        
        SELECT a.user_id
        , bux_account_created_dts as date
        , SUM(decode(b.transaction_type, 'COMMISSION',-amount * nvl(c.exchange_rate, 1),0))::numeric(20,2) as com
        , SUM(decode(b.transaction_type, 'DIVIDEND',-amount * nvl(c.exchange_rate, 1),0))::numeric(20,2) as div
        , SUM(decode(b.transaction_type, 'FINANCING_FEE',-amount * nvl(c.exchange_rate, 1),0))::numeric(20,2) as ff
        FROM temp_users a
        LEFT JOIN reporting.transactions b ON b.created_dts < a.bux_account_created_dts + interval '6 months' AND a.user_id = b.user_id
        LEFT JOIN reporting.exchange_rates_eur c on c.report_date = b.created_dts::date and c.currency = b.currency
        GROUP BY 1,2

    """

    labels = mungle_curv_cv(cur, query_curcv, medium_value=medium_value, high_value=high_value)
    print("Target values built")
    return labels



### CREATE ENTITY SET

def create_bux_entity_set(cohorts, user_details, daily_transactions):

	entityset_name = "bux_clv"

	# cohorts == list of length 1 if no data available for vix
	if len(cohorts) > 1:
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
	else:
		entityset_quads = (
		    # entity name, entity dataframe, entity index, time index
		    ['users', user_details, 'user_id', 'bux_account_created_dts'],
		    ['transactions', daily_transactions, 'transaction_id', 'date']
		    )

		entity_relationships = (
		    # parent entity, child entity, key
		    ['users', 'transactions', 'user_id']
		)

	es = utils.create_entity_set(entityset_name, entityset_quads, entity_relationships)
	print("Entity set built")
	return es