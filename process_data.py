import featuretools as ft
import pandas as pd
import os
from tqdm import tqdm


# def make_user_sample(orders, order_products, departments, products, user_ids, out_dir):
def make_user_sample(cohorts, user_details, daily_transactions, user_ids, out_dir): # , target_values):
    # orders_sample = orders[orders["user_id"].isin(user_ids)]
    # orders_keep = orders_sample["order_id"].values
    # order_products_sample = order_products[order_products["order_id"].isin(orders_keep)]
    user_details_sample = user_details[user_details['user_id'].isin(user_ids)]
    user_details_keep = user_details_sample['user_id'].values

    daily_transactions_sample = daily_transactions[daily_transactions['user_id'].isin(user_details_keep)]

    # target_values_sample = target_values[target_values['user_id'].isin(user_ids)]

    try:
        os.mkdir(out_dir)
    except:
        pass

    cohorts.to_csv(os.path.join(out_dir, "cohorts.csv"), index=None)
    user_details_sample.to_csv(os.path.join(out_dir, "user_details.csv"), index=None)
    daily_transactions_sample.to_csv(os.path.join(out_dir, "daily_transactions.csv"), index=None)
    # target_values_sample.to_csv(os.path.join(out_dir, "target_values.csv"), index=None)
    # order_products_sample.to_csv(os.path.join(out_dir, "order_products__prior.csv"), index=None)
    # orders_sample.to_csv(os.path.join(out_dir, "orders.csv"), index=None)
    # departments.to_csv(os.path.join(out_dir, "departments.csv"), index=None)
    # products.to_csv(os.path.join(out_dir, "products.csv"), index=None)


def main():
    data_dir = "data"
    # order_products = pd.concat([pd.read_csv(os.path.join(data_dir,"order_products__prior.csv")),
    #                            pd.read_csv(os.path.join(data_dir, "order_products__train.csv"))])
    # orders = pd.read_csv(os.path.join(data_dir, "orders.csv"))
    # departments = pd.read_csv(os.path.join(data_dir, "departments.csv"))
    # products = pd.read_csv(os.path.join(data_dir, "products.csv"))

    # users per cohort
    users_per_cohort = "2000"

    # cohorts entity
    cohorts = pd.read_csv("data/cohorts.csv")
    # users entity
    user_details = pd.read_csv("data/users_1y_6mCustomerValue_" + users_per_cohort + "_3w.csv")
    # transactions entity
    daily_transactions = pd.read_csv("data/cube_1y_6mCustomerValue_" + users_per_cohort + "_3w.csv")

    # target values
    # target_values = pd.read_csv("data/curcv_1y_6mCustomerValue_" + users_per_cohort + ".csv")


    # users_unique = orders["user_id"].unique()
    users_unique = user_details["user_id"].unique()
    chunksize = 1000
    part_num = 0
    partition_dir = "partitioned_data_chunks"
    try:
        os.mkdir(partition_dir)
    except:
        pass
    for i in tqdm(range(0, len(users_unique), chunksize)):
        users_keep = users_unique[i: i+chunksize]
        make_user_sample(cohorts, user_details, daily_transactions, users_keep, os.path.join(partition_dir, "part_%d" % part_num)) #  target_values)
        part_num += 1

if __name__ == "__main__":
    main()