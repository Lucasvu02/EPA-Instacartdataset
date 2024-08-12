Shopping Cart Product Prediction
This project uses data from CSV files related to orders to predict which products will be added to users' shopping carts. We employ logistic regression models to perform the predictions and evaluate the model's performance using various class balancing techniques.

Required Data Files
The project requires the following data files:

order_products__train.csv: Data on products in training orders.
order_products__prior.csv: Data on products in prior orders.
orders.csv: Information about the orders.
products.csv: Information about the products.
aisles.csv: Information about store aisles.
departments.csv: Information about store departments.
Steps
1. Read and Explore Data
python
Copy code
import pandas as pd

# Read data from CSV files
order_products_train = pd.read_csv('./order_products__train.csv')
order_products_prior = pd.read_csv('./order_products__prior.csv')
orders = pd.read_csv('./orders.csv')
products = pd.read_csv('./products.csv')
aisles = pd.read_csv('./aisles.csv')
departments = pd.read_csv('./departments.csv')

# Display the shape of the DataFrames
print("aisle: ", aisles.shape)
print("departments: ", departments.shape)
print("orders: ", orders.shape)
print("products: ", products.shape)
print("order_products_prior: ", order_products_prior.shape)
print("order_products_train: ", order_products_train.shape)
2. Data Processing
Merge data from different files to create comprehensive DataFrames containing user and product information.
Create new features from the order_products_prior data for machine learning models.
Add department information from the departments file.
python
Copy code
# Merge data
order_products_train_df = order_products_train.merge(orders.drop('eval_set', axis=1), on='order_id')
order_products_prior_df = order_products_prior.merge(orders.drop('eval_set', axis=1), on='order_id')

# Create new features
user_product_df = (order_products_prior_df.groupby(['product_id', 'user_id'], as_index=False)
                   .agg({'order_id': 'count'})
                   .rename(columns={'order_id': 'user_product_total_orders'}))

train_ids = order_products_train_df['user_id'].unique()
df_X = user_product_df[user_product_df['user_id'].isin(train_ids)]

# Add department information
f_departments_df = products.merge(departments, on='department_id')
f_departments_df = f_departments_df[['product_id', 'department']]
df_X = df_X.merge(f_departments_df, on='product_id')
df_X = pd.concat([df_X, pd.get_dummies(df_X['department'])], axis=1)
del df_X['department']
3. Train and Evaluate Model
Split the data into training and testing sets.
Train logistic regression models with different class balancing techniques.
Evaluate the model performance using F1 scores.
python
Copy code
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import MinMaxScaler

np.random.seed(42)
total_users = df_X['user_id'].unique()
test_users = np.random.choice(total_users, size=int(total_users.shape[0] * 0.2), replace=False)

cv_f1_scores = []
cv_f1_scores_balanced = []
cv_f1_scores_10fit = []

for test_user_set in test_user_sets:
    df_X_tr, df_X_te = df_X[~df_X['user_id'].isin(test_user_set)], df_X[df_X['user_id'].isin(test_user_set)]
    y_tr, y_te = df_X_tr['in_cart'], df_X_te['in_cart']
    X_tr, X_te = df_X_tr.drop(['product_id', 'user_id', 'latest_cart', 'in_cart'], axis=1), \
                 df_X_te.drop(['product_id', 'user_id', 'latest_cart', 'in_cart'], axis=1)

    scaler = MinMaxScaler()
    X_tr = pd.DataFrame(scaler.fit_transform(X_tr), columns=X_tr.columns)
    X_te = pd.DataFrame(scaler.transform(X_te), columns=X_te.columns)

    lr = LogisticRegression(C=10000000)
    lr_balanced = LogisticRegression(class_weight='balanced', C=10000000)
    lr_10x = LogisticRegression(class_weight={1: 6, 0: 1}, C=10000000)

    lr.fit(X_tr, y_tr)
    cv_f1_scores.append(f1_score(lr.predict(X_te), y_te))

    lr_balanced.fit(X_tr, y_tr)
    cv_f1_scores_balanced.append(f1_score(lr_balanced.predict(X_te), y_te))

    lr_10x.fit(X_tr, y_tr)
    cv_f1_scores_10fit.append(f1_score(lr_10x.predict(X_te), y_te))

print("cv_f1_scores: " + str(np.mean(cv_f1_scores)))
print("cv_f1_scores_balanced: " + str(np.mean(cv_f1_scores_balanced)))
print("cv_f1_scores_10fit: " + str(np.mean(cv_f1_scores_10fit)))
Required Libraries
pandas
numpy
matplotlib
scikit-learn
