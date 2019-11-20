import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.impute import  SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
# function to load data

HOUSING_PATH = os.path.join('', 'housing')


def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, 'housing.csv')
    return pd.read_csv(csv_path)


housing = load_housing_data()
print(housing.head())
print(housing.info())
print(housing['ocean_proximity'].value_counts())
print(housing.describe())

# housing.hist(bins=50, figsize=(20, 15))
# plt.show()

train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

housing['income_cat'] = pd.cut(housing['median_income'], bins=[0., 1.5, 3.0, 4.5, 6., np.inf], labels=[1, 2, 3, 4, 5])

housing['income_cat'].hist()
# plt.show()

# guaranteeing that the test set is representative of the overall population
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in split.split(housing, housing['income_cat']):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

strat_test_set['income_cat'].value_counts() / len(strat_test_set)

# removing the income_cat attribute so that the data is back to its original state
for set_ in (strat_train_set, strat_test_set):
    set_.drop('income_cat', axis=1, inplace=True)

# creating a copy of the training set
housing = strat_train_set.copy()

housing.plot(kind='scatter', x='longitude', y='latitude', alpha=0.1)
# plt.show()

housing.plot(kind='scatter', x='longitude', y='latitude', alpha=0.4,
             s=housing['population'] / 100, label='population', figsize=(10, 7), c='median_house_value',
             cmap=plt.get_cmap('jet'), colorbar=True)
# plt.legend()
# plt.show()

# correlations

corr_matrix = housing.corr()

print(corr_matrix['median_house_value'].sort_values(ascending=False))

attributes = ['median_house_value', 'median_income', 'total_rooms', 'housing_median_age']

scatter_matrix(housing[attributes], figsize=(12, 8))

housing.plot(kind='scatter', x='median_income', y='median_house_value', alpha=0.1)

# plt.show()

# attribute combinations
housing['rooms_per_household'] = housing['total_rooms'] / housing['households']
housing['bedrooms_per_room'] = housing['total_bedrooms'] / housing['total_rooms']
housing['population_per_household'] = housing['population'] / housing['households']

corr_matrix = housing.corr()
print(corr_matrix['median_house_value'].sort_values(ascending=False))

housing = strat_train_set.drop('median_house_value', axis=1)
housing_labels = strat_train_set['median_house_value'].copy()

# taking care of missing values using sklearn
housing_num = housing.drop('ocean_proximity', axis=1) # leaving only numerics

imputer = SimpleImputer(strategy='median')
imputer.fit(housing_num)

# transforming the training set
X = imputer.transform(housing_num) # format is numpy array

housing_tr = pd.DataFrame(X, columns=housing_num.columns)

# convert texts to numbers
housing_cat = housing[['ocean_proximity']]
ordinal_encoder = OrdinalEncoder()
housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)
ordinal_encoder.categories_

cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    # ('attribs_adder', Combined)
    ('std_scaler', StandardScaler())
])

num_attribs = list(housing_num)
cat_attribs = ['ocean_proximitnum_pipeliney']

full_pipeline = ColumnTransformer([
    ('num', num_pipeline, num_attribs),
    ('cat', OneHotEncoder(), cat_attribs)
])

housing_prepared = full_pipeline.fit_transform(housing)

lin_reg = LinearRegression()
lin_reg.fit(housing_prepared,housing_labels)


