from sklearn.model_selection import StratifiedShuffleSplit
import pandas as pd
import os
import numpy as np

HOUSING_PATH = './datasets/housing'

def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)

housing = load_housing_data()

# train_set, test_set = split_train_test(housing, 0.8)

housing["income_cat"] = pd.cut(housing["median_income"],
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1, 2, 3, 4, 5])

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

# Prepare the Data for Machine Learning Algorithms
housing = strat_train_set.drop("median_house_value", axis=1)  # drop labels for training set
housing_labels = strat_train_set["median_house_value"].copy()

# data transform with medians
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='median')

# Remove the text attribute because median can only be calculated on numerical attributes:
housing_num = housing.drop('ocean_proximity', axis=1)

imputer.fit(housing_num)
print('Imputer stats, \n', imputer.statistics_)

# data transform for the training set

X = imputer.transform(housing_num)
housing_tr = pd.DataFrame(X, columns = housing_num.columns, index=housing.index)

# Handling text and categorical attributes
housing_cat = housing[['ocean_proximity']]
print('Housing category, ', housing_cat.head())

# Encoding is possible this way but we are not using this method.
from sklearn.preprocessing import OrdinalEncoder
ordinal_encoder = OrdinalEncoder()
housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)

print('Housing category encoded, \n', housing_cat_encoded[:5])
print('Ordinal encoder categories, \n', ordinal_encoder.categories_)

# Instead, we use one hot encoder which is more common
from sklearn.preprocessing import OneHotEncoder

cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
print('Housing cat one hot encoded, \n', housing_cat_1hot)

# Again we not using the above hot encoded since it is so huge with many zeros - sparse and inefficient

housing_cat_1hot.toarray() # convert to a dense array - more space saving.

print('Arrayed one hot encoded, \n', housing_cat_1hot.toarray())

# Extra attributes
from sklearn.base import BaseEstimator, TransformerMixin

# column index
rooms_ix, bedrooms_ix, population_ix, households_ix = 3,4,5,6

class CombinedAttributesAdder(BaseEstimator)

















