import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

# read the data and store data in Df
train = pd.read_csv('../data/train.csv')
test = pd.read_csv('../data/test.csv')

# target object
y = train['SalePrice']

features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = train[features]
test_X = test[features]

# split into validation and training data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

# define model
rf_model = RandomForestRegressor(random_state=1)

# fit model
rf_model.fit(X, y)

test_predictions=rf_model.predict(test_X)

#output = pd.DataFrame({'Id': test.Id,a'SalePrice': test_predictions})
#output.to_csv('submission.csv', index=False)
