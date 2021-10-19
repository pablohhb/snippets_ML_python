# Index:
### 1. Preparing Data

### 2. Models:
  - Regression:
    - Single Models:
      - Linear Regression
      - K-Nearest Neighbors Regressor
      - Decission Tree
    - Ensemple Models:
      - Random Forest
      - Gradient Boosting
      - XGB Regressor
  - Classification:
    - Single Models:
      - Logistic Regression
      - K-Nearest Neighbors Classifier
      - Decission Tree Classifier
      - Support Vector Machine (SVM)
    - Ensemble Models:
      - Random Forest Classifier
      - Gradient Boosting Tree Classifier
      - XGB Classifier

### 3. Metrics:
  - Regression:
    - MAE
    - MAPE
    - RMSE
    - Correlation
    - BIAS
  - Classification:
    - Accuracy
    - Precision
    - Recall
    - ROC Curve
    - AUC
    - F1 Score

### 4. Evaluation:
  - Train Test Split
  - Cross Validation
  - Grid Search
  - Randomized Grid Search
 

# 1. Preparing Data:
```python
#Input
X = df[[features]] # pandas DataFrame
# Label
y = df["target] # pandas Series
```
# 2. Models:
## Regression:
### Single Models:
## Linear Regression
```python
# Load the library
from sklearn.linear_model import LinearRegression
# Create an instance of the model
reg = LinearRegression()
# Fit the regressor
reg.fit(X,y)
# Do predictions
reg.predict([[2540],[3500],[4000]])
```
## K-Nearest Neighbors Regressor
```python
# Load the library
from sklearn.neighbors import KNeighborsRegressor
# Create an instance
regk = KNeighborsRegressor(n_neighbors=2)
# Fit the data
regk.fit(X,y)
```
## Decision Tree
```python
# Load the library
from sklearn.tree import DecisionTreeRegressor
# Create an instance
regd = DecisionTreeRegressor(max_depth=3)
# Fit the data
regd.fit(X,y)
```
### Ensemble Models:
## 

# Metrics
## MAE
```python
# Load the scorer
from sklearn.metrics import mean_absolute_error
# Use against predictions
mean_absolute_error(reg.predict(X_test),y_test)
```
## MAPE
```python
# MAPE is not difined in sklearn; ihas to be created:
np.mean(np.abs(reg.predict(X_test)-y_test)/y_test)
```
## RMSE
```python
# Load the scorer
from sklearn.metrics import mean_squared_error
# Use against predictions (we must calculate the square root of the MSE)
np.sqrt(mean_squared_error(reg.predict(X_test),y_test))
```
## Correlation
```python
# Direct Calculation
np.corrcoef(reg.predict(X_test),y_test)[0][1]
# Custom Scorer
from sklearn.metrics import make_scorer
def corr(pred,y_test):
return np.corrcoef(pred,y_test)[0][1]
# Put the scorer in cross_val_score
cross_val_score(reg,X,y,cv=5,scoring=make_scorer(corr))
```
## Bias
```python
# Direct Calculation
np.mean(reg.predict(X_test)-y_test)
# Custom Scorer
from sklearn.metrics import make_scorer
def bias(pred,y_test):
return np.mean(pred-y_test)
# Put the scorer in cross_val_score
cross_val_score(reg,X,y,cv=5,scoring=make_scorer(bias))
```

# Evaluation
## Train Test Split
```python
# Load the library
from sklearn.model_selection import train_test_split
# Create 2 groups each with input and labels
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10)
# Fit only with training data
reg.fit(X_train,y_train)
```
## Cross Valuation
```python
# Load the library
from sklearn.model_selection import cross_val_score
# We calculate the metric for several subsets (determine by cv)
# With cv=5, we will have 5 results from 5 training/test
cross_val_score(reg,X,y,cv=5,scoring="neg_mean_squared_error")
```
## Grid Search
```python
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
reg_test = GridSearchCV(KNeighborsRegressor(),
param_grid={"n_neighbors":np.arange(3,50)})
# Fit will test all of the combinations
reg_test.fit(X,y)

# Best estimator and best parameters
reg_test.best_score_
reg_test.best_estimator_
reg_test.best_params_
```
## Randomized Search
