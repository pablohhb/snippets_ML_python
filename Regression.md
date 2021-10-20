# Index:
### 1. Preparing Data

### 2. Models:
  - Regression:
    - Single Models:
      - Linear Regression
      - K-Nearest Neighbors Regressor
      - Decission Tree Regressor
    - Ensemple Models:
      - Random Forest Regressor
      - Gradient Boosting Regressor
  - Classification:
    - Single Models:
      - Logistic Regression
      - K-Nearest Neighbors Classifier
      - Decission Tree Classifier
      - Support Vector Machine (SVM)
    - Ensemble Models:
      - Random Forest Classifier
      - Gradient Boosting Tree Classifier

### 3. Metrics:
  - Regression:
    - MAE -> Mean Absolute Error
    - MAPE -> Mean Absolute Percentage Error
    - RMSE -> Root Mean Squared Error
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
# Train the regressor
reg.fit(X,y)
# Do predictions
reg.predict([[x1],[x2],[x3]])
```
## K-Nearest Neighbors Regressor
```python
# Load the library
from sklearn.neighbors import KNeighborsRegressor
# Create an instance
regk = KNeighborsRegressor(n_neighbors=2)
# Train the data
regk.fit(X,y)
```
## Decision Tree Regressor
```python
# Load the library
from sklearn.tree import DecisionTreeRegressor
# Create an instance
regd = DecisionTreeRegressor(max_depth=3)
# Train the data
regd.fit(X,y)
```
### Ensemble Models:
## Random Forest Regressor
```python
# Load the library
from sklearn.ensemble import RandomForestRegressor
# Create an instance
reg = RandomForestRegressor(max_depth=3,
                            min_samples_leaf = 20,
                            n_estimators = 100)
# Train the data
reg.fit(X,y)
```
## Gradient Boosting Regressor
```python
# Load the library
from sklearn.ensemble import GradientBoostingRegressor
# Create the instance
reg = GradientBoostingRegressor(max_depth=4,
                                min_samples_leaf=20,
                                n_estimators=100)
# Train the data
reg.fit(X,y)
```

## Classification:
### Single Models:
## Logistic Regressor
```python
# Load the library
from sklearn.linear_model import LogisticRegression
# Create an instance of the classifier
clf=LogisticRegression()
# Train the data
clf.fit(X,y)
```
## K-Nearest Neighbors Classifier
```python
# Load the library
from sklearn.neighbors import KNeighborsClassifier
# Create an instance of the classifier
clfk = KNeighborsClassifier(n_neighbors=5)
# Train the data 
clfk.fit(X, y)
```
## Suport Vector Machine (SVN)
```python
# Load the library
from sklearn.svm import SVC
# Create an instance of the classifier
clf = SVC(kernel="linear",C=10)
# Train the data
clf.fit(X, y)
```
## Decission Tree Classifier
```python
# Load the library
from sklearn.tree import DecisionTreeClassifier
# Create an instance of the classifier
clft = DecisionTreeClassifier(min_samples_leaf=20, max_depth=10)
# Train the data
clft.fit(X, y)
```
### Ensemble Models:
## Random Forest Classifier
```python
# Load the library
from sklearn.ensemble import RandomForestClassifier
# Create an instance of the classifier
clf = RandomForestClassifier(max_depth=3,
                             min_samples_leaf=20,
                             n_estimators=100)
# Train the model
clf.fit(X, y)
```
## Gradient Boosting Tree Classifier
```python
# Load the library
from sklearn.ensemble import GradientBoostingClassifier
# Create an instance of the classifier
clf = GradientBoostingClassifier(max_depth=4,
                                   min_samples_leaf=20,
                                   n_estimators=100)
# Train the model
clf.fit(X, y)
```


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
# MAPE is not difined in sklearn; it has to be created:
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
# Load the library
from sklearn.model_selection import GridSearchCV
# Create an instance of the model (example: with KNN regressor)
reg = GridSearchCV(KNeighborsRegressor(),
                   param_grid={"n_neighbors":np.arange(3,50)},
                   cv = 5,
                   scoring = "neg_mean_absolute_error")
 # Train the model
 reg.fit(X, y)

# Get key info from the model
reg.best_score_
reg.best_estimator_
reg.best_params_
```

## Randomized Search
```python
# Load the library
from sklearn.model_selection import RandomizedSearchCV
# Create an instance of the model (example: Decission Tree Regressor)
reg = RandomizedSearchCV(DecisionTreeRegressor(),
                         param_distributions={"max_depth":[2,3,5,8,10],
                                              "min_samples_leaf":[5,10,15,20,30,40]},
                                               cv = 5,
                                               scoring="neg_mean_absolute_error")
# Train the model
reg.fit(X, y)
```
