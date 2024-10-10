# Metrics collected:

1. SESSION_ID: Unique identifier of the session.
2. Click_Image: Indicator of whether the user clicked on an image (0 or 1).
3. Read_Review: Indicator of whether the user has read a review (0 or 1).
4. Category_View: Indicator whether the user viewed the category (0 or 1).
5. Read_Details: Indicator whether the user has read the product details (0 or 1).
6. Video_View: Indicator whether the user has watched a product video (0 or 1).
7. Add_to_List: Indicator whether the user has added the product to the list (0 or 1).
8. Compare_Prc: Indicator if the user has compared prices (0 or 1).
9. View_Similar: Indicator of whether the user has viewed similar products (0 or 1).
10. Save_for_Later: Indicator whether the user has saved the product for later (0 or 1).
11. Personalized: Indicator of whether the user used personalized recommendations (0 or 1).
12. BUY: Indicator of whether the user purchased the product (0 or 1).

These metrics help to understand user behavior while browsing and purchasing on an e-commerce platform. 

# Predictive Model

We are going to implement the first stage of this entire Digital Marketing process, which would be the construction of a predictive model. 

## Imports
```python
from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import os
import matplotlib.pylab as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
```

These lines import the libraries needed for data analysis and modeling. The imported libraries are:

- `Pandas`: for data manipulation and analysis.
- `numpy`: for numerical operations.
- `OS`: For operations related to the operating system.
- `matplotlib.pylab`: for data visualization.
- `scikit-learn`: for machine learning modeling and evaluation.

## Function to load data

```python
def load_data(file_path):
    return pd.read_csv(file_path)
```

This function loads the data from a CSV file specified by the file path (`file_path`) and returns a pandas DataFrame.

## Function to inspect data

```python
def inspect_data(data):
     print(data.dtypes)
     print(data.head())
     print(data.describe())
     print(data.corr()['BUY'])
```

This function prints:

- The data types for each column (`data.dtypes`).
- The first five rows of the DataFrame (`data.head()`).
- A statistical summary of the data (`data.describe()`).
- The correlation of each column with the 'BUY' column (`data.corr()['BUY']`).

## Function to prepare data

```python
def prepare_data(data):
    # Correct the column name from 'Read_Reviews' to 'Read_Review'
    predictors = data[['Read_Review', 'Compare_Products', 'Add_to_List', 'Save_for_Later', 'Personalized', 'View_Similar']]
    targets = data.BUY
    return train_test_split(predictors, targets, test_size=0.3)
```

This function:

- Selects the predictor columns of interest (`Read_Review`, `Compare_Products`, `Add_to_List`, `Save_for_Later`, `Personalized`, `View_Similar`).
- Defines the target column as `BUY`.
- Splits the data into training and test sets using `train_test_split`, where 30% of the data is used for testing (`test_size=0.3`).

## Function to train model

```python
def train_model(X_train, y_train):
    model = GaussianNB()
    model.fit(X_train, y_train)
    return model
```

This function:

- Creates a Gaussian Naïve Bayes (`GaussianNB`) model.
- Trains the model with the training data (`X_train` and `y_train`).
- Returns the trained model.

## Function to evaluate model

```python
def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    print(confusion_matrix(y_test, predictions))
    print(accuracy_score(y_test, predictions))
    return model.predict_proba(X_test)
```

This function:

- Uses the model to make predictions on the test set (`X_test`).
- Prints the confusion matrix (`confusion_matrix`) and the accuracy of the model (`accuracy_score`).
- Returns the predictive probabilities for the test set (`model.predict_proba(X_test)`).

## Function to predict propensity

```python
def predict_propensity(model, data):
    data = np.array(data).reshape(1, -1)
    return model.predict_proba(data)[:,1]
```

This function:

- Converts the input data to a numpy array and adjusts the shape to be used in the model.
- Returns the probability of propensity predicted by the model.

## Execution Pipeline

```python
file_path = "/…/market_app_correlated.csv"
prospect_data = load_data(file_path)
inspect_data(prospect_data)
X_train, X_test, y_train, y_test = prepare_data(prospect_data)
model = train_model(X_train, y_train)
probabilities = evaluate_model(model, X_test, y_test)
```

These lines:

- Set the path of the CSV file.
- Load the data from the CSV.
- Inspect the uploaded data.
- Prepare the data for training and testing.
- Train the model with the training data.
- Evaluate the model with the test data.

## Simulations

### Predict propensity for new browsing data

```python
new_browsing_data = [0, 0, 0, 0, 0, 0]
print("New User: propensity:", predict_propensity(model, new_browsing_data))
```

**Result**:

```
New User: propensity: [0.19087601]
```

That is, simply by entering and logging in to the website or application, the chance of buying for that user is close to 19%.

### Predict propensity after adding to list

```python
add_to_list_data = [1, 1, 1, 0, 0, 0]
print("After Add_to_List: propensity:", predict_propensity(model, add_to_list_data))
```

**Result**:

```
After Add_to_List: propensity: [0.61234248]
```

When you add an item to your list, the chance of buying it goes up to 61%.

### Predict propensity after multiple interactions

```python
full_interaction_data = [1, 1, 1, 1, 1, 1]
print("Full Interaction: propensity:", predict_propensity(model, full_interaction_data))
```

**Result**:

```
Full Interaction: propensity: [0.80887743]
```

For those users who have made all possible interactions, the chance of purchase rises to about 81%.

## Conclusion

By leveraging data science and machine learning, this model significantly enhances the understanding of consumer actions and preferences. By systematically collecting and analyzing user interaction data, it becomes possible to predict purchasing behaviors with remarkable accuracy, enabling the creation of targeted marketing strategies that boost engagement and conversion rates. This case study underscores the potential of these techniques to drive substantial business growth and competitive advantage in the dynamic landscape of e-commerce.
```
