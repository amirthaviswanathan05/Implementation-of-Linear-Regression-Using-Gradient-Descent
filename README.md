# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import necessary libraries for numerical operations, data handling, and preprocessing.

2.Load the startup dataset (50_Startups.csv) using pandas.

3.Extract feature matrix X and target vector y from the dataset.

4.Convert feature and target values to float and reshape if necessary.

5.Standardize X and y using StandardScaler.

6.Add a column of ones to X to account for the bias (intercept) term.

7.Initialize model parameters (theta) to zeros.

8.Perform gradient descent to update theta by computing predictions and adjusting for error.

9.Input a new data point, scale it, and add the intercept term.

10.Predict the output using learned theta, then inverse-transform it to get the final result. 

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: AMIRTHAVARSHINI V
RegisterNumber: 212223040014
*/
```
```
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(X1, y, learning_rate=0.01, num_iters=1000):
    # Add a column of ones to X for the intercept term
  X = np.c_[np.ones(len(X1)), X1]
  # Initialize theta with zeros
  theta = np.zeros(X.shape[1]).reshape(-1,1)
  # Perform gradient descent
  for _ in range(num_iters):
    # Calculate predictions
    predictions = (X).dot(theta).reshape(-1, 1)
    # Calculate errors
    errors = (predictions - y).reshape(-1,1)
    # Update theta using gradient descent
    theta -= learning_rate * (1 / len(X1)) * X.T.dot(errors)
  return theta
data = pd.read_csv('50_Startups.csv',header=None)
print(data.head())

# Assuming the last column is your target variable 'y' and the preceding columns are your features 'X'
X = (data.iloc[1:, :-2].values)
print(X)
X1=X.astype(float)
scaler = StandardScaler()
y = (data.iloc[1:,-1].values).reshape(-1,1)
print(y)
X1_Scaled = scaler.fit_transform(X1)
Y1_Scaled = scaler.fit_transform(y)
print(X1_Scaled)
print(Y1_Scaled)

# Learn model parameters
theta = linear_regression(X1_Scaled, Y1_Scaled)

# Predict target value for a new data point
new_data = np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled = scaler.fit_transform(new_data)
prediction =np.dot(np.append(1, new_Scaled), theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(f"Predicted value: {pre}")
```
```
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def linear_regression(X, y, learning_rate=0.01, num_iters=1000):
    # Initialize weights (w) with zeros, and initialize bias (b) as zero
    w = np.zeros(X.shape[1]).reshape(-1, 1)
    b = 0

    # Perform gradient descent
    for _ in range(num_iters):
        # Calculate predictions: f = wx + b
        predictions = X.dot(w) + b

        # Calculate errors
        errors = predictions - y

        # Update weights (w) and bias (b) using gradient descent
        w -= learning_rate * (1 / len(X)) * X.T.dot(errors)
        b -= learning_rate * (1 / len(X)) * np.sum(errors)

    return w, b

# Load the dataset
data = pd.read_csv('50_Startups.csv', header=None)

# Assuming the last column is your target variable 'y' and the preceding columns are your features 'X'
X = data.iloc[1:, :-2].values.astype(float)
y = data.iloc[1:, -1].values.reshape(-1, 1).astype(float)

# Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Learn model parameters
w, b = linear_regression(X_scaled, y)

# Predict target value for a new data point (example values)
new_data = np.array([165349.2, 136897.8, 471784.1]).reshape(1, -1)
new_data_scaled = scaler.transform(new_data)

# Predict using the learned model
prediction = new_data_scaled.dot(w) + b

print(f"Predicted value: {prediction[0][0]}")
```
## Output:
![linear regression using gradient descent](sam.png)
![446831385-2416651a-01a6-4128-9ed4-da1494b993a7](https://github.com/user-attachments/assets/f78911da-d043-434e-860c-72da3a7a9c38)
![446831401-2bcffe09-2384-434e-b1e3-996a92dd0191](https://github.com/user-attachments/assets/97a05366-82c3-49b8-8b63-2f1ce74de030)
![446831418-f5f5e27d-8673-46ef-81b6-89266f177811](https://github.com/user-attachments/assets/72cc0af0-6464-45eb-b6c0-506eca363565)
![446831464-fa5e5f8c-0dd0-48ba-9a6a-b9fbf21e08eb](https://github.com/user-attachments/assets/70ca66e5-410c-472c-ade8-8acc86eb1d88)
![446831479-294a2851-75a8-4f64-8888-3d93867751bb](https://github.com/user-attachments/assets/d0f67bac-f6df-435f-bd59-e1d180f606fd)


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
