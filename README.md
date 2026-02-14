# Implementation of Multivariate Linear Regression
## Aim
To write a python program to implement multivariate linear regression and predict the output.
## Equipment’s required:
1.	Hardware – PCs
2.	Anaconda – Python 3.7 Installation / Moodle-Code Runner
## Algorithm:
```
Step1
import pandas as pd.

Step2
Read the csv file.

Step3
Get the value of X and y variables

Step4
Create the linear regression model and fit.

Step5
Predict the CO2 emission of a car where the weight is 2300kg, and the volume is 1300cm cube.
```
## Program:
```
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split

# -----------------------------
# Load the Boston dataset properly
# -----------------------------
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep=r"\s+", skiprows=22, header=None)

# Correct reshaping (VERY IMPORTANT)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]

# Remove any NaN values (safety check)
mask = ~(np.isnan(data).any(axis=1) | np.isnan(target))
X = data[mask]
y = target[mask]

# -----------------------------
# Split into train and test sets
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=1
)

# -----------------------------
# Create and train Linear Regression model
# -----------------------------
reg = linear_model.LinearRegression()
reg.fit(X_train, y_train)

# -----------------------------
# Print Results
# -----------------------------
print("Coefficients:", reg.coef_)
print("Variance score:", reg.score(X_test, y_test))

# -----------------------------
# Plot Residual Errors
# -----------------------------
plt.style.use("fivethirtyeight")

# Training residuals
plt.scatter(
    reg.predict(X_train),
    reg.predict(X_train) - y_train,
    color="green",
    s=10,
    label="Train data",
)

# Testing residuals
plt.scatter(
    reg.predict(X_test),
    reg.predict(X_test) - y_test,
    color="blue",
    s=10,
    label="Test data",
)

# Zero residual line
plt.hlines(y=0, xmin=0, xmax=50, linewidth=2)

plt.legend(loc="upper right")
plt.title("Residual errors")
plt.show()






```
## Output:
<img width="675" height="85" alt="Screenshot 2026-02-14 085852" src="https://github.com/user-attachments/assets/5fd4a813-0f47-4fb1-856c-07873006c719" />


### Insert your output
<img width="640" height="480" alt="Figure_1" src="https://github.com/user-attachments/assets/19d04a7a-9932-4d97-ab62-224c6a0e5fcd" />



## Result
Thus the multivariate linear regression is implemented and predicted the output using python program.
