# %%
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression


# load the data
X, y = load_diabetes(return_X_y=True)
print(X.shape, y.shape)

# division into training and testing
np.random.seed(42)
order = np.random.permutation(len(y))
tst = np.sort(order[:221])
tr = np.sort(order[221:])

Xtr = X[tr, :]
Xtst = X[tst, :]
Ytr = y[tr]
Ytst = y[tst]

reg = LinearRegression(fit_intercept=False).fit(Xtr, Ytr)

# %%

Ypred = reg.predict(Xtst)

# The coefficients
print("Coefficients: \n", reg.coef_)

# The mean squared error
print("Mean squared error: %.2f" % mean_squared_error(Ytst, Ypred))

# %%
# The Root mean squared error
print("Root mean squared error: %.2f" % np.sqrt(mean_squared_error(Ytst, Ypred)))
