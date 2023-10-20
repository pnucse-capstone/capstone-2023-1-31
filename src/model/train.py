import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RationalQuadratic
from joblib import dump

data = pd.read_csv('window_data.csv')

X = data[['value_1', 'value_2', 'value_3', 'value_4', 'value_5', 'value_6']]
y = data[['x', 'y']]

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)

X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

kernel_x = RationalQuadratic(length_scale=6.877, alpha=2.27)
kernel_y = RationalQuadratic(length_scale=2.43609, alpha=11)

gpr_x = GaussianProcessRegressor(kernel=kernel_x, random_state=42)
gpr_x.fit(X_train, y_train['x'])
gpr_y = GaussianProcessRegressor(kernel=kernel_y, random_state=42)
gpr_y.fit(X_train, y_train['y'])

dump(gpr_x, 'windowX.joblib')
dump(gpr_y, 'windowY.joblib')

