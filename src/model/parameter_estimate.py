import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RationalQuadratic, DotProduct, WhiteKernel, ExpSineSquared, Matern
from sklearn.preprocessing import StandardScaler
from skopt import BayesSearchCV
from skopt.space import Real, Categorical
import os
import warnings

warnings.filterwarnings(action='ignore')

os.environ['JOBLIB_TEMP_FOLDER'] = '/tmp'

data = pd.read_csv('flatten.csv', encoding='utf-8')
X = data[['value_1', 'value_2', 'value_3', 'value_4', 'value_5', 'value_6']]
y = data[['x', 'y']]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


def convert_kernel_string(params):
    kernel_type = params["kernel"]

    if kernel_type == "DotProduct":
        params["kernel"] = DotProduct(sigma_0=params.get("kernel__sigma_0", 1.0),
                                      sigma_0_bounds=params.get("kernel__sigma_0_bounds", (1e-5, 1e5)))

    elif kernel_type == "WhiteKernel":
        params["kernel"] = WhiteKernel(noise_level=params.get("kernel__noise_level", 1.0),
                                       noise_level_bounds=params.get("kernel__noise_level_bounds", (1e-5, 1e5)))

    elif kernel_type == "ExpSineSquared":
        params["kernel"] = ExpSineSquared(length_scale=params.get("kernel__length_scale", 1.0),
                                          periodicity=params.get("kernel__periodicity", 1.0),
                                          length_scale_bounds=params.get("kernel__length_scale_bounds", (1e-5, 1e5)),
                                          periodicity_bounds=params.get("kernel__periodicity_bounds", (1e-5, 1e5)))

    elif kernel_type == "Matern":
        params["kernel"] = Matern(length_scale=params.get("kernel__length_scale", 1.0),
                                  nu=params.get("kernel__nu", 1.5),
                                  length_scale_bounds=params.get("kernel__length_scale_bounds", (1e-5, 1e5)))

    elif kernel_type == "Matern+RationalQuadratic":
        matern = Matern(length_scale=params.get("matern__length_scale", 1.0),
                        nu=params.get("matern__nu", 1.5),
                        length_scale_bounds=params.get("matern__length_scale_bounds", (1e-5, 1e5)))

        rq = RationalQuadratic(length_scale=params.get("rq__length_scale", 1.0),
                               alpha=params.get("rq__alpha", 1.0),
                               length_scale_bounds=params.get("rq__length_scale_bounds", (1e-5, 1e5)),
                               alpha_bounds=params.get("rq__alpha_bounds", (1e-5, 1e5)))

        params["kernel"] = matern + rq

    return params


class CustomGPR(GaussianProcessRegressor):
    def set_params(self, **params):
        params = convert_kernel_string(params)
        super().set_params(**params)
        return self


gpr_x = CustomGPR(random_state=42)
gpr_y = CustomGPR(random_state=42)

kernel_search_spaces = {
    'ExpSineSquared': {
        'kernel': Categorical(['ExpSineSquared']),
        'kernel__length_scale': Real(1.0, 5.0, prior='uniform'),
        'kernel__periodicity': Real(0.5, 20.0, prior='uniform'),
        'alpha': Real(1e-10, 10, prior='log-uniform')
    },
    'Matern': {
        'kernel': Categorical(['Matern']),
        'kernel__length_scale': Real(1.0, 5.0, prior='uniform'),
        'kernel__nu': Real(0.5, 2.5, prior='uniform'),
        'alpha': Real(1e-10, 10, prior='log-uniform')
    },
    'Matern+RationalQuadratic': {
        'kernel': Categorical(['Matern+RationalQuadratic']),
        'matern__length_scale': Real(1.0, 5.0, prior='uniform'),
        'matern__nu': Real(0.5, 2.5, prior='uniform'),
        'matern__length_scale_bounds': (1e-5, 1e5),  # Add bounds if needed
        'rq__length_scale': Real(1.0, 5.0, prior='uniform'),
        'rq__alpha': Real(0.5, 2.5, prior='uniform'),
        'rq__length_scale_bounds': (1e-5, 1e5),  # Add bounds if needed
        'rq__alpha_bounds': (1e-5, 1e5),  # Add bounds if needed
        'alpha': Real(1e-10, 10, prior='log-uniform')
    }
}


class CustomVerboseCallback:
    def __init__(self, n_total, patience=15):
        self.n_total = n_total
        self.n_called = 0
        self.best_score = float('inf')
        self.not_improved_count = 0
        self.patience = patience

    def __call__(self, res):
        self.n_called += 1
        current_score = res.func_vals[-1]
        current_params = res.x_iters[-1]

        print(f"Iteration No: {self.n_called}/{self.n_total}")
        print(f"Parameters: {current_params}")
        print(f"Score: {current_score}")
        print("=" * 50)

        if current_score >= self.best_score:
            self.not_improved_count += 1
        else:
            self.best_score = current_score
            self.not_improved_count = 0

        if self.not_improved_count >= self.patience:
            return True


for kernel_name, search_space in kernel_search_spaces.items():
    custom_verbose_cb = CustomVerboseCallback(n_total=30, patience=15)

    print(f"\nOptimizing hyperparameters for {kernel_name} kernel and x coordinate...")
    bayes_search_x = BayesSearchCV(
        estimator=gpr_x,
        search_spaces=search_space,
        n_iter=40,
        cv=3,
        n_jobs=-1,
        scoring='neg_mean_squared_error',
        random_state=42
    )

    bayes_search_x.fit(X_scaled, y['x'], callback=[custom_verbose_cb])
    print("\nBest hyperparameters for x:", bayes_search_x.best_params_)

    print(f"\nOptimizing hyperparameters for {kernel_name} kernel and y coordinate...")
    bayes_search_y = BayesSearchCV(
        estimator=gpr_y,
        search_spaces=search_space,
        n_iter=40,
        cv=3,
        n_jobs=-1,
        scoring='neg_mean_squared_error',
        random_state=42
    )

    bayes_search_y.fit(X_scaled, y['y'], callback=[custom_verbose_cb])
    print("\nBest hyperparameters for y:", bayes_search_y.best_params_)