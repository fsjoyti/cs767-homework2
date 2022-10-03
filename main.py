import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import UnivariateSpline, CubicSpline
import operator
import time
from sklearn.linear_model import Lasso, Ridge, LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from statistics import mean


def create_2d_dataset(n):
    means = np.zeros((3, 2))
    means[0] = np.array([1.2, 0.4])
    means[1] = np.array([-4.4, 1.0])
    means[2] = np.array([4.1, -0.3])

    covariances = np.zeros((3, 2, 2))
    covariances[0] = np.array([[0.8, -0.4], [-0.4, 1.0]])
    covariances[1] = np.array([[1.2, -0.8], [-0.8, 1.0]])
    covariances[2] = np.array([[1.2, 0.6], [0.6, 3.0]])
    x = []
    y = []
    for k in range(3):
        x_tmp, y_tmp = np.random.multivariate_normal(means[k], covariances[k], n).T
        x = np.hstack([x, x_tmp])
        y = np.hstack([y, y_tmp])

    data = np.vstack([x, y])
    data = data.T
    plot_2d_points(data)
    return data


def plot_2d_points(points):
    figure = plt.figure()
    x, y = points[:, 0], points[:, 1]
    plt.plot(x, y, '.', alpha=0.5)
    plt.axis('equal')
    plt.show()


def piecewise_linear_regression(points):
    n, points = sort_points(points)
    x, y = points[:, 0], points[:, 1]
    start = time.process_time()
    spl = UnivariateSpline(x, y, k=1, s=0.5)
    x_new = np.linspace(x.min(), x.max(), n)
    y_new = spl(x_new)
    plt.figure(figsize=(10, 8))
    plt.plot(x_new, y_new, '-')
    plt.plot(x, y, 'o')
    plt.title('Piecewise linear regression')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
    return time.process_time() - start


def piecewise_polynomial_regression(points):
    n, points = sort_points(points)
    x, y = points[:, 0], points[:, 1]
    start = time.process_time()
    f = CubicSpline(x, y, bc_type='natural')
    x_new = np.linspace(x.min(), x.max(), n)
    y_new = f(x_new)
    plt.figure(figsize=(10, 8))
    plt.plot(x_new, y_new, '-')
    plt.plot(x, y, 'o')
    plt.title('Piecewise polynomial regression')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
    return time.process_time() - start


def polynomial_regression(points, degree):
    x, y = reshape_points(points)
    start = time.process_time()
    polynomial_features = PolynomialFeatures(degree=degree)
    x_poly = polynomial_features.fit_transform(x)
    model = LinearRegression()
    model.fit(x_poly, y)
    y_poly_pred = model.predict(x_poly)
    plt.scatter(x, y, s=10)
    # sort the values of x before line plot
    sort_axis = operator.itemgetter(0)
    sorted_zip = sorted(zip(x, y_poly_pred), key=sort_axis)
    x, y_poly_pred = zip(*sorted_zip)
    plt.title('Polynomial Regression')
    plt.plot(x, y_poly_pred, color='m')
    plt.show()
    return time.process_time() - start


def ridge_regression(points, degree):
    x, y = reshape_points(points)
    start = time.process_time()
    polynomial_features = PolynomialFeatures(degree=degree)
    x_poly = polynomial_features.fit_transform(x)
    ridge = Ridge(alpha=0.5)
    ridge.fit(x_poly, y)
    y_poly_pred = ridge.predict(x_poly)
    rmse = np.sqrt(mean_squared_error(y, y_poly_pred))
    r2 = r2_score(y, y_poly_pred)
    accuracy = ridge.score(x_poly, y)
    plt.scatter(x, y, s=10)
    sort_axis = operator.itemgetter(0)
    sorted_zip = sorted(zip(x, y_poly_pred), key=sort_axis)
    x, y_poly_pred = zip(*sorted_zip)
    plt.title('Ridge Regression')
    plt.plot(x, y_poly_pred, color='m')
    plt.show()
    return time.process_time() - start


def lasso_regression(points, degree):
    x, y = reshape_points(points)
    start = time.process_time()
    polynomial_features = PolynomialFeatures(degree=degree)
    x_poly = polynomial_features.fit_transform(x)
    lasso = Lasso(alpha=0.7)
    lasso.fit(x_poly, y)
    y_poly_pred = lasso.predict(x_poly)
    plt.scatter(x, y, s=10)
    sort_axis = operator.itemgetter(0)
    sorted_zip = sorted(zip(x, y_poly_pred), key=sort_axis)
    x, y_poly_pred = zip(*sorted_zip)
    plt.title('Lasso Regression')
    plt.plot(x, y_poly_pred, color='m')
    plt.show()
    return time.process_time() - start


def reshape_points(points):
    x, y = points[:, 0], points[:, 1]
    x = x[:, np.newaxis]
    y = y[:, np.newaxis]
    return x, y


def sort_points(points):
    n = points.shape[0]
    ind = np.lexsort((points[:, 1], points[:, 0]))
    points = points[ind]
    return n, points


if __name__ == '__main__':

    print()
    piecewise_linear_execution_times = []
    piecewise_polynomial_execution_times = []
    polynomial_regression_execution_times = []
    ridge_regression_execution_times = []
    lasso_regression_execution_times = []

    polynomial_regression_degree_twelve_execution_times = []
    ridge_regression_execution_degree_twelve_times = []
    lasso_regression_execution_degree_twelve_execution_times = []

    polynomial_regression_degree_six_execution_times = []
    ridge_regression_execution_degree_six_times = []
    lasso_regression_execution_degree_six_execution_times = []

    polynomial_regression_degree_twenty_execution_times = []
    ridge_regression_execution_degree_twenty_times = []
    lasso_regression_execution_degree_twenty_execution_times = []

    for i in range(10):
        dataset = create_2d_dataset(10)
        piecewise_linear_execution_times.append(piecewise_linear_regression(dataset))
        piecewise_polynomial_execution_times.append(piecewise_polynomial_regression(dataset))
        polynomial_regression_execution_times.append(polynomial_regression(dataset, 3))
        ridge_regression_execution_times.append(ridge_regression(dataset, 3))
        lasso_regression_execution_times.append(lasso_regression(dataset, 3))

        polynomial_regression_degree_six_execution_times.append(polynomial_regression(dataset, 6))
        ridge_regression_execution_degree_six_times.append(ridge_regression(dataset, 6))
        lasso_regression_execution_degree_six_execution_times.append(lasso_regression(dataset, 6))

        polynomial_regression_degree_twelve_execution_times.append(polynomial_regression(dataset, 12))
        ridge_regression_execution_degree_twelve_times.append(ridge_regression(dataset, 12))
        lasso_regression_execution_degree_twelve_execution_times.append(lasso_regression(dataset, 12))

        polynomial_regression_degree_twenty_execution_times.append(polynomial_regression(dataset, 20))
        ridge_regression_execution_degree_twenty_times.append(ridge_regression(dataset, 20))
        lasso_regression_execution_degree_twenty_execution_times.append(lasso_regression(dataset, 20))

    print()
    print('Piecewise linear regression times: ', piecewise_linear_execution_times)
    print('Average piecewise linear regression time ', mean(piecewise_linear_execution_times))
    print()
    print('Piecewise polynomial execution times: ', piecewise_polynomial_execution_times)
    print('Average piecewise polynomial regression time ', mean(piecewise_polynomial_execution_times))
    print()
    print('Polynomial regression degree 3 execution times:  ', polynomial_regression_execution_times)
    print('Average polynomial regression degree 3 time ', mean(polynomial_regression_execution_times))
    print()

    print('Ridge regression degree 3 execution times:  ', ridge_regression_execution_times)
    print('Average ridge regression degree 3 time ', mean(ridge_regression_execution_times))
    print()
    print('Lasso regression degree 3 execution times: ', lasso_regression_execution_times)
    print('Average lasso regression degree 3 time ', mean(lasso_regression_execution_times))
    print()

    print('Polynomial regression degree 8 execution times: ', polynomial_regression_degree_six_execution_times)
    print('Average polynomial regression degree 8 time ', mean(polynomial_regression_degree_six_execution_times))
    print()
    print('Ridge regression degree 8 execution times: ', ridge_regression_execution_degree_six_times)
    print('Average ridge regression degree 8 time ', mean(ridge_regression_execution_degree_six_times))
    print()
    print('Lasso regression degree 8 execution times: ', lasso_regression_execution_degree_six_execution_times)
    print('Average lasso regression degree 8 times ',
          mean(lasso_regression_execution_degree_six_execution_times))
    print()

    print('Polynomial regression degree 12 execution times: ', polynomial_regression_degree_twelve_execution_times)
    print('Average polynomial regression degree 4 time ', mean(polynomial_regression_degree_twelve_execution_times))
    print()
    print('Ridge regression degree 12 execution times:  ', ridge_regression_execution_degree_twelve_times)
    print('Average ridge regression degree 4 degree 4 time ', mean(ridge_regression_execution_degree_twelve_times))
    print()
    print('Lasso regression degree 12 execution times: ', lasso_regression_execution_degree_twelve_execution_times)
    print('Average lasso regression degree 12 degree 12 time ',
          mean(lasso_regression_execution_degree_twelve_execution_times))
    print()

    print('Polynomial regression degree 20 execution times: ', polynomial_regression_degree_twenty_execution_times)
    print('Average polynomial regression degree 20 times ', mean(polynomial_regression_degree_twenty_execution_times))
    print()
    print('Ridge regression degree 20 execution times: ', ridge_regression_execution_degree_twenty_times)
    print('Average ridge regression degree 20 time ', mean(ridge_regression_execution_degree_twenty_times))
    print()
    print('Lasso regression degree 20 execution times: ', lasso_regression_execution_degree_twenty_execution_times)
    print('Average lasso regression degree 20 times', mean(lasso_regression_execution_degree_twenty_execution_times))
    print()
