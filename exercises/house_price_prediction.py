import math
from os import path

from IMLearn.metrics import loss_functions
from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from numpy import transpose, linalg

pio.templates.default = "simple_white"


def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    # open the file path that was given
    open_file = pd.read_csv(filename).dropna().drop_duplicates()

    feat = ["id", "lat", "long", "date"]
    for f in feat:
        open_file = open_file.drop(f, axis=1)

    # make sure these features receive values over 0
    pos_feat = ["price", "sqft_living", "sqft_above", "yr_built", "sqft_living15", "sqft_lot15"]
    for p in pos_feat:
        open_file = open_file[open_file[p] > 0]

    # make sure these features receive non-negative values
    non_neg = ["bathrooms", "floors", "sqft_basement", "yr_renovated"]
    for n in non_neg:
        open_file = open_file[open_file[n] >= 0]

    # update conditions on these features
    open_file = open_file[open_file["waterfront"].isin([0, 1])]
    open_file = open_file[open_file["view"].isin(range(5))]
    open_file = open_file[open_file["condition"].isin(range(1, 6))]
    open_file = open_file[open_file["grade"].isin(range(1, 15))]
    open_file["decade_built"] = np.where(
        open_file["yr_renovated"] >= np.percentile(open_file.yr_renovated.unique(), 70),
        1, 0)
    open_file = open_file[open_file["bedrooms"] < 20]
    open_file = open_file[open_file["sqft_lot"] < 1250000]
    open_file = open_file[open_file["sqft_lot15"] < 500000]
    open_file = open_file[open_file["sqft_lot15"] < 500000]
    # remove this column from the table.
    open_file = open_file.drop("yr_renovated", axis=1)
    # open_file["yr_built"] = open_file["yr_built"].astype(int)

    open_file = open_file.drop("yr_built", axis=1)
    # open_file = pd.get_dummies(open_file, prefix='zipcode_', columns=['zipcode'])
    # open_file = pd.get_dummies(open_file, prefix='decade_built_', columns=['decade_built'])

    open_file.insert(0, 'intercept', 1, True)
    return open_file.drop("price", 1), open_file.price


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """

    for i in X:
        # X = X.loc[0: ~(X.columns.str.contains('^zipcode_', case=False) | X.columns.str.contains
        # ('^decade_built_', case=False))].drop("intercept", 1)

        for f in X:
            var_mul = np.std(X[f]) * np.std(y)
            calc = np.cov(X[f], y)[0, 1] / var_mul
            figure = px.scatter(pd.DataFrame({'x': X[f], 'y': y}),
                                x="x", y="y", trendline="ols", title=f"corrolation between {f} values and response <br>pearson corrolation {calc}",
                                labels={"x": f"{f} values", "y": "response valiue"})

            pio.write_image(figure, path.join(output_path, "pearson correlation of %s.png" % f))


if __name__ == '__main__':
    np.random.seed(0)

    # Question 1 - Load and preprocessing of housing prices dataset
    x, y = load_data("/Users/hannahmordecai/Desktop/iml/IML.HUJI/datasets/house_prices.csv")

    # Question 2 - Feature evaluation with respect to response
    feature_evaluation(x, y, "/Users/hannahmordecai/Desktop/untitled folder")

    # Question 3 - Split samples into training- and testing sets.
    split_tuple = split_train_test(x, y, 0.75)

    train_x = split_tuple[0]
    train_y = split_tuple[1]
    test_x = split_tuple[2].to_numpy()
    test_y = split_tuple[3].to_numpy()

    avg_loss = np.zeros(91, )
    p_size = np.zeros(91, )
    results = np.zeros(91, )
    loss_arr = np.zeros(10, )
    for p in range(10, 101):
        avg = 0
        for i in range(10):
            a = LinearRegression()
            sam = split_train_test(train_x, train_y, (p / 100))
            sam_p1 = sam[0].to_numpy()
            sam_p2 = sam[1].to_numpy()
            a.fit(sam_p1, sam_p2)
            mse = a.loss(test_x, test_y)
            loss_arr[i] = mse
            avg = avg + mse
        avg = avg / 10
        avg_loss[p - 10] = avg
        results[p - 10] = math.sqrt(np.var(loss_arr)) * 2
        p_size[p -10] = p

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)

    fig = go.Figure([
        go.Scatter(
            name='average loss',
            x=p_size,
            y=avg_loss,
            mode='lines',
            line=dict(color='rgb(31, 119, 180)'),
        ),
        go.Scatter(
            name='Upper Bound',
            x=p_size,
            y=avg_loss + results,
            mode='lines',
            marker=dict(color="#444"),
            line=dict(width=0),
            showlegend=False
        ),

        go.Scatter(
            name='Lower Bound',
            x=p_size,
            y=avg_loss - results,
            marker=dict(color="#444"),
            line=dict(width=0),
            mode='lines',
            fillcolor='rgba(68, 68, 68, 0.3)',
            fill='tonexty',
            showlegend=False
        )
    ])
    fig.update_layout(
        xaxis_title='%p',
        yaxis_title='average loss',
        title='mean loss as function of %p with confidence interval',
        hovermode="x"
    )
    fig.write_image("/Users/hannahmordecai/Desktop/untitled folder/fig0.png")

    fig.show()
