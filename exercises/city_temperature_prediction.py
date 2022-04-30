from IMLearn.learners.regressors import LinearRegression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
from datetime import datetime
import pandas as pd
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go

pio.templates.default = "simple_white"


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    # open the file path that was given
    data_frame = pd.read_csv(filename).dropna().drop_duplicates()

    for i in ["Month", "Day", "Year"]:
        data_frame = data_frame[data_frame[i] > 0]

    date_np = data_frame["Date"].to_numpy()
    data_frame = data_frame[data_frame["Day"].isin(range(1, 31))]
    data_frame = data_frame[data_frame["Month"].isin(range(1, 13))]
    data_frame = data_frame[data_frame["Temp"] > 0]

    len_date = len(date_np)
    day_of_year = np.zeros(len_date, )
    for k in range(len_date):
        t = datetime.strptime(date_np[k], "%Y-%m-%d")
        day_of_year[k] = t.timetuple().tm_yday
    s_DayOfYear = pd.Series(day_of_year)
    data_frame["dayOfyear"] = s_DayOfYear
    data_frame["dayOfyear"] = (data_frame["dayOfyear"]).astype(int)

    # data_frame = data_frame.drop("Year", 1)
    data_frame = data_frame.drop("Date", axis=1)
    # data_frame = pd.get_dummies(data_frame, prefix='decade_', columns=['decade'])
    # data_frame = pd.get_dummies(data_frame, prefix='Month_', columns=['Month'])
    # data_frame = pd.get_dummies(data_frame, prefix='Day_', columns=['Day'])
    return data_frame


if __name__ == '__main__':
    np.random.seed(0)

    # Question 1 - Load and preprocessing of city temperature dataset
    data_frame = load_data(r"/Users/hannahmordecai/Desktop/iml/IML.HUJI/datasets/City_Temperature.csv")
    # find samples from israel
    dt_israel = data_frame.loc[data_frame["Country"] == "Israel"].copy()
    dt_israel["Year"] = dt_israel["Year"].astype(str)

    # Question 2 - Exploring data for specific country
    figure = px.scatter(dt_israel, x="dayOfyear", y="Temp", color="Year",
                        title="temp according to day of year in israel", width=700, height=400,
                        template="simple_white")

    figure.update_traces(marker_size=3)
    # figure.write_image("/Users/hannahmordecai/Desktop/untitled folder/fig1.png")
    figure.show()

    month_dev = dt_israel.groupby("Month").agg('std').reset_index()
    month_dev["Month"] = month_dev["Month"].astype(str)
    figure_2 = px.histogram(month_dev, x="Month", y="Temp",
                            title="monthly temperture deviation",
                            width=700, height=400,
                            )
    # figure_2.write_image("/Users/hannahmordecai/Desktop/untitled folder/fig2.png")
    figure_2.show()

    # Question 3 - Exploring differences between countries
    cont = data_frame.drop("City", axis=1)
    cont = cont.groupby(["Month", "Country"]).agg({'Temp': ['std', 'mean']}).reset_index()
    cont.columns = ["Month", "Country", "Temp_std", "Temp_average"]
    fig3 = px.line(cont, x="Month", y="Temp_average", color="Country", error_y="Temp_std",
                   title="average monthly temperature of countries",
                   width=700, height=400,
                   template="simple_white"
                   )
    fig3.update_traces(marker_size=3)
    # fig3.write_image("/Users/hannahmordecai/Desktop/untitled folder/fig3.png")
    fig3.show()

    # Question 4 - Fitting model for different values of `k`
    train_x, train_y, test_x, test_y = split_train_test(dt_israel.dayOfyear, dt_israel.Temp, 0.75)

    k_arr = np.zeros(10,)
    loss_l = np.zeros(10,)
    for k in range(1, 11):
        k_arr[k-1] = k
        a = PolynomialFitting(k)
        x_np = train_x.to_numpy()
        y_np = train_y.to_numpy()
        a.fit(x_np, y_np)
        error_val = np.round(a.loss(test_x.to_numpy(), test_y.to_numpy()), 2)
        loss_l[k-1] = error_val
        print("the k value is ", k, " and the loss value is ", error_val)
    data_cols = {'k_degree': k_arr, 'loss_err': loss_l}
    data_f = pd.DataFrame(data=data_cols)
    data_f['k_degree'] = data_f['k_degree'].astype(int)

    fig4 = px.histogram(data_f, x="k_degree", y ="loss_err", histnorm ="",nbins=10,
                     title="test error recorded for values of k",
                     width=700, height=400,
                        )
    # fig4.write_image("/Users/hannahmordecai/Desktop/untitled folder/fig4.png")
    fig4.show()

    # Question 5 - Evaluating fitted model on different countries
    b = PolynomialFitting(5)
    loss_lc = np.array([])
    b.fit(dt_israel.dayOfyear, dt_israel.Temp)
    all_cont = data_frame.Country.unique()
    all_cont = all_cont[all_cont != "Israel"]
    for c in all_cont:
        cd = data_frame[data_frame.Country == c]
        loss_error = np.round(b.loss(cd.dayOfyear, cd.Temp),2)
        loss_lc = np.append(loss_lc, loss_error)
    fig5 = px.bar(x=all_cont, y=loss_lc, title=f"model error of other countrys ",
                  labels={"x": "Country", "y": "Error"})
    # fig5.write_image("/Users/hannahmordecai/Desktop/untitled folder/fig5.png")
    fig5.show()

