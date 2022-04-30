from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    # 1000 samples to be drawn
    sam = np.random.normal(10, 1, 1000)
    uni_g = UnivariateGaussian()
    uni_g.fit(sam)
    # print the samples in format: (mu,var)
    print("(" + str(uni_g.mu_) + ", " + str(uni_g.var_) + ")\n")

    # Question 2 - Empirically showing sample mean is consistent
    sam_arr = np.array([0] * 1000, dtype=np.int64)
    # sam_arr = np.zeros(1000).astype(np.int64)
    abs_distance = []
    sam_arr[0] = 10
    for i in range(1, 100):
        sam_arr[i] = sam_arr[i - 1] + 10
        a = sam[0:sam_arr[i]]
        abs_distance.append(abs(np.mean(a) - 10))
    go.Figure([go.Scatter(x=sam_arr, y=abs_distance,
                          mode='markers+lines',
                          name=r'$\widehat\mu$')],
              layout=go.Layout(title=r"$\text{ the distance"
                                     r" between the estimated "
                                     r"value and the true expectation}$",
                               xaxis_title="$m\\text{ - the number of samples}$",
                               yaxis_title="$\\text{ |estimated expectation-true expectation| }$",
                               height=300)).show()

    # Question 3 - Plotting Empirical PDF of fitted model
    pdf_sam = uni_g.pdf(sam)
    go.Figure([go.Scatter(x=sam, y=pdf_sam, mode='markers',
                          marker=dict(color="black"), showlegend=False)],
              layout=go.Layout(title=r"$\text{ Empirical pdf}$", xaxis_title="$\\text{ value}$",
                               yaxis_title="$\\text{ pdf }$", height=300)).show()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    val_ = np.array([[1, 0.2, 0, 0.5], [0.2, 2, 0, 0], [0, 0, 1, 0], [0.5, 0, 0, 1]])
    mu_ = np.array([0, 0, 4, 0])
    sam_multi = MultivariateGaussian()
    sam_val = np.random.multivariate_normal(mu_, val_, 1000)
    sam_multi.fit(sam_val)
    print(str(sam_multi.mu_) + "\n" + str(sam_multi.cov_) + "\n")

    # Question 5 - Likelihood evaluation
    ll = np.zeros((200, 200))
    ll_single = np.zeros(200)
    first_f = np.linspace(-10, 10, 200)
    for outer in range(200):
        for inner in range(200):
            ll_single[inner] = sam_multi. \
                log_likelihood(np.array([first_f[outer], 0,
                                         first_f[inner], 0]), val_, sam_val)
        ll[outer] = ll_single

    figure = px.imshow(ll, labels=dict(x="value of f1 ", y=" value of f3",
                                       color="log likelihood"), x=first_f, y=first_f)
    figure.update_xaxes(side="top")
    figure.show()

    # Question 6 - Maximum likelihood
    temp = np.unravel_index(ll.argmax(), ll.shape)
    print("(f1,f3) = (" + str(first_f[temp[0]]) + "," + str(first_f[temp[1]]) + ")")


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
