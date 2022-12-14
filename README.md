### Prediction-with-Optimization-Models
*OptimizationModels.ipynb* \
Write a generic function which will be able to fit a Gaussian process regression model to each of provided data sets via maximum likelihood estimation. \
Find the conditional predictive distribution of $y_p$ given $y$, $x$, $x_p$, and $\theta = (\sigma^2_n, \sigma^2_s, l)$. Given everything is a multivariate normal distribution, this conditional distribution is
```math
y_p | y, \theta \sim \text{MVN}(\mu^\star, \Sigma^\star)
```
where
```math
\begin{align*}
\mu^\star &= \mu_p + \Sigma(x_p, x) \, \Sigma(x)^{-1} \, (y - \mu) \\
\Sigma^\star &= \Sigma(x_p) - \Sigma(x_p, x) \, \Sigma(x)^{-1} \, \Sigma(x, x_p)
\end{align*}
```
In these formulae, $\Sigma(x_p)$ is the $n_p \times n_p$ covariance matrix constructed from the $n_p$ prediction locations and $\Sigma(x_p, x)$ is the $n_p \times n$ cross covariance matrix constructed from the $n_p$ prediction locations and the $n$ data locations. Note that $\Sigma(x_p, x)^T = \Sigma(x, x_p)$. As mentioned in the preceding task - we will assume that $\mu$ and $\mu_p$ are 0.
Implement a python function which calculates the mean and covariance of conditional distribution described above and meets the requirements. \
Take the result from the `predict()` function and generate a plot showing the mean predicted `y` (across prediction samples) as well as a shaded region showing a 95% confidence interval (empirically determined from the prediction samples)

*NewtonOptimization_torch.ipynb* \


