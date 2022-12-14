### Prediction-with-Optimization-Models
### *OptimizationModels.ipynb* 
#### Task1
Write a generic function which will be able to fit a Gaussian process regression model to each of provided data sets via maximum likelihood estimation. \
For a Gaussian process regression we will assume a model with the following form,
```math
y \sim \text{MVN}(\mu, \Sigma(x))
```
which implies that our observed values of $y$ are derived from a multivariate normal distribution with a specific mean and covariance structure. For the sake of simplicity, we will assume that $\mu$ will always be 0 (a vector of zeros in this case) for these models. The covariance ($\Sigma$) will be given by a Gaussian / squared exponential kernel that is a function of the $x$ values, specifically their distances from one another.

Explicitly, the elements of the covariance matrix are constructed using the following formula,
```math
\Sigma_{i,j}(x) = \text{cov}(x_i, x_j) = \sigma^2_n \, \mathcal{I}_{i = j} + \sigma^2_s \exp{\left(- \frac{(x_i-x_j)^2}{2l}\right)} 
```
where $\sigma^2_n$, $\sigma^2_s$ and $l$ are the models parameters (what we will be estimating via MLE from the data).

* $\sigma^2_n$ - this is the nugget variance parameter and represents irreducible measurement error. Note: $\mathcal{I}_{i = j}$  is an indicator function which is 1 when $i=j$ and 0 otherwise, meaning the nugget variance is only added to the diagonal of covariance matrix.
* $\sigma^2_s$ - this is the scale variance parameter and determines the average distance away from the mean that can be taken by the function.
* $l$ - this is the length-scale parameter which determines the range of the "spatial" dependence between points. Larger values of $l$ result in *less* wiggly functions (greater spatial dependence) and smaller values result in *more* wiggly functions (lesser spatial dependence) - values are relative to the scale of $x$.

In order to fit the model the goal is to determine the optimal values of these three parameters given the data. We will be accomplishing this via maximum likelihood. Given our multivariate normal model we can then take the MVN density,
```math
f(y) = \frac{1}{\sqrt{\det(2\pi\Sigma(x))}} \exp \left[-\frac{1}{2} (y-\mu)^T \Sigma(x)^{-1} (y-\mu) \right]
```
we can derive the log likelihood as 
```math
\ln L(y) = -\frac{1}{2} \left[n \ln (2\pi) + \ln (\det \Sigma(x)) + (y-\mu)^T \Sigma(x)^{-1} (y-\mu) \right].
```
The goal therefore is to find,
```math
\underset{\sigma^2_n, \sigma^2_s, l}{\text{argmax}} \, L(y) \quad \text{or} \quad \underset{\sigma^2_n, \sigma^2_s, l}{\text{argmin}} \, -L(y).
```
#### Task2
Once the model parameters have been obtained the goal will be to predict (i.e. draw samples from) our Gaussian process model for new values of $x$. Specifically, we want to provide an fine, equally space grid of $x$ values from which we will predict the value of the function. Multiple independent predictions (draws) can then be average to get an overall estimate of the underlying smooth function for each data set.

Therefore the goal is to find the conditional predictive distribution of $y_p$ given $y$, $x$, $x_p$, and $\theta = (\sigma^2_n, \sigma^2_s, l)$. Given everything is a multivariate normal distribution, this conditional distribution is
$$ y_p | y, \theta \sim \text{MVN}(\mu^\star, \Sigma^\star)$$  
where
```math
\begin{align*}
\mu^\star &= \mu_p + \Sigma(x_p, x) \, \Sigma(x)^{-1} \, (y - \mu) \\
\Sigma^\star &= \Sigma(x_p) - \Sigma(x_p, x) \, \Sigma(x)^{-1} \, \Sigma(x, x_p)
\end{align*}
```
In these formulae, $\Sigma(x_p)$ is the $n_p \times n_p$ covariance matrix constructed from the $n_p$ prediction locations and $\Sigma(x_p, x)$ is the $n_p \times n$ cross covariance matrix constructed from the $n_p$ prediction locations and the $n$ data locations. Note that $\Sigma(x_p, x)^T = \Sigma(x, x_p)$. As mentioned in the preceding task - we will assume that $\mu$ and $\mu_p$ are 0. \
#### Task3
Take the result from the `predict()` function and generate a plot showing the mean predicted `y` (across prediction samples) as well as a shaded region showing a 95% confidence interval (empirically determined from the prediction samples)

Optionally, the user should be able to provide the original data set `d` which would then be overlayed as a scatter plot.

*NewtonOptimization_torch.ipynb* \


