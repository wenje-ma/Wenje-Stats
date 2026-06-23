# ch2_gp_visualization.R
# Gaussian Process prior, data, and posterior visualization
# Usage: Rscript ch2_gp_visualization.R <in_dir> <out_dir>

args <- commandArgs(trailingOnly = TRUE)
in_dir <- args[1]
out_dir <- args[2]

library(rkriging)
library(MASS)
library(pdist)
library(Matrix)

# Read data
data <- read.csv(file.path(in_dir, "data.csv"))
params <- read.csv(file.path(in_dir, "params.csv"))

theta <- params$theta[1]
n_samples <- params$n_samples[1]
seed <- params$seed[1]

set.seed(seed)

D <- data$x[!is.na(data$x)]
y <- data$y[!is.na(data$y)]
test <- data$test[!is.na(data$test)]

n <- length(D)
N <- length(test)

# Create kernel with fixed theta
kernel <- Get.Kernel(theta / sqrt(2), type = "Gaussian")

# Fit kriging model
a <- Fit.Kriging(D, y, fit = FALSE, kernel = kernel)
tau2 <- Get.Kriging.Parameters(a)$nu2
pred <- Predict.Kriging(a, test)

# Prior distribution: samples from GP prior
S0 <- Evaluate.Kernel(kernel, test)
u_prior <- mvrnorm(n = n_samples, mu = rep(0, N), Sigma = tau2 * S0)

# Posterior distribution: samples from GP posterior
R <- Evaluate.Kernel(kernel, D)
L <- chol(R + 10^(-10) * diag(n))
M <- t(as.matrix(pdist(cbind(D), cbind(test))))
M <- exp(-M^2 / theta^2)
b <- forwardsolve(t(L), t(M))
S <- tau2 * (S0 - t(b) %*% b)
S <- nearPD(S)$mat
me <- pred$mean
u_post <- mvrnorm(n = n_samples, mu = me, Sigma = as.matrix(S))

# Save prior samples
write.csv(
  data.frame(t(u_prior)),
  file.path(out_dir, "prior_samples.csv"),
  row.names = FALSE
)

# Save posterior samples
write.csv(
  data.frame(t(u_post)),
  file.path(out_dir, "posterior_samples.csv"),
  row.names = FALSE
)

# Save prediction (mean and sd)
write.csv(
  data.frame(test = test, mean = pred$mean, sd = pred$sd),
  file.path(out_dir, "prediction.csv"),
  row.names = FALSE
)
