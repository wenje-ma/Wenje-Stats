# ch9_kriging_functional.R
# Functional output Kriging and optimal design for model calibration
# Usage: Rscript ch9_kriging_functional.R <in_dir> <out_dir>

# Suppress plot output from ICAOD's locally() function
pdf(NULL)

args <- commandArgs(trailingOnly = TRUE)
in_dir <- args[1]
out_dir <- args[2]

library(rkriging)
library(FNN)
library(ICAOD)
library(numDeriv)

# Read parameters
params <- read.csv(file.path(in_dir, "params.csv"))
seed <- params$seed[1]
functional_dir <- as.character(params$functional_dir[1])

set.seed(seed)

# Load design matrix from results.csv
MP <- read.csv(file.path(functional_dir, "results.csv"), header = TRUE)
q <- 5
D <- MP[-51, 1:q]  # Exclude run51 (best prediction)
colnames(D) <- NULL

# Transform to 0-1 scale
# Diffusivity, (1e-12,1e-9), log
D[, 1] <- (log(D[, 1]) - log(1e-12)) / (log(1e-9) - log(1e-12))
# Surface Concentration, (4e-3,5e-3), original
D[, 2] <- (D[, 2] - 4e-3) / (5e-3 - 4e-3)
# Accessible Polymer Concentration: (5e-3,6e-3), original
D[, 3] <- (D[, 3] - 5e-3) / (6e-3 - 5e-3)
# Hindering, (500,2500), original
D[, 4] <- (D[, 4] - 500) / (2500 - 500)
# Reaction rate, (1e-3,1e1), log
D[, 5] <- (log(D[, 5]) - log(1e-3)) / (log(1e1) - log(1e-3))

N <- dim(D)[1]

# Time points
pdet <- c(seq(0, 5000, length.out = 2001), seq(5000, 120000, length.out = 4601)[-1])
M_full <- length(pdet)

# Read functional outputs
filename <- paste0("run", 1:50, ".txt")
Y_full <- matrix(0, nrow = N, ncol = M_full)
for (i in 1:N) {
  Y_full[i, ] <- read.table(file.path(functional_dir, filename[i]))$V1
}

# Subsample to manageable size
tsamp <- seq(0, max(pdet), length = 1001)
M <- length(tsamp)
Ysamp <- matrix(0, nrow = N, ncol = M)
for (i in 1:N) {
  Ysamp[i, ] <- knn.reg(train = cbind(pdet), test = cbind(tsamp), y = Y_full[i, ], k = 1)$pred
}

tsamp01 <- tsamp / max(tsamp)

# Kriging correlation functions
D1 <- as.matrix(D[, 1:q])

r.x <- function(u, theta.x) {
  A <- t(t(D1) - u)
  basis.x <- function(h) exp(-sum((h / theta.x)^2))
  vec <- apply(A, 1, basis.x)
  return(vec)
}

r.t <- function(v, theta.t) {
  basis.t <- function(h) exp(-abs(h / theta.t))
  vec <- basis.t(v - tsamp01)
  return(vec)
}

Rtinv.mat <- function(rho) {
  A <- diag(M)
  for (i in 2:(M - 1)) {
    A[i, i - 1] <- A[i, i + 1] <- -rho
    A[i, i] <- 1 + rho^2
  }
  A[1, 2] <- A[M, M - 1] <- -rho
  return(A / (1 - rho^2))
}

# Maximum likelihood estimation
ML <- function(para) {
  theta.x <- para[1:q]
  theta.t <- para[q + 1]
  Rx <- apply(D1, 1, r.x, theta = theta.x)
  Rxinv <- solve(Rx + 10^(-6) * diag(N))
  rho <- exp(-1 / ((M - 1) * theta.t))
  Rtinv <- Rtinv.mat(rho)
  a <- c(t((Rxinv %*% rep(1, N)) %*% (t(rep(1, M)) %*% Rtinv)))
  b <- c(t(Rxinv %*% Ysamp %*% Rtinv))
  mu <- sum(b) / sum(a)
  sigma2 <- 1 / (M * N) * sum((c(t(Ysamp)) - mu) * (b - mu * a))
  val <- M * N * log(sigma2) + M * determinant(Rx, logarithm = TRUE)$mod[1] + M * N * log(1 - rho^2)
  return(val)
}

# Initial estimates from separate Kriging fits
y_mean_x <- apply(Ysamp, 1, mean)
a <- Fit.Kriging(D1, y_mean_x, kernel.parameters = list(type = "Gaussian"))
ini.x <- Get.Kriging.Parameters(a)$lengthscale * sqrt(2)

y_mean_t <- apply(Ysamp, 2, mean)
a <- Fit.Kriging(tsamp / max(tsamp), y_mean_t, kernel.parameters = list(type = "Matern12"))
ini.t <- Get.Kriging.Parameters(a)$lengthscale

ini <- c(ini.x, ini.t)
a.opt <- optim(ini, ML, lower = ini / 10, upper = ini * 10, method = "L-BFGS-B")
theta <- a.opt$par

# Compute prediction coefficients
theta.x <- theta[1:q]
theta.t <- theta[q + 1]
Rx <- apply(D1, 1, r.x, theta = theta.x)
A <- solve(Rx + 10^(-6) * diag(N))
Rxinv <- A
for (i in 1:3) Rxinv <- A %*% (diag(N) + 10^(-6) * Rxinv)
rho <- exp(-1 / ((M - 1) * theta.t))
Rtinv <- Rtinv.mat(rho)
a <- c(t((Rxinv %*% rep(1, N)) %*% (t(rep(1, M)) %*% Rtinv)))
b <- c(t(Rxinv %*% Ysamp %*% Rtinv))
mu <- sum(b) / sum(a)
coef <- c(b - mu * a)
COEF <- matrix(coef, nrow = N, ncol = M, byrow = TRUE)

# Prediction function
hhat <- function(ti, eta) drop(mu + t(r.x(eta, theta.x) %*% COEF %*% r.t(ti, theta.t)))

# FIM function for optimal design
fim <- function(x, w, param) {
  m <- length(w)
  S <- matrix(0, nrow = m, ncol = q)
  X <- matrix(x, nrow = m)
  for (i in 1:m) {
    heta <- function(eta) hhat(X[i, ], eta)
    S[i, ] <- grad(heta, x = param)
  }
  M_fim <- t(S) %*% diag(w) %*% S
  return(M_fim)
}

# Find optimal design using ICAOD
m <- q  # Number of design points
opdes <- locally(
  fimfunc = fim,
  lx = c(0), ux = c(1),
  inipars = rep(0.5, q),
  iter = 100, k = m,
  family = gaussian(),
  ICA.control = list("trace" = FALSE)
)

D_opt <- matrix(as.numeric(opdes$design[2:(m + 1)]), nrow = m, ncol = 1)
w <- as.numeric(opdes$design[(m + 2):(2 * m + 1)])

# Compute predictions at optimal design points
pred <- apply(cbind(tsamp01), 1, hhat, eta = rep(0.5, q))
y_D <- knn.reg(train = cbind(tsamp01), test = cbind(D_opt), y = pred, k = 1)$pred

# Save outputs
write.csv(
  data.frame(time = D_opt, y_pred = y_D, weight = w),
  file.path(out_dir, "optimal_design.csv"),
  row.names = FALSE
)

write.csv(
  data.frame(time = tsamp, pred = pred),
  file.path(out_dir, "prediction.csv"),
  row.names = FALSE
)
