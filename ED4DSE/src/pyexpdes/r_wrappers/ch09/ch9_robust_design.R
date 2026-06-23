# ch9_robust_design.R
# Robust design via support points resampling
# Two-factor exponential model: Locally D-optimal vs Robust design
# Usage: Rscript ch9_robust_design.R <in_dir> <out_dir>

args <- commandArgs(trailingOnly = TRUE)
in_dir <- args[1]
out_dir <- args[2]

library(support)

# Read parameters
params <- read.csv(file.path(in_dir, "params.csv"))
seed <- params$seed[1]
N <- params$N[1]  # Number of parameter samples
n_select <- params$n_select[1]  # Number of design points

set.seed(seed)

# Support points resampling function (from https://github.com/BillHuang01/PQMC/blob/master/scripts/lib.R)
sp.sample <- function(x, n, prob, tol = 1e-6, iter.max = 10) {
  if (is.null(dim(x))) stop("x must be a matrix!")
  N <- nrow(x)
  x.dist <- as.matrix(dist(x))
  x.measure <- c(x.dist %*% prob)

  # Initial sample
  idx <- c(which.min(x.measure))
  if (prob[idx[1]] < 1 / N) idx[1] <- which.max(prob)

  for (i in 2:n) {
    measure <- x.measure - apply(matrix(x.dist[, idx], ncol = (i - 1)), 1, sum) / i
    idx <- c(idx, which.min(measure))
  }
  edist <- 2 * sum(x.measure[idx]) / n - sum(x.dist[idx, idx]) / n^2

  # Iterative update
  iter <- 0
  while (TRUE) {
    iter <- iter + 1
    for (i in 1:n) {
      measure <- x.measure - apply(x.dist[, idx[-i]], 1, sum) / n
      idx[i] <- which.min(measure)
    }
    edist.new <- 2 * sum(x.measure[idx]) / n - sum(x.dist[idx, idx]) / n^2
    if ((edist - edist.new) < tol || iter > iter.max) {
      break
    } else {
      edist <- edist.new
    }
  }
  return(idx)
}

# Locally D-optimal design for two-factor exponential model
# y = eta0 + exp(-eta1*x1) + exp(-eta2*x2)
# At eta = (0, 2, 2), the D-optimal design is:
L <- matrix(c(0, 0, 1/2, 0, 0, 1/2, 1/2, 1/2), ncol = 2, byrow = TRUE)

# Generate parameter samples using support points
p <- 2
samp <- sp(n = N, p = 2, dist.str = c("uniform", "normal"))$sp
# Transform: eta1 ~ Uniform(1, 3), eta2 ~ Normal(2, 0.5)
samp[, 1] <- 1 + 2 * samp[, 1]
samp[, 2] <- 2 + 1/2 * samp[, 2]

# Build candidate design matrix from different parameter values
D <- NULL
for (i in 1:N) {
  # For each parameter sample, compute the locally optimal design points
  D <- rbind(D, matrix(c(
    0, 0,
    1/samp[i, 1], 0,
    0, 1/samp[i, 2],
    1/samp[i, 1], 1/samp[i, 2]
  ), ncol = 2, byrow = TRUE))
}

# Select robust design using support points resampling
ind <- sp.sample(D, n_select, prob = rep(0.25 / N, nrow(D)))

# Save outputs
write.csv(L, file.path(out_dir, "local_design.csv"), row.names = FALSE)
write.csv(D, file.path(out_dir, "candidate_design.csv"), row.names = FALSE)
write.csv(D[ind, ], file.path(out_dir, "robust_design.csv"), row.names = FALSE)
