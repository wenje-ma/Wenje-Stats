# ch10_support_subsample.R
# Support points and subsample for bivariate normal data
# Usage: Rscript ch10_support_subsample.R <in_dir> <out_dir>

args <- commandArgs(trailingOnly = TRUE)
in_dir <- args[1]
out_dir <- args[2]

library(support)
library(SPlit)
library(MASS)

# Read parameters
params <- read.csv(file.path(in_dir, "params.csv"))
seed <- params$seed[1]
N <- params$N[1]
n <- params$n[1]
p <- params$p[1]
rho <- params$rho[1]

set.seed(seed)

# Generate correlated bivariate normal data
Sigma <- diag(p)
Sigma <- rho^abs(row(Sigma) - col(Sigma))
D <- mvrnorm(n = N, mu = rep(0, p), Sigma = Sigma)

# Compute support points
S <- sp(n, p, dist.samp = D)$sp

# Get subsample indices using SPlit::subsample
ind <- subsample(D, S)

# Save outputs
write.csv(D, file.path(out_dir, "data.csv"), row.names = FALSE)
write.csv(S, file.path(out_dir, "support_points.csv"), row.names = FALSE)
write.csv(D[ind, ], file.path(out_dir, "subsample.csv"), row.names = FALSE)
