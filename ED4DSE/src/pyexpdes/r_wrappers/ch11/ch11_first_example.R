# ch11_first_example.R
# FIRST algorithm demonstration with synthetic correlated data
# Usage: Rscript ch11_first_example.R <in_dir> <out_dir>

args <- commandArgs(trailingOnly = TRUE)
in_dir <- args[1]
out_dir <- args[2]

library(MASS)
library(first)

# Read parameters
params <- read.csv(file.path(in_dir, "params.csv"))
seed <- params$seed[1]
N <- params$N[1]
rho <- params$rho[1]

set.seed(seed)

# Generate correlated data (as in R code)
p <- 2
Sigma <- diag(p)
Sigma <- rho^abs(row(Sigma) - col(Sigma))
D <- mvrnorm(n = N, mu = rep(0, p), Sigma = Sigma)
D <- cbind(D, rnorm(N))  # Add independent x3

# Response: y = 3*x1 + 2*x2 + sqrt(3)*x3
y <- 3 * D[, 1] + 2 * D[, 2] + sqrt(3) * D[, 3]

# Run FIRST algorithm
first_result <- first(X = D, y = y)
first_rank_result <- first.rank(X = D, y = y, noise = FALSE)

# Also run with x2 removed (to show effect)
first_without_x2 <- first(X = D[, -2], y = y)

# Save FIRST scores
write.csv(
  data.frame(
    variable = paste0("x", 1:3),
    importance = as.numeric(first_result)
  ),
  file.path(out_dir, "first_scores.csv"),
  row.names = FALSE
)

# Save FIRST without x2
write.csv(
  data.frame(
    variable = c("x1", "x3"),
    importance = as.numeric(first_without_x2)
  ),
  file.path(out_dir, "first_scores_without_x2.csv"),
  row.names = FALSE
)

# Save FIRST rankings
write.csv(
  data.frame(
    rank = 1:length(first_rank_result),
    variable = paste0("x", first_rank_result)
  ),
  file.path(out_dir, "first_rankings.csv"),
  row.names = FALSE
)
