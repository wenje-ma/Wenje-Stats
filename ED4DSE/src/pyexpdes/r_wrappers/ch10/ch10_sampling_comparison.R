# ch10_sampling_comparison.R
# Compare sampling methods: SPlit, twin, LPM1, LPM2
# Usage: Rscript ch10_sampling_comparison.R <in_dir> <out_dir>

args <- commandArgs(trailingOnly = TRUE)
in_dir <- args[1]
out_dir <- args[2]

library(twinning)
library(SPlit)
library(BalancedSampling)

# Read parameters
params <- read.csv(file.path(in_dir, "params.csv"))
seed <- params$seed[1]
N <- params$N[1]
n <- params$n[1]
n_rep <- params$n_rep[1]

set.seed(seed)

# Generate base data
X <- matrix(rnorm(2 * N), ncol = 2)

# Energy distance function
energy <- function(X, S) {
  n <- nrow(S)
  N <- nrow(X)
  # E[||X - S||] - 0.5 * E[||S - S'||]
  term1 <- mean(as.matrix(dist(rbind(S, X)))[1:n, (n + 1):(n + N)])
  term2 <- mean(as.matrix(dist(S)))
  return(2 * term1 - term2)
}

# Initialize result vectors
edsplit <- edtwin <- edlpm1 <- edlpm2 <- numeric(n_rep)
sbsplit <- sbtwin <- sblpm1 <- sblpm2 <- numeric(n_rep)

for (i in 1:n_rep) {
  # SPlit
  ind <- SPlit(X, splitRatio = n / N)
  edsplit[i] <- energy(X, X[ind, ])
  sbsplit[i] <- sb(rep(n / N, N), X, ind)

  # Twinning
  ind <- twin(X, r = N / n)
  edtwin[i] <- energy(X, X[ind, ])
  sbtwin[i] <- sb(rep(n / N, N), X, ind)

  # LPM1
  ind <- lpm1(rep(n / N, N), X)
  edlpm1[i] <- energy(X, X[ind, ])
  sblpm1[i] <- sb(rep(n / N, N), X, ind)

  # LPM2
  ind <- lpm2(rep(n / N, N), X)
  edlpm2[i] <- energy(X, X[ind, ])
  sblpm2[i] <- sb(rep(n / N, N), X, ind)
}

# Save outputs
write.csv(
  data.frame(SPlit = edsplit, twin = edtwin, LPM1 = edlpm1, LPM2 = edlpm2),
  file.path(out_dir, "energy_distance.csv"),
  row.names = FALSE
)
write.csv(
  data.frame(SPlit = sbsplit, twin = sbtwin, LPM1 = sblpm1, LPM2 = sblpm2),
  file.path(out_dir, "spatial_balance.csv"),
  row.names = FALSE
)
