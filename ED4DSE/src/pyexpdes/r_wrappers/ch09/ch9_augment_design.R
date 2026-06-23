# ch9_augment_design.R
# MaxPro design augmentation for robust design
# Usage: Rscript ch9_augment_design.R <in_dir> <out_dir>

args <- commandArgs(trailingOnly = TRUE)
in_dir <- args[1]
out_dir <- args[2]

library(support)
library(MaxPro)

# Read parameters
params <- read.csv(file.path(in_dir, "params.csv"))
seed <- params$seed[1]
N <- params$N[1]  # Number of parameter samples
n_initial <- params$n_initial[1]  # Initial design points
n_new <- params$n_new[1]  # New points to add
n_cand <- params$n_cand[1]  # Candidate points for augmentation

set.seed(seed)

# Support points resampling function
sp.sample <- function(x, n, prob, tol = 1e-6, iter.max = 10) {
  if (is.null(dim(x))) stop("x must be a matrix!")
  N <- nrow(x)
  x.dist <- as.matrix(dist(x))
  x.measure <- c(x.dist %*% prob)

  idx <- c(which.min(x.measure))
  if (prob[idx[1]] < 1 / N) idx[1] <- which.max(prob)

  for (i in 2:n) {
    measure <- x.measure - apply(matrix(x.dist[, idx], ncol = (i - 1)), 1, sum) / i
    idx <- c(idx, which.min(measure))
  }
  edist <- 2 * sum(x.measure[idx]) / n - sum(x.dist[idx, idx]) / n^2

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

# Generate parameter samples using support points
p <- 2
samp <- sp(n = N, p = 2, dist.str = c("uniform", "normal"))$sp
samp[, 1] <- 1 + 2 * samp[, 1]  # eta1 ~ Uniform(1, 3)
samp[, 2] <- 2 + 1/2 * samp[, 2]  # eta2 ~ Normal(2, 0.5)

# Build candidate design matrix
D <- NULL
for (i in 1:N) {
  D <- rbind(D, matrix(c(
    0, 0,
    1/samp[i, 1], 0,
    0, 1/samp[i, 2],
    1/samp[i, 1], 1/samp[i, 2]
  ), ncol = 2, byrow = TRUE))
}

# Select initial robust design
ind <- sp.sample(D, n_initial, prob = rep(0.25 / N, nrow(D)))
D_initial <- D[ind, ]

# Generate candidate points using MaxPro LHD
cand <- MaxProLHD(n_cand, 2)$Design

# Augment design using MaxPro criterion
aug_result <- MaxProAugment(ExistDesign = D_initial, CandDesign = cand, nNew = n_new)
D_new <- aug_result$Design[(n_initial + 1):(n_initial + n_new), ]

# Save outputs
write.csv(D, file.path(out_dir, "candidate_design.csv"), row.names = FALSE)
write.csv(D_initial, file.path(out_dir, "initial_design.csv"), row.names = FALSE)
write.csv(D_new, file.path(out_dir, "new_design.csv"), row.names = FALSE)
