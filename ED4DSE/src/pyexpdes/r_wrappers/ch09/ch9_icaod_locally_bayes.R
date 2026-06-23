# ch9_icaod_locally_bayes.R
# ICAOD locally and Bayesian D-optimal design for two-factor exponential model
# Usage: Rscript ch9_icaod_locally_bayes.R <in_dir> <out_dir>

# Suppress plot output
pdf(NULL)

args <- commandArgs(trailingOnly = TRUE)
in_dir <- args[1]
out_dir <- args[2]

library(ICAOD)

# Read parameters
params <- read.csv(file.path(in_dir, "params.csv"))
seed <- params$seed[1]
m <- params$m[1]  # Number of design points

set.seed(seed)

# Two-factor exponential model formula: eta0 + exp(-eta1*x1) + exp(-eta2*x2)

# Locally D-optimal design
opdes_local <- locally(
  formula = ~ eta0 + exp(-eta1 * x1) + exp(-eta2 * x2),
  predvars = c("x1", "x2"),
  parvars = c("eta0", "eta1", "eta2"),
  lx = c(0, 0), ux = c(1, 1),
  inipars = c(0, 2, 2),
  iter = 100, k = m,
  family = gaussian(),
  ICA.control = list("trace" = FALSE)
)

# Extract design matrix and weights
D_local <- matrix(as.numeric(opdes_local$design[2:(2 * m + 1)]), nrow = m, ncol = 2)
w_local <- as.numeric(opdes_local$design[(2 * m + 2):(2 * m + 1 + m)])
det_local <- as.numeric(opdes_local$design[2 * m + 2 + m])

# Save locally D-optimal design
write.csv(
  data.frame(x1 = D_local[, 1], x2 = D_local[, 2], weight = w_local),
  file.path(out_dir, "local_design.csv"),
  row.names = FALSE
)

# Pseudo-Bayesian D-optimal design
pr <- uniform(lower = c(-10, 1, 1), upper = c(10, 3, 3))
opdes_bayes <- bayes(
  formula = ~ eta0 + exp(-eta1 * x1) + exp(-eta2 * x2),
  predvars = c("x1", "x2"),
  parvars = c("eta0", "eta1", "eta2"),
  lx = c(0, 0), ux = c(1, 1),
  prior = pr,
  iter = 100, k = m,
  family = gaussian(),
  ICA.control = list("trace" = FALSE)
)

# Extract design matrix and weights
D_bayes <- matrix(as.numeric(opdes_bayes$design[2:(2 * m + 1)]), nrow = m, ncol = 2)
w_bayes <- as.numeric(opdes_bayes$design[(2 * m + 2):(2 * m + 1 + m)])
det_bayes <- as.numeric(opdes_bayes$design[2 * m + 2 + m])

# Save Bayesian D-optimal design
write.csv(
  data.frame(x1 = D_bayes[, 1], x2 = D_bayes[, 2], weight = w_bayes),
  file.path(out_dir, "bayes_design.csv"),
  row.names = FALSE
)

# Save summary info
write.csv(
  data.frame(
    design = c("locally", "bayes"),
    det_criterion = c(det_local, det_bayes)
  ),
  file.path(out_dir, "design_summary.csv"),
  row.names = FALSE
)
