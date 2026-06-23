# ch2_gp_regression.R
# Gaussian Process Regression with equi-spaced vs D-optimal designs
# Usage: Rscript ch2_gp_regression.R <in_dir> <out_dir>

args <- commandArgs(trailingOnly = TRUE)
in_dir <- args[1]
out_dir <- args[2]

library(rkriging)
library(AlgDesign)

# Read data
data <- read.csv(file.path(in_dir, "data.csv"))
params <- read.csv(file.path(in_dir, "params.csv"))

seed <- params$seed[1]
n <- params$n[1]
re <- params$re[1]  # replicates

set.seed(seed)

# Equi-spaced design
D_equi <- ((1:n) - 1) / (n - 1)
D_equi <- rep(D_equi, re)
e <- rnorm(n * re, sd = 0.1)

# D-optimal design
cand <- data.frame(x = seq(-1, 1, length = 301))
a <- optFederov(
  ~ 1 + x + I(x^2) + I(x^3) + I(x^4) + I(x^5) + I(x^6) + I(x^7) + I(x^8) + I(x^9),
  nTrials = n, data = cand, approximate = FALSE
)
D_opt <- rep(c(a$design[, 1]), re)
D_opt <- (D_opt + 1) / 2

# Test points
test <- seq(0, 1, length = 301)

# Function
f <- function(x) sin(10 * pi * x) / (1 + 64 * (x - 0.25)^2) + x^2

# Equi-spaced: GP regression
y_equi <- f(D_equi) + e
a_equi <- Fit.Kriging(D_equi, y_equi, interpolation = FALSE, kernel.parameters = list(type = "Gaussian"))
pred_equi <- Predict.Kriging(a_equi, test)

# D-optimal: GP regression
y_opt <- f(D_opt) + e
a_opt <- Fit.Kriging(D_opt, y_opt, interpolation = FALSE, kernel.parameters = list(type = "Gaussian"))
pred_opt <- Predict.Kriging(a_opt, test)

# Save equi-spaced results
write.csv(
  data.frame(x = D_equi, y = y_equi),
  file.path(out_dir, "equi_data.csv"),
  row.names = FALSE
)
write.csv(
  data.frame(test = test, mean = pred_equi$mean, sd = pred_equi$sd),
  file.path(out_dir, "equi_pred.csv"),
  row.names = FALSE
)

# Save D-optimal results
write.csv(
  data.frame(x = D_opt, y = y_opt),
  file.path(out_dir, "dopt_data.csv"),
  row.names = FALSE
)
write.csv(
  data.frame(test = test, mean = pred_opt$mean, sd = pred_opt$sd),
  file.path(out_dir, "dopt_pred.csv"),
  row.names = FALSE
)
