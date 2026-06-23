# ch10_wind_gp.R
# Wind energy data subsampling with GP fit
# Usage: Rscript ch10_wind_gp.R <in_dir> <out_dir>

args <- commandArgs(trailingOnly = TRUE)
in_dir <- args[1]
out_dir <- args[2]

library(support)
library(SPlit)
library(rkriging)

# Read parameters
params <- read.csv(file.path(in_dir, "params.csv"))
seed <- params$seed[1]
n <- params$n[1]

# Read wind data
wind <- read.csv(file.path(in_dir, "wind.csv"), header = TRUE)
p <- 2
D <- as.matrix(wind[, c("WindSpeed", "Power")])

set.seed(seed)

# Compute support points and subsample
S <- sp(n, p, dist.samp = D)$sp
ind <- subsample(D, S)
S <- D[ind, ]

# Fit GP model (speed -> power)
a <- Fit.Kriging(
  X = cbind(S[, 1]),
  y = S[, 2],
  kernel.parameters = list(type = "Gaussian"),
  interpolation = FALSE
)

# Predict on evaluation grid
ev <- seq(min(D[, 1]), max(D[, 1]), length = 1000)
pred <- Predict.Kriging(a, cbind(ev))$mean

# Save outputs
write.csv(S, file.path(out_dir, "subsample.csv"), row.names = FALSE)
write.csv(
  data.frame(speed = ev, power = pred),
  file.path(out_dir, "prediction.csv"),
  row.names = FALSE
)
