# ch11_twingp.R
# Twin-Gaussian Process for wind energy data
# Usage: Rscript ch11_twingp.R <in_dir> <out_dir>

args <- commandArgs(trailingOnly = TRUE)
in_dir <- args[1]
out_dir <- args[2]

library(twingp)

# Read parameters
params <- read.csv(file.path(in_dir, "params.csv"))
seed <- params$seed[1]

# Read wind data
wind <- read.csv(file.path(in_dir, "wind.csv"), header = TRUE)
D <- as.matrix(wind[, c("WindSpeed", "Power")])

# Evaluation grid
ev <- seq(min(D[, 1]), max(D[, 1]), length = 1000)

set.seed(seed)

# Fit TwinGP
a.twin <- twingp(x = cbind(D[, 1]), y = D[, 2], x_test = cbind(ev))
pred.twin <- a.twin$mu

# Save outputs
write.csv(
  data.frame(speed = ev, power = pred.twin),
  file.path(out_dir, "prediction.csv"),
  row.names = FALSE
)
