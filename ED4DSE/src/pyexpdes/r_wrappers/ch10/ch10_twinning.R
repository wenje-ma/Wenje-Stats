# ch10_twinning.R
# Data twinning example
# Usage: Rscript ch10_twinning.R <in_dir> <out_dir>

args <- commandArgs(trailingOnly = TRUE)
in_dir <- args[1]
out_dir <- args[2]

library(twinning)

# Read parameters
params <- read.csv(file.path(in_dir, "params.csv"))
seed <- params$seed[1]
N <- params$N[1]
r <- params$r[1]  # N/n ratio

set.seed(seed)

# Generate synthetic data
x <- scale(rnorm(N))
y <- scale(x^2 + rnorm(N))
D <- cbind(x, y)

# Twinning
ind.twin <- twin(D, r = r)

# Save outputs
write.csv(D, file.path(out_dir, "data.csv"), row.names = FALSE)
write.csv(data.frame(idx = ind.twin), file.path(out_dir, "twin_idx.csv"), row.names = FALSE)
