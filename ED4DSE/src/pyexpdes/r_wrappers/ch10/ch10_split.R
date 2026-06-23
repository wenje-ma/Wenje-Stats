# ch10_split.R
# SPlit for optimal data splitting
# Usage: Rscript ch10_split.R <in_dir> <out_dir>

args <- commandArgs(trailingOnly = TRUE)
in_dir <- args[1]
out_dir <- args[2]

library(SPlit)

# Read parameters
params <- read.csv(file.path(in_dir, "params.csv"))
seed <- params$seed[1]
N <- params$N[1]
Ntest <- params$Ntest[1]

set.seed(seed)

# Generate synthetic data
x1 <- scale(rnorm(N))
x2 <- scale(x1^2 + rnorm(N))
D <- matrix(cbind(x1, x2), ncol = 2)

# Generate labels (3 classes)
y <- rep(2, N)
y[rbinom(N, 1, pnorm(-(6 * x1 + x2 + 5))) == 1] <- 1
y[rbinom(N, 1, pnorm(-(-6 * x1 + x2 + 5))) == 1] <- 3

# Create data frame for SPlit
Dy <- data.frame(x1 = D[, 1], x2 = D[, 2], y = as.factor(y))

# SPlit optimal split
ind.opt <- SPlit(Dy, splitRatio = Ntest / N)

# Random split
ind.random <- sample(1:N, Ntest)

# Save outputs
write.csv(D, file.path(out_dir, "data.csv"), row.names = FALSE)
write.csv(data.frame(y = y), file.path(out_dir, "labels.csv"), row.names = FALSE)
write.csv(data.frame(idx = ind.random), file.path(out_dir, "random_idx.csv"), row.names = FALSE)
write.csv(data.frame(idx = ind.opt), file.path(out_dir, "split_idx.csv"), row.names = FALSE)
