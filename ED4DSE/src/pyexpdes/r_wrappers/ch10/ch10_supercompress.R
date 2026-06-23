# ch10_supercompress.R
# Supervised compression vs unsupervised (k-means)
# Usage: Rscript ch10_supercompress.R <in_dir> <out_dir>

args <- commandArgs(trailingOnly = TRUE)
in_dir <- args[1]
out_dir <- args[2]

library(supercompress)
library(FNN)

# Read parameters
params <- read.csv(file.path(in_dir, "params.csv"))
seed <- params$seed[1]
N <- params$N[1]
n <- params$n[1]
p <- params$p[1]

set.seed(seed)

# Michalewicz function
f <- function(x) {
  p <- length(x)
  x <- pi * x
  val <- -sum(sin(x) * (sin((1:p) * x^2 / pi))^20)
  return(val)
}

# Generate random data
x <- matrix(runif(N * p), ncol = p)
y <- apply(x, 1, f) + 0.01 * rnorm(N)

# Unsupervised: K-means
a0 <- kmeans(x, n, iter.max = 100)
D0 <- a0$centers
cluster <- a0$cluster
y0 <- sapply(split(y, cluster), mean)

# Supervised: supercompress
a <- supercompress(n, x, y)
Dopt <- a$D
ybar <- a$ybar

# Compute RMSE for both
true <- apply(x, 1, f)

pred.kmeans <- knn.reg(D0, test = x, y0, k = 1)$pred
rmse.kmeans <- sqrt(mean((pred.kmeans - true)^2))

pred.super <- knn.reg(Dopt, test = x, ybar, k = 1)$pred
rmse.super <- sqrt(mean((pred.super - true)^2))

# Generate grid for visualization
N.plot <- 250
p1 <- seq(0, 1, length = N.plot)
p2 <- seq(0, 1, length = N.plot)
grid <- expand.grid(p1 = p1, p2 = p2)
fc <- apply(grid, 1, f)

# Save outputs
write.csv(D0, file.path(out_dir, "kmeans_centers.csv"), row.names = FALSE)
write.csv(Dopt, file.path(out_dir, "supercompress_centers.csv"), row.names = FALSE)
write.csv(
  data.frame(p1 = grid$p1, p2 = grid$p2, f = fc),
  file.path(out_dir, "grid.csv"),
  row.names = FALSE
)
write.csv(
  data.frame(method = c("kmeans", "supercompress"), rmse = c(rmse.kmeans, rmse.super)),
  file.path(out_dir, "rmse.csv"),
  row.names = FALSE
)
