args <- commandArgs(trailingOnly = TRUE)
in_dir <- args[1]
out_dir <- args[2]

suppressMessages(library(support))
suppressMessages(library(adaptMCMC))

# Read parameters
params <- read.csv(file.path(in_dir, "params.csv"))
n <- as.integer(params$n[1])
p <- as.integer(params$p[1])
seed <- as.integer(params$seed[1])
n_mcmc <- as.integer(params$n_mcmc[1])

set.seed(seed)

# Banana-shaped density (log scale)
logf <- function(para) {
  l1 <- -40
  u1 <- 40
  l2 <- -25
  u2 <- 10
  x1 <- l1 + (u1 - l1) * para[1]
  x2 <- l2 + (u2 - l2) * para[2]
  val <- -0.5 * (x1^2 / 100 + (x2 + 0.03 * x1^2 - 3)^2)
  return(val)
}

# Generate density grid for plotting
N.plot <- 100
p1 <- seq(0, 1, length.out = N.plot)
p2 <- seq(0, 1, length.out = N.plot)
grid <- expand.grid(p1 = p1, p2 = p2)
grid$fc <- exp(apply(grid, 1, logf))
write.csv(grid, file.path(out_dir, "banana_grid.csv"), row.names = FALSE)

# Run MCMC
a <- MCMC(logf, n = n_mcmc, init = c(0.5, 0.8), acc.rate = 0.4)
X <- a$samples
write.csv(as.data.frame(X), file.path(out_dir, "mcmc_samples.csv"), row.names = FALSE)

# Get support points from MCMC samples
D <- sp(n, p, dist.samp = X)$sp
write.csv(as.data.frame(D), file.path(out_dir, "support_points.csv"), row.names = FALSE)

