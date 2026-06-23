args <- commandArgs(trailingOnly = TRUE)
in_dir <- args[1]
out_dir <- args[2]

suppressMessages(library(support))
suppressMessages(library(mnormt))

# Read parameters
params <- read.csv(file.path(in_dir, "params.csv"))
n <- as.integer(params$n[1])
p <- as.integer(params$p[1])
dist_type <- as.character(params$dist_type[1])
seed <- as.integer(params$seed[1])

set.seed(seed)

if (dist_type == "normal") {
  # Independent normal distribution
  D <- sp(n, p, dist.str = rep("normal", p))$sp
  write.csv(as.data.frame(D), file.path(out_dir, "support_points.csv"), row.names = FALSE)

} else if (dist_type == "gamma") {
  # Gamma(2,1) distribution
  dist.param <- vector("list", p)
  for (l in 1:p) {
    dist.param[[l]] <- c(2, 1)
  }
  D <- sp(n, p, dist.str = rep("gamma", p), dist.param = dist.param)$sp
  write.csv(as.data.frame(D), file.path(out_dir, "support_points.csv"), row.names = FALSE)

} else if (dist_type == "correlated_normal") {
  # Correlated normal with rho=0.5
  N <- 10000
  X <- rmnorm(N, mean = c(0, 0), varcov = matrix(c(1, 0.5, 0.5, 1), 2, 2))
  D <- sp(n, p, dist.samp = X)$sp
  write.csv(as.data.frame(D), file.path(out_dir, "support_points.csv"), row.names = FALSE)

} else if (dist_type == "from_samples") {
  # From provided samples
  samples <- read.csv(file.path(in_dir, "samples.csv"))
  X <- as.matrix(samples)
  D <- sp(n, p, dist.samp = X)$sp
  write.csv(as.data.frame(D), file.path(out_dir, "support_points.csv"), row.names = FALSE)
}

