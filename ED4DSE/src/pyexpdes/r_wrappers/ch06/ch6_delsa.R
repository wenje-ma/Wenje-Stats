args <- commandArgs(trailingOnly = TRUE)
in_dir <- args[1]
out_dir <- args[2]

suppressMessages(library(sensitivity))
suppressMessages(library(SFDesign))

# Borehole function
borehole_f <- function(x) {
  lower <- c(0.05, 100, 63070, 990, 63.1, 700, 1120, 9855)
  upper <- c(0.15, 50000, 115600, 1110, 116, 820, 1680, 12045)
  x <- lower + x * (upper - lower)
  val <- 2 * pi * x[3] * (x[4] - x[6]) / (log(x[2] / x[1]) * (1 + 2 * x[7] * x[3] / (log(x[2] / x[1]) * x[1]^2 * x[8]) + x[3] / x[5]))
  return(val)
}
borehole <- function(X) apply(X, 1, borehole_f)

# Read parameters
params <- read.csv(file.path(in_dir, "params.csv"))
p <- as.integer(params$p[1])
m <- as.integer(params$m[1])
n_rep <- as.integer(params$n_rep[1])
seed <- as.integer(params$seed[1])

set.seed(seed)

# DELSA analysis with multiple replications
nu <- matrix(0, nrow = n_rep, ncol = p)
for (i in 1:n_rep) {
  X <- matrix(runif(m * p), ncol = p)
  ad <- delsa(model = borehole, X0 = X, varprior = rep(1, p))
  nu[i, ] <- colMeans(ad$deriv^2)
  nu[i, ] <- nu[i, ] / sum(nu[i, ])
}

colnames(nu) <- paste0("x", 1:p)
write.csv(as.data.frame(nu), file.path(out_dir, "delsa_results.csv"), row.names = FALSE)

