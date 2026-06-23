args <- commandArgs(trailingOnly = TRUE)
in_dir <- args[1]
out_dir <- args[2]

suppressMessages(library(sensitivity))
suppressMessages(library(SFDesign))

# Read parameters
params <- read.csv(file.path(in_dir, "params.csv"))
p <- as.integer(params$p[1])
seed <- as.integer(params$seed[1])

set.seed(seed)

# Test function: sum of sines with different frequencies
f <- function(x) {
  val <- 0
  for (k in 1:p) val <- val + sin((k^2 + 1) * pi * x[k]) / k
  return(val)
}
f1 <- function(X) apply(X, 1, f)

# Derivative-based sensitivity (DELSA)
X <- uniformLHD(10 * p, p)$design
a <- delsa(model = f1, X0 = X, varprior = rep(1, p))
nu <- colMeans(a$deriv^2)
nu <- nu / sum(nu)

# Variance-based sensitivity (Sobol' total indices)
m <- 10000
A <- matrix(runif(m * p), nrow = m)
B <- matrix(runif(m * p), nrow = m)
a.sen <- soboljansen(model = f1, X1 = data.frame(A), X2 = data.frame(B))
tot <- pmax(unlist(a.sen$T), 0)

results <- data.frame(
  variable = paste0("x", 1:p),
  nu = nu,
  T = tot
)
write.csv(results, file.path(out_dir, "comparison.csv"), row.names = FALSE)

