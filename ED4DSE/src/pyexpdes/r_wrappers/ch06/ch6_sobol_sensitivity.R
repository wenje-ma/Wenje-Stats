args <- commandArgs(trailingOnly = TRUE)
in_dir <- args[1]
out_dir <- args[2]

suppressMessages(library(sensitivity))
suppressMessages(library(SFDesign))
suppressMessages(library(rkriging))

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
seed <- as.integer(params$seed[1])

set.seed(seed)

# Generate matrices for Sobol' analysis
A <- matrix(runif(m * p), nrow = m)
B <- matrix(runif(m * p), nrow = m)

# Sobol' sensitivity analysis
a.sen <- soboljansen(model = borehole, X1 = data.frame(A), X2 = data.frame(B))

# Extract first-order and total indices
sf <- pmax(unlist(a.sen$S), 0)
st <- pmax(unlist(a.sen$T), 0)

results <- data.frame(
  variable = paste0("x", 1:p),
  first_order = sf,
  total = st
)
write.csv(results, file.path(out_dir, "sobol_indices.csv"), row.names = FALSE)

# Main effects (using kriging surrogate)
n <- 10 * p
D <- maxpro.optim(maxproLHD(n, p)$design)$design
y <- apply(D, 1, borehole_f)

a.fit <- Fit.Kriging(D, y, kernel.parameters = list(type = "Gaussian"))
fhat <- function(x, fit) Predict.Kriging(fit, matrix(x, ncol = p))$mean

N <- 10 * p
D.eval <- uniformLHD(N, p)$design
f.eval <- apply(D.eval, 1, fhat, a.fit)
f0 <- mean(f.eval)

x.ev <- seq(0, 1, length = 20)
M <- matrix(0, nrow = 20, ncol = p)
for (j in 1:p) {
  D0 <- D.eval
  for (i in 1:20) {
    D0[, j] <- rep(x.ev[i], N)
    M[i, j] <- mean(apply(D0, 1, fhat, a.fit))
  }
  M[, j] <- M[, j] - f0
}

main_effects <- data.frame(x = x.ev)
for (j in 1:p) {
  main_effects[[paste0("x", j)]] <- M[, j]
}
write.csv(main_effects, file.path(out_dir, "main_effects.csv"), row.names = FALSE)

