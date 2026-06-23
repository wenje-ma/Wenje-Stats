args <- commandArgs(trailingOnly = TRUE)
in_dir <- args[1]
out_dir <- args[2]

suppressMessages(library(SFDesign))
suppressMessages(library(spacefillr))

# Read parameters
params <- read.csv(file.path(in_dir, "params.csv"))
n_seq <- as.integer(unlist(strsplit(as.character(params$n_seq[1]), ",")))
n_rep <- as.integer(params$n_rep[1])
p <- as.integer(params$p[1])
seed <- as.integer(params$seed[1])

set.seed(seed)

# Integration target function
h <- function(x) exp(-sum((x - 0.5)^2))

# True integral value (for p=2)
# pi * (pnorm(0.5*sqrt(2)) - pnorm(-0.5*sqrt(2)))^2
true_val <- pi * (pnorm(0.5 * sqrt(2)) - pnorm(-0.5 * sqrt(2)))^p

# Simulation results
results <- data.frame()

for (j in seq_along(n_seq)) {
  n <- n_seq[j]
  for (i in 1:n_rep) {
    # Random (Monte Carlo)
    R <- matrix(runif(n * p), ncol = p)
    rand_crit <- uniform.crit(R)
    rand_int <- mean(apply(R, 1, h))

    # Sobol
    S <- generate_sobol_set(n, p, seed = sample(1:10000, 1))
    sob_crit <- uniform.crit(S)
    sob_int <- mean(apply(S, 1, h))

    # Uniform design
    D <- uniform.optim(uniformLHD(n, p)$design)$design
    uni_crit <- uniform.crit(D)
    uni_int <- mean(apply(D, 1, h))

    results <- rbind(results, data.frame(
      n = n, rep = i,
      rand_crit = rand_crit, sob_crit = sob_crit, uni_crit = uni_crit,
      rand_int = rand_int, sob_int = sob_int, uni_int = uni_int
    ))
  }
}

write.csv(results, file.path(out_dir, "simulation_results.csv"), row.names = FALSE)
write.csv(data.frame(true_val = true_val), file.path(out_dir, "true_val.csv"), row.names = FALSE)

