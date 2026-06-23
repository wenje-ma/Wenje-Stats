args <- commandArgs(trailingOnly = TRUE)
in_dir <- args[1]
out_dir <- args[2]

suppressMessages(library(SFDesign))
suppressMessages(library(spacefillr))

# Read parameters
params <- read.csv(file.path(in_dir, "params.csv"))
n <- as.integer(params$n[1])
p <- as.integer(params$p[1])
seed <- as.integer(params$seed[1])

set.seed(seed)

# Generate uniform design
D <- uniform.optim(uniformLHD(n, p)$design)$design
write.csv(as.data.frame(D), file.path(out_dir, "uniform_design.csv"), row.names = FALSE)

# Generate random design (Monte Carlo)
R <- matrix(runif(n * p), ncol = p)
write.csv(as.data.frame(R), file.path(out_dir, "random_design.csv"), row.names = FALSE)

# Generate Sobol sequence
S <- generate_sobol_set(n, p, seed = sample(1:10000, 1))
write.csv(as.data.frame(S), file.path(out_dir, "sobol_design.csv"), row.names = FALSE)

# Compute uniformity criteria
crits <- data.frame(
  method = c("random", "sobol", "uniform"),
  crit = c(uniform.crit(R), uniform.crit(S), uniform.crit(D))
)
write.csv(crits, file.path(out_dir, "uniform_crit.csv"), row.names = FALSE)

