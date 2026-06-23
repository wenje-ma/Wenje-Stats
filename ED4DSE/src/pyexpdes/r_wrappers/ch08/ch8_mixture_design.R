# ch8_mixture_design.R
# Mixture designs: SLD, SCD, and space-filling designs
# Usage: Rscript ch8_mixture_design.R <in_dir> <out_dir>

args <- commandArgs(trailingOnly = TRUE)
in_dir <- args[1]
out_dir <- args[2]

library(mixexp)
library(spacefillr)
library(support)

# Read parameters
params <- read.csv(file.path(in_dir, "params.csv"))
seed <- params$seed[1]

# Simplex Lattice Design (SLD) for 3 components, degree 2
D1 <- SLD(3, 2)
write.csv(D1, file.path(out_dir, "sld_design.csv"), row.names = FALSE)

# Simplex Centroid Design (SCD) for 3 components
D2 <- SCD(3)
write.csv(D2, file.path(out_dir, "scd_design.csv"), row.names = FALSE)

# Space-filling design on simplex using Sobol sequence
p <- 3
N <- 10000
u <- generate_sobol_set(N, p - 1)
x <- matrix(0, nrow = N, ncol = p)
x[, 1] <- (1 - u[, 1]^(1/2))
x[, 2] <- u[, 1]^(1/2) * (1 - u[, 2])
x[, 3] <- u[, 1]^(1/2) * u[, 2]

set.seed(seed)
n <- 11
D3 <- sp(n, p, dist.samp = x)$sp
write.csv(D3, file.path(out_dir, "spacefilling_design.csv"), row.names = FALSE)

# Constrained space-filling design (0.3 < x1 + x2 < 0.7)
CAND <- x
C1 <- (x[, 1] + x[, 2] < 0.7)
C2 <- (x[, 1] + x[, 2] > 0.3)
CAND <- CAND[C1 & C2, ]

set.seed(seed)
D4 <- sp(n, p, dist.samp = CAND)$sp
write.csv(D4, file.path(out_dir, "spacefilling_constrained.csv"), row.names = FALSE)
