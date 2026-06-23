args <- commandArgs(trailingOnly = TRUE)
in_dir <- args[1]
out_dir <- args[2]

suppressMessages(library(MOFAT))

# Borehole function
borehole_f <- function(x) {
  lower <- c(0.05, 100, 63070, 990, 63.1, 700, 1120, 9855)
  upper <- c(0.15, 50000, 115600, 1110, 116, 820, 1680, 12045)
  x <- lower + x * (upper - lower)
  val <- 2 * pi * x[3] * (x[4] - x[6]) / (log(x[2] / x[1]) * (1 + 2 * x[7] * x[3] / (log(x[2] / x[1]) * x[1]^2 * x[8]) + x[3] / x[5]))
  return(val)
}

# Read parameters
params <- read.csv(file.path(in_dir, "params.csv"))
p <- as.integer(params$p[1])
l <- as.integer(params$l[1])
n_rep <- as.integer(params$n_rep[1])
seed <- as.integer(params$seed[1])

set.seed(seed)

# Generate MOFAT design for visualization (p=2)
if (p == 2) {
  D_uniform <- mofat(p, l, method = "uniform")
  write.csv(as.data.frame(D_uniform), file.path(out_dir, "mofat_uniform_2d.csv"), row.names = FALSE)

  D_projection <- mofat(p, l, method = "projection")
  write.csv(as.data.frame(D_projection), file.path(out_dir, "mofat_projection_2d.csv"), row.names = FALSE)
}

# MOFAT analysis with borehole (p=8)
mustar_all <- matrix(0, nrow = n_rep, ncol = 8)
for (i in 1:n_rep) {
  d <- mofat(p = 8, l = l)
  y <- apply(d, 1, borehole_f)
  m <- measure(d, y)$mustar
  m <- m / sum(m)  # Normalize
  mustar_all[i, ] <- m
}

colnames(mustar_all) <- paste0("x", 1:8)
write.csv(as.data.frame(mustar_all), file.path(out_dir, "mofat_boxplot.csv"), row.names = FALSE)

