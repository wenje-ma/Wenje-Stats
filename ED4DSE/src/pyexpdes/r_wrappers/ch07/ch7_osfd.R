args <- commandArgs(trailingOnly = TRUE)
in_dir <- args[1]
out_dir <- args[2]

suppressMessages(library(OSFD))
suppressMessages(library(MaxPro))

# Read parameters
params <- read.csv(file.path(in_dir, "params.csv"))
p <- as.integer(params$p[1])
q <- as.integer(params$q[1])
n <- as.integer(params$n[1])
n0 <- as.integer(params$n0[1])
seed <- as.integer(params$seed[1])

set.seed(seed)

# Inverse design function (polar coordinates example)
f <- function(x) {
  y1 <- 1 / (x[1]^2 + x[2]^2 + 0.01)^(1 / 2)
  if (x[2] == 0) {
    y2 <- 0
  } else if (x[1] == 0) {
    y2 <- pi / 2
  } else {
    y2 <- atan(x[2] / x[1])
  }
  return(c(y1 = y1, y2 = y2))
}

# Initial design
D0 <- MaxProLHD(n0, p)$Design

# Run OSFD
osfd <- OSFD(D = D0, f = f, p = p, q = q, n = n, method = "Greedy")

# Output results
write.csv(as.data.frame(osfd$D), file.path(out_dir, "design_D.csv"), row.names = FALSE)
write.csv(as.data.frame(osfd$Y), file.path(out_dir, "design_Y.csv"), row.names = FALSE)

