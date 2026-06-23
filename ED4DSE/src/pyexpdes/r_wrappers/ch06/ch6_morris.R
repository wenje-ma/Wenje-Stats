args <- commandArgs(trailingOnly = TRUE)
in_dir <- args[1]
out_dir <- args[2]

suppressMessages(library(sensitivity))

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
r <- as.integer(params$r[1])
n_rep <- as.integer(params$n_rep[1])
seed <- as.integer(params$seed[1])

set.seed(seed)

# Single Morris analysis for mu/sigma plot
a <- morris(
  model = borehole, factors = p, r = r,
  design = list(type = "oat", levels = 4, grid.jump = 2)
)

mu <- abs(apply(a$ee, 2, mean))
sigma <- apply(a$ee, 2, sd)
mustar <- apply(a$ee, 2, function(x) mean(abs(x)))

results <- data.frame(
  variable = paste0("x", 1:p),
  mu = mu,
  sigma = sigma,
  mustar = mustar
)
write.csv(results, file.path(out_dir, "morris_single.csv"), row.names = FALSE)

# Multiple replications for boxplot
mustar_all <- matrix(0, nrow = n_rep, ncol = p)
for (i in 1:n_rep) {
  am <- morris(
    model = borehole, factors = p, r = r,
    design = list(type = "oat", levels = 4, grid.jump = 2)
  )
  mustar_all[i, ] <- apply(am$ee, 2, function(x) mean(abs(x)))
}

colnames(mustar_all) <- paste0("x", 1:p)
write.csv(as.data.frame(mustar_all), file.path(out_dir, "morris_boxplot.csv"), row.names = FALSE)

