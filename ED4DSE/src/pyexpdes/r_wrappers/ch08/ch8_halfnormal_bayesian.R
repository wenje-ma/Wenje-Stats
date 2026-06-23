# ch8_halfnormal_bayesian.R
# Bayesian-inspired half-normal plot using Kriging + HiGarrote
# Usage: Rscript ch8_halfnormal_bayesian.R <in_dir> <out_dir>

args <- commandArgs(trailingOnly = TRUE)
in_dir <- args[1]
out_dir <- args[2]

library(rkriging)
library(HiGarrote)

# Read inputs
design <- read.csv(file.path(in_dir, "design.csv"))
response <- read.csv(file.path(in_dir, "response.csv"))

A <- design$A
B <- design$B
C <- design$C
D <- design$D
y <- response$y

# Fit Kriging model
d <- cbind(A, B, C, D)
a <- Fit.Kriging(d, y, kernel.parameters = list(type = "Gaussian"))
theta <- Get.Kriging.Parameters(a)$lengthscale
mu <- Get.Kriging.Parameters(a)$mu

# Calculate gamma values
gam <- (1 - exp(-2^2 / (2 * theta))) / (1 + exp(-2^2 / (2 * theta)))

# Build full factorial design matrix
X <- model.matrix(lm(y ~ (A + B + C + D)^4), data = data.frame(A, B, C, D))

# Calculate G matrix (diagonal weights)
G <- diag(c(model.matrix(lm(1 ~ (gam[1] + gam[2] + gam[3] + gam[4])^4))))

# Calculate Bayesian effects
alpha <- G %*% t(X) %*% solve(X %*% G %*% t(X)) %*% (y - mu)
eff <- alpha[-1]  # Exclude intercept

# Effect names
eff_names <- c("1", "2", "3", "4", "12", "13", "14", "23", "24", "34",
               "123", "124", "134", "234", "1234")

# Output
result <- data.frame(
  name = eff_names,
  effect = as.numeric(eff)
)
write.csv(result, file.path(out_dir, "bayesian_effects.csv"), row.names = FALSE)

# Also run HiGarrote for variable selection
hg_result <- HiGarrote(d, y)
hg_df <- data.frame(
  variable = names(hg_result$coefficients),
  coefficient = as.numeric(hg_result$coefficients)
)
write.csv(hg_df, file.path(out_dir, "higarrote_result.csv"), row.names = FALSE)
