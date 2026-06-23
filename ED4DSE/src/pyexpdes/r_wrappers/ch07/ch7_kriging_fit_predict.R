args <- commandArgs(trailingOnly = TRUE)
in_dir <- args[1]
out_dir <- args[2]

suppressMessages(library(rkriging))

# Read design points and responses
D <- as.matrix(read.csv(file.path(in_dir, "design.csv")))
y <- read.csv(file.path(in_dir, "response.csv"))$y
test <- as.matrix(read.csv(file.path(in_dir, "test.csv")))

# Read optional initial theta
theta_file <- file.path(in_dir, "theta.csv")
if (file.exists(theta_file)) {
  theta <- read.csv(theta_file)$theta
  kernel <- Gaussian.Kernel(theta)
  obj1 <- Fit.Kriging(D, y, model = "OK", kernel = kernel)
  obj2 <- Fit.Kriging(D, y, kernel.parameters = list(type = "Gaussian"))
  if (obj1$get_nllh() < obj2$get_nllh()) obj <- obj1 else obj <- obj2
} else {
  obj <- Fit.Kriging(D, y, kernel.parameters = list(type = "Gaussian"))
}

# Get fitted parameters
theta_fitted <- obj$get_lengthscale()
write.csv(data.frame(theta = theta_fitted), file.path(out_dir, "theta.csv"), row.names = FALSE)

# Predict on test points
pred <- Predict.Kriging(obj, test)
results <- data.frame(
  mean = pred$mean,
  sd = pred$sd
)
write.csv(results, file.path(out_dir, "predictions.csv"), row.names = FALSE)

# Also output correlation matrix info for ALM/ALC/ALMV calculations
n <- nrow(D)
p <- ncol(D)
kernel <- Gaussian.Kernel(theta_fitted)
R <- Evaluate.Kernel(kernel, D)
Rinv <- solve(R + 1e-6 * diag(n))
write.csv(as.data.frame(Rinv), file.path(out_dir, "Rinv.csv"), row.names = FALSE)

