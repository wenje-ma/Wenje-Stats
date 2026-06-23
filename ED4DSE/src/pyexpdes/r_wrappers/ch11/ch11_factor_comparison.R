# ch11_factor_comparison.R
# Factor selection with RF/GP comparison using FIRST algorithm
# Usage: Rscript ch11_factor_comparison.R <in_dir> <out_dir>

args <- commandArgs(trailingOnly = TRUE)
in_dir <- args[1]
out_dir <- args[2]

library(SPlit)
library(randomForest)
library(rkriging)
library(first)

# Read parameters
params <- read.csv(file.path(in_dir, "params.csv"))
seed <- params$seed[1]
n_rep <- params$n_rep[1]
split_ratio <- params$split_ratio[1]

# Read concrete data
concrete <- read.csv(file.path(in_dir, "concrete.csv"))

# Scale the data
concrete.scale <- data.frame(scale(concrete))
colnames(concrete.scale)[9] <- "y"

set.seed(seed)

rmse.RF <- rmse.GP <- matrix(0, nrow = n_rep, ncol = 2)

for (i in 1:n_rep) {
  # Split data
  ind <- SPlit(concrete, splitRatio = split_ratio)
  train <- concrete.scale[-ind, ]
  test <- concrete.scale[ind, ]

  # ----- All factors -----
  # Random Forest
  a <- randomForest(x = train[, 1:8], y = train[, 9])
  pred <- predict(a, test[, 1:8])
  rmse.RF[i, 1] <- sqrt(mean((test$y - pred)^2))

  # Gaussian Process
  a <- Fit.Kriging(
    X = as.matrix(train[, 1:8]),
    y = train[, 9],
    interpolation = FALSE,
    kernel.parameters = list(type = "Gaussian")
  )
  pred <- Predict.Kriging(a, as.matrix(test[, 1:8]))$mean
  rmse.GP[i, 1] <- sqrt(mean((test$y - pred)^2))

  # ----- Selected factors using FIRST -----
  sel <- first(X = as.matrix(train[, -9]), y = train[, 9])
  sel <- (sel > 0)

  # Random Forest with selected factors
  a <- randomForest(x = train[, -9][, sel], y = train[, 9])
  pred <- predict(a, test[, -9][, sel])
  rmse.RF[i, 2] <- sqrt(mean((test$y - pred)^2))

  # Gaussian Process with selected factors
  a <- Fit.Kriging(
    X = as.matrix(train[, -9][, sel]),
    y = train[, 9],
    interpolation = FALSE,
    kernel.parameters = list(type = "Gaussian")
  )
  pred <- Predict.Kriging(a, as.matrix(test[, -9][, sel]))$mean
  rmse.GP[i, 2] <- sqrt(mean((test$y - pred)^2))
}

# Save outputs
write.csv(
  data.frame(full = rmse.RF[, 1], select = rmse.RF[, 2]),
  file.path(out_dir, "rmse_rf.csv"),
  row.names = FALSE
)
write.csv(
  data.frame(full = rmse.GP[, 1], select = rmse.GP[, 2]),
  file.path(out_dir, "rmse_gp.csv"),
  row.names = FALSE
)
