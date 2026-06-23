args <- commandArgs(trailingOnly = TRUE)
in_dir <- args[1]
out_dir <- args[2]

suppressMessages(library(rkriging))
suppressMessages(library(MaxPro))
suppressMessages(library(spacefillr))
suppressMessages(library(FNN))
suppressMessages(library(pdist))

# Read parameters
params <- read.csv(file.path(in_dir, "params.csv"))
method <- as.character(params$method[1])
p <- as.integer(params$p[1])
n0 <- as.integer(params$n0[1])
n_new <- as.integer(params$n_new[1])
seed <- as.integer(params$seed[1])

set.seed(seed)

# Branin function (scaled to [0,1]^p, uses first 2 dims)
f <- function(x) {
  x1 <- x[1] * 15 - 5
  x2 <- x[2] * 15
  val1 <- (x2 - 5.1 / (4 * pi^2) * (x1^2) + 5 / pi * x1 - 6)^2
  val2 <- 10 * (1 - 1 / (8 * pi)) * cos(x1) + 10
  return(val1 + val2)
}

# Initial design using MaxPro
D0 <- MaxPro(MaxProLHD(n0, p)$Design)$Design
y0 <- apply(D0, 1, f)

# Test points for RMSE calculation
test <- generate_sobol_set(1000 * p, p, seed = sample(10^6, 1))
truey <- apply(test, 1, f)

# ============================================================
# ALC: Active Learning Cohn
# ============================================================
ALC <- function(D0, y0, Nnew) {
  D <- D0
  y <- y0
  n <- nrow(D)
  rmse <- numeric(Nnew + 1)

  for (i in 1:Nnew) {
    CAND0 <- generate_sobol_set(100 * p, p, seed = sample(10^6, 1))
    if (i == 1) theta <- rep(1, p)

    kernel <- Gaussian.Kernel(theta)
    obj1 <- Fit.Kriging(D, y, model = "OK", kernel = kernel)
    obj2 <- Fit.Kriging(D, y, kernel.parameters = list(type = "Gaussian"))
    if (obj1$get_nllh() < obj2$get_nllh()) obj <- obj1 else obj <- obj2
    theta <- obj$get_lengthscale()

    pred <- Predict.Kriging(obj, test)$mean
    rmse[i] <- sqrt(mean((pred - truey)^2))

    kernel <- Gaussian.Kernel(theta)
    R <- Evaluate.Kernel(kernel, D)
    Rinv <- solve(R + 10^(-6) * diag(n))

    r <- function(x, theta) {
      if (p > 1) {
        A <- D - rep(1, n) %*% t(x)
        g <- exp(-0.5 * apply(abs(A %*% diag(1 / theta))^2, 1, sum))
      } else {
        g <- exp(-0.5 * abs((D - x) / theta)^2)
      }
      return(g)
    }
    basis <- function(h, theta) exp(-0.5 * sum((h / theta)^2))

    opt <- function(v, theta) {
      a <- r(v, theta)
      b <- c(Rinv %*% a)
      numer <- function(x, v, theta) {
        val <- (basis(x - v, theta) - sum(b * r(x, theta)))^2
        return(val)
      }
      val <- mean(apply(CAND0, 1, numer, v = v, theta = theta)) / max(1 - sum(a * b), 10^(-6))
      return(val)
    }

    d <- c(knn.dist(D, k = 1)) / 2
    dis <- as.matrix(pdist(D, CAND0))
    N <- dim(CAND0)[1]
    ind <- NULL
    for (j in 1:n) ind <- c(ind, (1:N)[dis[j, ] < d[j]])
    if (length(ind) > 0) CAND <- CAND0[-ind, ] else CAND <- CAND0

    xnew <- CAND[which.max(apply(CAND, 1, opt, theta = theta)), ]
    D <- rbind(D, xnew)
    y <- c(y, f(xnew))
    n <- n + 1
  }

  kernel <- Gaussian.Kernel(theta)
  obj1 <- Fit.Kriging(D, y, model = "OK", kernel = kernel)
  obj2 <- Fit.Kriging(D, y, kernel.parameters = list(type = "Gaussian"))
  if (obj1$get_nllh() < obj2$get_nllh()) obj <- obj1 else obj <- obj2
  pred <- Predict.Kriging(obj, test)$mean
  rmse[Nnew + 1] <- sqrt(mean((pred - truey)^2))

  return(list("D" = D, "rmse" = rmse))
}

# ============================================================
# ALMV: Active Learning MacKay Variance
# ============================================================
ALMV <- function(D0, y0, Nnew) {
  D <- D0
  y <- y0
  n <- nrow(D)
  rmse <- numeric(Nnew + 1)

  for (i in 1:Nnew) {
    CAND0 <- generate_sobol_set(100 * p, p, seed = sample(10^6, 1))
    if (i == 1) theta <- rep(1, p)

    kernel <- Gaussian.Kernel(theta)
    obj1 <- Fit.Kriging(D, y, model = "OK", kernel = kernel)
    obj2 <- Fit.Kriging(D, y, kernel.parameters = list(type = "Gaussian"))
    if (obj1$get_nllh() < obj2$get_nllh()) obj <- obj1 else obj <- obj2
    theta <- obj$get_lengthscale()

    pred <- Predict.Kriging(obj, test)$mean
    rmse[i] <- sqrt(mean((pred - truey)^2))

    kernel <- Gaussian.Kernel(theta)
    R <- Evaluate.Kernel(kernel, D)
    U <- chol(R + 10^(-6) * diag(n))

    r <- function(x, theta) {
      if (p > 1) {
        A <- D - rep(1, n) %*% t(x)
        g <- exp(-0.5 * apply(abs(A %*% diag(1 / theta))^2, 1, sum))
      } else {
        g <- exp(-0.5 * abs((D - x) / theta)^2)
      }
      return(g)
    }
    basis <- function(h, theta) exp(-0.5 * sum((h / theta)^2))

    opt <- function(v, theta) {
      b <- forwardsolve(t(U), r(v, theta))
      numer <- function(x, v, theta) {
        a <- forwardsolve(t(U), r(x, theta))
        val <- 1 - sum(a^2) - (basis(x - v, theta) - sum(a * b))^2 / max(1 - sum(b^2), 10^(-6))
        return(val)
      }
      val <- max(apply(CAND0, 1, numer, v = v, theta = theta)) / max(1 - sum(b^2), 10^(-6))
      return(val)
    }

    d <- c(knn.dist(D, k = 1)) / 2
    dis <- as.matrix(pdist(D, CAND0))
    N <- dim(CAND0)[1]
    ind <- NULL
    for (j in 1:n) ind <- c(ind, (1:N)[dis[j, ] < d[j]])
    if (length(ind) > 0) CAND <- CAND0[-ind, ] else CAND <- CAND0

    xnew <- CAND[which.min(apply(CAND, 1, opt, theta = theta)), ]
    D <- rbind(D, xnew)
    y <- c(y, f(xnew))
    n <- n + 1
  }

  kernel <- Gaussian.Kernel(theta)
  obj1 <- Fit.Kriging(D, y, model = "OK", kernel = kernel)
  obj2 <- Fit.Kriging(D, y, kernel.parameters = list(type = "Gaussian"))
  if (obj1$get_nllh() < obj2$get_nllh()) obj <- obj1 else obj <- obj2
  pred <- Predict.Kriging(obj, test)$mean
  rmse[Nnew + 1] <- sqrt(mean((pred - truey)^2))

  return(list("D" = D, "rmse" = rmse))
}

# ============================================================
# ALM: Active Learning MacKay
# ============================================================
ALM <- function(D0, y0, Nnew) {
  D <- D0
  y <- y0
  n <- nrow(D)
  rmse <- numeric(Nnew + 1)

  for (i in 1:Nnew) {
    CAND0 <- generate_sobol_set(1000 * p, p, seed = sample(10^6, 1))
    if (i == 1) theta <- rep(1, p)

    kernel <- Gaussian.Kernel(theta)
    obj1 <- Fit.Kriging(D, y, model = "OK", kernel = kernel)
    obj2 <- Fit.Kriging(D, y, kernel.parameters = list(type = "Gaussian"))
    if (obj1$get_nllh() < obj2$get_nllh()) obj <- obj1 else obj <- obj2
    theta <- obj$get_lengthscale()

    pred <- Predict.Kriging(obj, test)$mean
    rmse[i] <- sqrt(mean((pred - truey)^2))

    kernel <- Gaussian.Kernel(theta)
    R <- Evaluate.Kernel(kernel, D)
    Rinv <- solve(R + 10^(-6) * diag(n))

    r <- function(x, theta) {
      if (p > 1) {
        A <- D - rep(1, n) %*% t(x)
        g <- exp(-0.5 * apply(abs(A %*% diag(1 / theta))^2, 1, sum))
      } else {
        g <- exp(-0.5 * abs((D - x) / theta)^2)
      }
      return(g)
    }

    opt <- function(v, theta) {
      a <- r(v, theta)
      b <- c(Rinv %*% a)
      val <- 1 - sum(a * b)
      return(val)
    }

    d <- c(knn.dist(D, k = 1)) / 2
    dis <- as.matrix(pdist(D, CAND0))
    N <- dim(CAND0)[1]
    ind <- NULL
    for (j in 1:n) ind <- c(ind, (1:N)[dis[j, ] < d[j]])
    if (length(ind) > 0) CAND <- CAND0[-ind, ] else CAND <- CAND0

    xnew <- CAND[which.max(apply(CAND, 1, opt, theta = theta)), ]
    D <- rbind(D, xnew)
    y <- c(y, f(xnew))
    n <- n + 1
  }

  kernel <- Gaussian.Kernel(theta)
  obj1 <- Fit.Kriging(D, y, model = "OK", kernel = kernel)
  obj2 <- Fit.Kriging(D, y, kernel.parameters = list(type = "Gaussian"))
  if (obj1$get_nllh() < obj2$get_nllh()) obj <- obj1 else obj <- obj2
  pred <- Predict.Kriging(obj, test)$mean
  rmse[Nnew + 1] <- sqrt(mean((pred - truey)^2))

  return(list("D" = D, "rmse" = rmse))
}

# Run selected method
if (method == "ALC") {
  result <- ALC(D0, y0, n_new)
} else if (method == "ALMV") {
  result <- ALMV(D0, y0, n_new)
} else if (method == "ALM") {
  result <- ALM(D0, y0, n_new)
}

# Save results
write.csv(as.data.frame(result$D), file.path(out_dir, "design.csv"), row.names = FALSE)
write.csv(data.frame(rmse = result$rmse), file.path(out_dir, "rmse.csv"), row.names = FALSE)
write.csv(data.frame(n0 = n0), file.path(out_dir, "params_out.csv"), row.names = FALSE)
