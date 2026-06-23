args <- commandArgs(trailingOnly = TRUE)
in_dir <- args[1]
out_dir <- args[2]

suppressMessages(library(MaxPro))
suppressMessages(library(SFDesign))
suppressMessages(library(rkriging))

# Branin definition exactly as in ch4.R
branin <- function(x){
  x1 <- x[1] * 15 - 5
  x2 <- x[2] * 15
  val1 <- (x2 - 5.1/(4 * pi^2) * (x1^2) + 5/pi * x1 - 6)^2
  val2 <- 10 * (1 - 1/(8 * pi)) * cos(x1) + 10
  return(val1 + val2)
}

p <- 2
n <- 10
set.seed(123)

D <- MaxPro(MaxProLHD(n,p)$Design)$Design
CAND <- CandPoints(N=10000,2)
A_all <- MaxProAugment(ExistDesign = D, CandDesign = CAND, nNew = 10)$Design
# Keep only the new points, as in the script
A <- A_all[(n+1):20,]

y <- apply(D, 1, branin)
a <- Fit.Kriging(D, y, kernel.parameters = list(type="Gaussian"))

trueA <- apply(A, 1, branin)
predA <- Predict.Kriging(a, A)$mean
rmse0 <- sqrt(mean((predA-trueA)^2))

rmse <- numeric(100)
for(i in 1:100){
  u <- matrix(runif(p*10), nrow=10, ncol=p)
  pred <- Predict.Kriging(a, u)$mean
  tru <- apply(u, 1, branin)
  rmse[i] <- sqrt(mean((pred-tru)^2))
}

write.csv(as.data.frame(D), file.path(out_dir, "design_D.csv"), row.names = FALSE)
write.csv(as.data.frame(A), file.path(out_dir, "design_A.csv"), row.names = FALSE)
write.csv(data.frame(rmse = rmse), file.path(out_dir, "rmse.csv"), row.names = FALSE)
write.csv(data.frame(rmse0 = rmse0), file.path(out_dir, "rmse0.csv"), row.names = FALSE)


