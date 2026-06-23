args <- commandArgs(trailingOnly = TRUE)
in_dir <- args[1]
out_dir <- args[2]

suppressMessages(library(mined))
suppressMessages(library(MaxPro))
suppressMessages(library(SFDesign))

# Match "Bayesian computation" section up to the two image plots (initial vs MED).
logf <- function(para)
{
  l1 <- -40
  u1 <- 40
  l2 <- -25
  u2 <- 10
  x1 <- l1 + (u1 - l1) * para[1]
  x2 <- l2 + (u2 - l2) * para[2]
  val <- -.5 * (x1 ^2 / 100 + (x2+ .03 * x1^2 -3)^2)
  return(val)
}

N.plot <- 300
p1 <- seq(0, 1, length.out = N.plot)
p2 <- seq(0, 1, length.out = N.plot)
fc <- matrix(0.0, N.plot, N.plot)
for(i in 1:N.plot){
  for(j in 1:N.plot){
    fc[i, j] <- exp(logf(c(p1[i], p2[j])))
  }
}

set.seed(8)
p <- 2
n <- 20
ini <- MaxPro(MaxProLHD(n,p)$Design)$Design
res <- mined(ini, logf, K_iter = 5)
D <- res$points
cand <- res$cand

# Export grids in long form for CSV handoff
grid <- expand.grid(p1=p1, p2=p2)
grid$fc <- as.vector(fc)
write.csv(grid, file.path(out_dir, "banana_grid.csv"), row.names = FALSE)
write.csv(as.data.frame(ini), file.path(out_dir, "banana_ini.csv"), row.names = FALSE)
write.csv(as.data.frame(D), file.path(out_dir, "banana_med.csv"), row.names = FALSE)
write.csv(as.data.frame(cand), file.path(out_dir, "banana_cand.csv"), row.names = FALSE)


