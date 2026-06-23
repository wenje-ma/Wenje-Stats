args <- commandArgs(trailingOnly = TRUE)
in_dir <- args[1]
out_dir <- args[2]

suppressMessages(library(cubature))
suppressMessages(library(adaptMCMC))

# Banana target as in ch4.R (same logf)
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

nlogh <- function(para) -logf(para)
h <- function(para) exp(-nlogh(para))
target <- function(para) .5*logf(para)

N.plot <- 300
p1 <- seq(0, 1, length.out = N.plot)
p2 <- seq(0, 1, length.out = N.plot)
v <- cbind(p1, p2)

# Exact marginals by numerical integration (as in ch4.R)
denom <- adaptIntegrate(h, c(0,0), c(1,1))$int

exact <- matrix(0, nrow=N.plot, ncol=2)

h1 <- function(theta1){
  apply(cbind(theta1, theta2), 1, h)
}
val <- p2
for(i in 1:N.plot){
  theta2 <- p2[i]
  val[i] <- integrate(h1, 0, 1)$val
}
exact[,2] <- val/denom

h2 <- function(theta2){
  apply(cbind(theta1, theta2), 1, h)
}
val <- p1
for(i in 1:N.plot){
  theta1 <- p1[i]
  val[i] <- integrate(h2, 0, 1)$val
}
exact[,1] <- val/denom

# MCMC (as in ch4.R)
p <- 2
s <- (2.4/sqrt(2))^2 * diag(p)
set.seed(8)
out <- MCMC(logf, n=10000, init=rep(.5,p), scale=s, adapt=TRUE, acc.rate=.05)
theta <- out$samples

write.csv(data.frame(p=p1, exact=exact[,1]), file.path(out_dir, "exact_theta1.csv"), row.names = FALSE)
write.csv(data.frame(p=p2, exact=exact[,2]), file.path(out_dir, "exact_theta2.csv"), row.names = FALSE)
write.csv(as.data.frame(theta), file.path(out_dir, "mcmc_samples.csv"), row.names = FALSE)


