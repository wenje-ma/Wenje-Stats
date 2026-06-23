args <- commandArgs(trailingOnly = TRUE)
in_dir <- args[1]
out_dir <- args[2]

suppressMessages(library(SFDesign))
suppressMessages(library(fitdistrplus))

# Match the book script chunk (pair-wise distance distributions)
p <- 6
n <- 64
set.seed(1)
D0 <- maximin.optim(maximinLHD(n, p)$design, find.best.ini=TRUE)$design
D1 <- maxpro.optim(maxproLHD(n, p)$design)$design

d0 <- c(dist(D0))
d1 <- c(dist(D1))

# Fit Beta to scaled distances: d1/sqrt(p)
a <- fitdist(d1/sqrt(p), "beta")
est <- a$estimate

write.csv(data.frame(d0 = d0), file.path(out_dir, "dist_maximin.csv"), row.names = FALSE)
write.csv(data.frame(d1 = d1), file.path(out_dir, "dist_maxpro.csv"), row.names = FALSE)
write.csv(data.frame(a = est[1], b = est[2], p = p), file.path(out_dir, "beta_fit.csv"), row.names = FALSE)


