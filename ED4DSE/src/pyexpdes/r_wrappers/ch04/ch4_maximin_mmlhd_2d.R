args <- commandArgs(trailingOnly = TRUE)
in_dir <- args[1]
out_dir <- args[2]

suppressMessages(library(SFDesign))

# -----------------------------
# Maximin design (n=7, p=2)
# -----------------------------
p <- 2
n <- 7
set.seed(1)
D_mm <- maximinLHD(n, p)$design
D_mm <- maximin.optim(D_mm, sa=TRUE, find.best.ini=TRUE)$design
r <- min(dist(D_mm)) / 2
write.csv(as.data.frame(D_mm), file.path(out_dir, "maximin_7_2.csv"), row.names = FALSE)
write.csv(data.frame(r = r), file.path(out_dir, "maximin_7_2_radius.csv"), row.names = FALSE)

# -----------------------------
# Sequential maximin (n=7 then n=13), candidates = 5-level full factorial
# -----------------------------
set.seed(1)
D0 <- matrix(c(.5,.5), nrow=1, ncol=p)
CAND <- full.factorial(p=2, level=5)
D7 <- maximin.augment(n=7, p=2, D.ini=D0, candidate=CAND)
D13 <- maximin.augment(n=13, p=2, D.ini=D7, candidate=CAND)
D13_opt <- maximin.optim(D13, sa=TRUE, find.best.ini=TRUE)$design

write.csv(as.data.frame(D7), file.path(out_dir, "seq_maximin_7_2.csv"), row.names = FALSE)
write.csv(as.data.frame(D13), file.path(out_dir, "seq_maximin_13_2.csv"), row.names = FALSE)
write.csv(as.data.frame(D13_opt), file.path(out_dir, "seq_maximin_13_2_optimized.csv"), row.names = FALSE)

# -----------------------------
# MmLHD designs (n=4,7,20; p=2)
# -----------------------------
n <- 4
D_mmlhd_4 <- maximinLHD(n, p)$design
write.csv(as.data.frame(D_mmlhd_4), file.path(out_dir, "mmlhd_4_2.csv"), row.names = FALSE)

n <- 7
D_mmlhd_7 <- maximinLHD(n, p)$design
write.csv(as.data.frame(D_mmlhd_7), file.path(out_dir, "mmlhd_7_2.csv"), row.names = FALSE)

n <- 20
set.seed(1)
D_mmlhd_20 <- maximinLHD(n, p)$design
write.csv(as.data.frame(D_mmlhd_20), file.path(out_dir, "mmlhd_20_2.csv"), row.names = FALSE)

# Chebyshev transform (D1 = (1+cos(pi*D))/2)
D_cheb <- (1 + cos(pi * D_mmlhd_20)) / 2
write.csv(as.data.frame(D_cheb), file.path(out_dir, "chebyshev_mmlhd_20_2.csv"), row.names = FALSE)


