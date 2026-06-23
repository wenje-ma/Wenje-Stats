args <- commandArgs(trailingOnly = TRUE)
in_dir <- args[1]
out_dir <- args[2]
num <- as.integer(args[3])

suppressMessages(library(mined))

cand <- read.csv(file.path(in_dir, "candidates.csv"))
w <- read.csv(file.path(in_dir, "weights.csv"))

CAND <- as.matrix(cand)
lf <- log(as.numeric(w[,1]))

pts <- SelectMinED(candidates = CAND, candlf = lf, n = num, s = 1)$points
write.csv(as.data.frame(pts), file.path(out_dir, "points.csv"), row.names = FALSE)


