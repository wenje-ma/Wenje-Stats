args <- commandArgs(trailingOnly = TRUE)
in_dir <- args[1]
out_dir <- args[2]
num_new <- as.integer(args[3])

suppressMessages(library(maximin))

cand <- read.csv(file.path(in_dir, "candidates.csv"))
orig <- read.csv(file.path(in_dir, "initial.csv"))

Xcand <- as.matrix(cand)
Xorig <- as.matrix(orig)

inds <- maximin.cand(num_new, Xcand = Xcand, Xorig = Xorig)$inds
pts <- Xcand[inds,]
pts <- rbind(Xorig, pts)

write.csv(as.data.frame(pts), file.path(out_dir, "points.csv"), row.names = FALSE)


