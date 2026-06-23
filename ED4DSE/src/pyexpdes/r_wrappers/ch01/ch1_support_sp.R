args <- commandArgs(trailingOnly = TRUE)
in_dir <- args[1]
out_dir <- args[2]
n <- as.integer(args[3])

suppressMessages(library(support))

dist_samples <- read.csv(file.path(in_dir, "dist_samples.csv"))
dist_mat <- as.matrix(dist_samples)

a <- sp(n, 1, dist.samp = dist_mat)
out <- data.frame(D = a$sp)
write.csv(out, file.path(out_dir, "design.csv"), row.names = FALSE)


