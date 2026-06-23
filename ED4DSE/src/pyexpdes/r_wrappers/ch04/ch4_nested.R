args <- commandArgs(trailingOnly = TRUE)
in_dir <- args[1]
out_dir <- args[2]

suppressMessages(library(MaxPro))
suppressMessages(library(SFDesign))

n1 <- 100
n2 <- 10
p <- 2

set.seed(1)
D1 <- maxpro.optim(maxproLHD(n1,p)$design)$design
D2 <- maxpro.remove(D1, n.remove = n1-n2, delta = 1/n1^2)

write.csv(as.data.frame(D1), file.path(out_dir, "nested_full.csv"), row.names = FALSE)
write.csv(as.data.frame(D2), file.path(out_dir, "nested_subset.csv"), row.names = FALSE)


