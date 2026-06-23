args <- commandArgs(trailingOnly = TRUE)
in_dir <- args[1]
out_dir <- args[2]

suppressMessages(library(SFDesign))
suppressMessages(library(MaxPro))

# Match the book script chunk (p=5, n=50)
p <- 5
n <- 50
set.seed(2)
D_maxpro <- maxpro.optim(maxproLHD(n,p)$design)$design
D_mmlhd <- maximinLHD(n,p)$design

write.csv(as.data.frame(D_maxpro), file.path(out_dir, "maxpro_50_5.csv"), row.names = FALSE)
write.csv(as.data.frame(D_mmlhd), file.path(out_dir, "mmlhd_50_5.csv"), row.names = FALSE)


