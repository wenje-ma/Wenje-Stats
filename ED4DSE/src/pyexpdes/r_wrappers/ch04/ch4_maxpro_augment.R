args <- commandArgs(trailingOnly = TRUE)
in_dir <- args[1]
out_dir <- args[2]
new_n <- as.integer(args[3])

suppressMessages(library(MaxPro))

exist <- read.csv(file.path(in_dir, "exist.csv"))
cand <- read.csv(file.path(in_dir, "cand.csv"))

ExistDesign <- as.matrix(exist)
CandDesign <- as.matrix(cand)

a <- MaxProAugment(ExistDesign = ExistDesign, CandDesign = CandDesign, nNew = new_n)$Design
write.csv(as.data.frame(a), file.path(out_dir, "design.csv"), row.names = FALSE)


