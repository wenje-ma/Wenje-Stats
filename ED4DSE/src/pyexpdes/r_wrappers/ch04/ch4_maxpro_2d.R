args <- commandArgs(trailingOnly = TRUE)
in_dir <- args[1]
out_dir <- args[2]

suppressMessages(library(MaxPro))
suppressMessages(library(SFDesign))

# Match the book script chunk (see ch4.R)
p <- 2

# MaxPro(7,2)
n <- 7
a <- MaxPro(MaxProLHD(n, p)$Design)
D7 <- a$Design
write.csv(as.data.frame(D7), file.path(out_dir, "maxpro_7_2.csv"), row.names = FALSE)

# MaxPro(20,2) - script uses set.seed(10) and maxpro.optim(..., iteration=100)
n <- 20
set.seed(10)
D <- maxpro.optim(maxproLHD(n, p)$design, iteration = 100)$design
write.csv(as.data.frame(D), file.path(out_dir, "maxpro_20_2.csv"), row.names = FALSE)


