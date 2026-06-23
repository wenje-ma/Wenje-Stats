args <- commandArgs(trailingOnly = TRUE)
in_dir <- args[1]
out_dir <- args[2]

suppressMessages(library(SLHD))
suppressMessages(library(MaxPro))
suppressMessages(library(SFDesign))

p <- 2
n <- 9

# Random "bad" LHD as in script
set.seed(1)
D0 <- cbind(rep(c(1,2,3), rep(3,3)), 1:9, sample(1:9))
D0[,2:3] <- (D0[,2:3] - .5) / 9

# SLHD
a <- maximinSLHD(3,3,2)
D1 <- a$StandDesign

# MaxProQQ
set.seed(3)
ini <- cbind(MaxPro(MaxProLHD(n,p)$Design)$Design, rep(1:3, c(3,3,3)))
a2 <- MaxProQQ(InitialDesign = ini, p_nom = 1)
D2 <- a2$Design

write.csv(as.data.frame(D0), file.path(out_dir, "random_lhd.csv"), row.names = FALSE)
write.csv(as.data.frame(D1), file.path(out_dir, "slhd.csv"), row.names = FALSE)
write.csv(as.data.frame(D2), file.path(out_dir, "maxproqq.csv"), row.names = FALSE)


