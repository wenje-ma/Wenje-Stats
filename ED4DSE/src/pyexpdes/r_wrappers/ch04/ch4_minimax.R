args <- commandArgs(trailingOnly = TRUE)
in_dir <- args[1]
out_dir <- args[2]

suppressMessages(library(minimaxdesign))

params <- read.csv(file.path(in_dir, "params.csv"))
n <- as.integer(params$n[1])
p <- as.integer(params$p[1])
q <- as.integer(params$q[1])
seed <- as.integer(params$seed[1])

set.seed(seed)

DmM <- matrix(c(.5,.5,.5,0,.5,1,1/3-1/12*sqrt(7),.25,1/3-1/12*sqrt(7),.75,
                2/3+1/12*sqrt(7),.25,2/3+1/12*sqrt(7),.75), nrow=7, ncol=2, byrow=TRUE)

D_opt <- minimax(N=n, p=p, q=q, ini=DmM)

write.csv(as.data.frame(D_opt), file.path(out_dir, "minimax_design.csv"), row.names = FALSE)
write.csv(as.data.frame(DmM), file.path(out_dir, "theoretical_design.csv"), row.names = FALSE)
