args <- commandArgs(trailingOnly = TRUE)
in_dir <- args[1]
out_dir <- args[2]

suppressMessages(library(SFDesign))
suppressMessages(library(MaxPro))

# Match ch4.R branching design section.
best_seed <- NA
best_val <- Inf
best_ind1 <- NA
best_ind2 <- NA

for(k in 1:20){
  set.seed(k)
  shared_nested.ini <- maxpro.optim(maxproLHD(30,6+2)$design)$design
  branch <- rep(c(0,1),c(15,15))
  ini <- cbind(shared_nested.ini, branch)
  D0 <- MaxProQQ(ini, p_nom=1)$Design

  na_nested <- matrix(NA, nrow=15, ncol=2)
  val <- matrix(Inf, nrow=8, ncol=8)
  D1 <- D0[1:15,-9]
  D2 <- D0[16:30,-9]

  for(i in 1:7){
    for(j in (i+1):8){
      B <- D2
      B[,c(i,j)] <- na_nested
      A <- D1
      A[,c(i,j)] <- (apply(D1[,c(i,j)],2,rank)-.5)/15
      A[,c(i,j)] <- maxpro.optim(A[,c(i,j)])$design
      D <- cbind(rbind(A,B), branch)
      val[i,j] <- MaxProMeasure(D[,-c(i,j)]) + MaxProMeasure(D[1:15,-9]) + MaxProMeasure(D[16:30,-c(9,i,j)])
    }
  }

  vmin <- min(val)
  if(vmin < best_val){
    best_val <- vmin
    best_seed <- k
  }
}

# Construct optimal branching design using best seed (ch4.R uses opt.seed=12)
opt.seed <- best_seed
set.seed(opt.seed)
shared_nested.ini <- maxpro.optim(maxproLHD(30,6+2)$design)$design
branch <- rep(c(0,1),c(15,15))
ini <- cbind(shared_nested.ini, branch)
D0 <- MaxProQQ(ini, p_nom=1)$Design

na_nested <- matrix(NA, nrow=15, ncol=2)
val <- matrix(Inf, nrow=8, ncol=8)
D1 <- D0[1:15,-9]
D2 <- D0[16:30,-9]
for(i in 1:7){
  for(j in (i+1):8){
    B <- D2
    B[,c(i,j)] <- na_nested
    A <- D1
    A[,c(i,j)] <- (apply(D1[,c(i,j)],2,rank)-.5)/15
    A[,c(i,j)] <- maxpro.optim(A[,c(i,j)])$design
    D <- cbind(rbind(A,B), branch)
    val[i,j] <- MaxProMeasure(D[1:15,-9]) + MaxProMeasure(D[16:30,-c(9,i,j)])
  }
}
ind <- which(val == min(val), arr.ind = TRUE)[1,]

B <- D2
B[,ind] <- na_nested
A <- D1
A[,ind] <- (apply(D1[,ind],2,rank)-.5)/15
A[,ind] <- maxpro.optim(A[,ind])$design
D <- cbind(rbind(A,B), branch)

# Re-order columns like ch4.R
temp <- D
temp[,1] <- D[,9]
temp[,2:3] <- D[,ind]
temp[,4:9] <- D[,-c(9,ind)]
D <- temp
colnames(D) <- c("branch", "nested1","nested2", paste("shared",sep="", 1:6))

write.csv(as.data.frame(D), file.path(out_dir, "branching_design.csv"), row.names = FALSE)
write.csv(data.frame(opt_seed=opt.seed, best_val=best_val, ind1=ind[1], ind2=ind[2]), file.path(out_dir, "branching_meta.csv"), row.names = FALSE)


