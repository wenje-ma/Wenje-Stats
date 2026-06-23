# ch8_orthogonal_array.R
# L18 orthogonal array analysis with MaxPro and Energy distance
# Usage: Rscript ch8_orthogonal_array.R <in_dir> <out_dir>

args <- commandArgs(trailingOnly = TRUE)
in_dir <- args[1]
out_dir <- args[2]

library(DoE.base)
library(SFDesign)
library(twinning)

# Read parameters
params <- read.csv(file.path(in_dir, "params.csv"))
n_sim <- params$n_sim[1]
seed <- params$seed[1]

# Full factorial for 4 factors, 3 levels
full <- full.factorial(4, 3)

# L9 with two replicates (D1)
D <- undesign(oa.design(nfactors = 4, nlevels = 3, randomize = FALSE))
D <- data.matrix(D)
D1 <- rbind(D, D)
D1 <- (D1 - 1) / 2
mp1 <- maxpro.crit(D1, delta = 1/3^2) / 3^2
en1 <- energy(full, D1)

# L18 design (D2) - columns 2:5
D <- undesign(oa.design(ID = L18, randomize = FALSE))
D <- data.matrix(D)
D2 <- D[, 2:5]
D2 <- (D2 - 1) / 2
mp2 <- maxpro.crit(D2, delta = 1/3^2) / 3^2
en2 <- energy(full, D2)

# L18 design (D3) - columns 3:6
D <- undesign(oa.design(ID = L18, randomize = FALSE))
D <- data.matrix(D)
D3 <- D[, 3:6]
D3 <- (D3 - 1) / 2
mp3 <- maxpro.crit(D3, delta = 1/3^2) / 3^2
en3 <- energy(full, D3)

# Generate all 27 L18 variants
A <- array(data = rep(0, 18 * 4 * 27), dim = c(18, 4, 27))
D <- undesign(oa.design(ID = L18, randomize = FALSE))
D <- data.matrix(D)
D <- D[, 3:6]
D1_col <- (D[, 1] - 1) / 2

A1 <- lapply(1:3, function(i) {
  D2_col <- D[, 2]
  if (i == 2) {
    D2_col <- ifelse(D2_col == 1, 1, ifelse(D2_col == 2, 3, 2))
  } else if (i == 3) {
    D2_col <- ifelse(D2_col == 1, 2, ifelse(D2_col == 2, 1, 3))
  }
  (D2_col - 1) / 2
})

A2 <- lapply(1:3, function(i) {
  D3_col <- D[, 3]
  if (i == 2) {
    D3_col <- ifelse(D3_col == 1, 1, ifelse(D3_col == 2, 3, 2))
  } else if (i == 3) {
    D3_col <- ifelse(D3_col == 1, 2, ifelse(D3_col == 2, 1, 3))
  }
  (D3_col - 1) / 2
})

A3 <- lapply(1:3, function(i) {
  D4_col <- D[, 4]
  if (i == 2) {
    D4_col <- ifelse(D4_col == 1, 1, ifelse(D4_col == 2, 3, 2))
  } else if (i == 3) {
    D4_col <- ifelse(D4_col == 1, 2, ifelse(D4_col == 2, 1, 3))
  }
  (D4_col - 1) / 2
})

a <- expand.grid(c(1, 2, 3), c(1, 2, 3), c(1, 2, 3))
for (i in 1:27) {
  idx <- a[i, ]
  A[, , i] <- cbind(D1_col, A1[[idx$Var1]], A2[[idx$Var2]], A3[[idx$Var3]])
}

val_mp <- val_en <- numeric(27)
for (k in 1:27) {
  val_mp[k] <- maxpro.crit(A[, , k], delta = 1/3^2) / 3^2
  val_en[k] <- energy(full, A[, , k])
}

# Monte Carlo simulation
set.seed(seed)
val_mp_sim <- val_en_sim <- numeric(n_sim)
for (i in 1:n_sim) {
  set.seed(i)
  D_samp <- full[sample(1:81, 18), ]
  val_mp_sim[i] <- maxpro.crit(D_samp, delta = 1/3^2) / 3^2
  val_en_sim[i] <- energy(full, D_samp)
}

# Save simulation results
sim_results <- data.frame(
  maxpro = val_mp_sim,
  energy = val_en_sim
)
write.csv(sim_results, file.path(out_dir, "simulation_results.csv"), row.names = FALSE)

# Save L18 variant metrics
variant_metrics <- data.frame(
  variant = 1:27,
  maxpro = val_mp,
  energy = val_en
)
write.csv(variant_metrics, file.path(out_dir, "variant_metrics.csv"), row.names = FALSE)

# Save special design metrics
design_metrics <- data.frame(
  design = c("D1", "D2", "D3"),
  maxpro = c(mp1, mp2, mp3),
  energy = c(en1, en2, en3)
)
write.csv(design_metrics, file.path(out_dir, "design_metrics.csv"), row.names = FALSE)
