# ch8_cast_fatigue.R
# Cast fatigue experiment analysis using HiGarrote
# Usage: Rscript ch8_cast_fatigue.R <in_dir> <out_dir>

args <- commandArgs(trailingOnly = TRUE)
in_dir <- args[1]
out_dir <- args[2]

library(HiGarrote)

# Read cast fatigue data
cast <- read.csv(file.path(in_dir, "cast.csv"))

# Design matrix (factors F1-G) and response
X <- as.matrix(cast[, 1:7])
y <- cast$y

# Run HiGarrote for variable selection
hg_result <- HiGarrote(X, y)

# HiGarrote returns $nng_estimate with named coefficients
eff_vals <- hg_result$nng_estimate
eff_names <- names(eff_vals)
if (is.null(eff_names)) {
  eff_names <- paste0("V", seq_along(eff_vals))
}

write.csv(
  data.frame(var_name = eff_names, coefficient = as.numeric(eff_vals)),
  file.path(out_dir, "higarrote_result.csv"),
  row.names = FALSE
)

# Fit linear models for comparison
# Full model
full_model <- lm(y ~ ., data = cast)
full_summary <- summary(full_model)

# Save full model summary
write.csv(
  data.frame(
    variable = rownames(full_summary$coefficients),
    estimate = full_summary$coefficients[, 1],
    std_error = full_summary$coefficients[, 2],
    t_value = full_summary$coefficients[, 3],
    p_value = full_summary$coefficients[, 4]
  ),
  file.path(out_dir, "full_model.csv"),
  row.names = FALSE
)

# Best model based on HiGarrote (F1, D, G, F1:G, D:G)
# Note: The R code shows: summary(lm(y~F1*G+D+D:G,data=cast))
best_model <- lm(y ~ F1 * G + D + D:G, data = cast)
best_summary <- summary(best_model)

write.csv(
  data.frame(
    variable = rownames(best_summary$coefficients),
    estimate = best_summary$coefficients[, 1],
    std_error = best_summary$coefficients[, 2],
    t_value = best_summary$coefficients[, 3],
    p_value = best_summary$coefficients[, 4]
  ),
  file.path(out_dir, "best_model.csv"),
  row.names = FALSE
)

# Save model comparison metrics
write.csv(
  data.frame(
    model = c("full", "best"),
    r_squared = c(full_summary$r.squared, best_summary$r.squared),
    adj_r_squared = c(full_summary$adj.r.squared, best_summary$adj.r.squared),
    residual_se = c(full_summary$sigma, best_summary$sigma)
  ),
  file.path(out_dir, "model_comparison.csv"),
  row.names = FALSE
)
