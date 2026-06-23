args <- commandArgs(trailingOnly = TRUE)
in_dir <- args[1]
out_dir <- args[2]

suppressMessages(library(rkriging))

df <- read.csv(file.path(in_dir, "data.csv"))

# The Python side may provide a single CSV that contains:
# - observation rows: x,y present; test is NA
# - prediction grid rows: test present; x,y are NA
obs_idx <- which(!is.na(df$x) & !is.na(df$y))
test_idx <- which(!is.na(df$test))

x <- df$x[obs_idx]
y <- df$y[obs_idx]
test <- df$test[test_idx]

a.gp <- Fit.Kriging(x, y,
                    interpolation = FALSE,
                    fit = TRUE,
                    model = "OK",
                    kernel.parameters = list(type = "Gaussian"))
pred <- Predict.Kriging(a.gp, test)

out <- data.frame(test = test, mean = pred$mean, sd = pred$sd)
write.csv(out, file.path(out_dir, "pred.csv"), row.names = FALSE)


