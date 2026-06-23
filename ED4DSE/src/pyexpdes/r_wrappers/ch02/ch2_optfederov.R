args <- commandArgs(trailingOnly = TRUE)
in_dir <- args[1]
out_dir <- args[2]
n <- as.integer(args[3])

suppressMessages(library(AlgDesign))

# Candidate set in [-1, 1]
cand <- data.frame(x = seq(-1, 1, length = 301))

# Match the book script: polynomial up to degree 9 (intercept + x + ... + x^9)
form <- ~ 1 + x + I(x^2) + I(x^3) + I(x^4) + I(x^5) + I(x^6) + I(x^7) + I(x^8) + I(x^9)
a <- optFederov(form, nTrials = n, data = cand, approximate = FALSE)

out <- data.frame(x = a$design[, 1])
write.csv(out, file.path(out_dir, "design.csv"), row.names = FALSE)


