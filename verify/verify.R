# The inference behind "The cost" section, done independently in base R.
#
# The README claims two things that are not single numbers. First, that at 128
# NFE on 8 gaussians the reflowed model and its teacher are level rather than
# one winning, quoting the per seed ranges. Second, that the reflowed model at
# 1 NFE is as good as itself at 128 NFE on moons. Both are claims about spread
# and about a difference being small, and nothing in the repo tested either:
# the Python only ever took medians. This re-derives the quoted ranges from
# results/nfe-quality.csv and runs an exact paired sign flip permutation test
# on the three seeds, which is the strongest test three paired seeds support.
#
#   Rscript verify/verify.R <repo-root>

args <- commandArgs(trailingOnly = TRUE)
root <- if (length(args) > 0) args[1] else "."
nfe <- read.csv(file.path(root, "results", "nfe-quality.csv"), stringsAsFactors = FALSE)
# Wrapping is not meaningful here, so the prose is flattened before matching.
readme <- gsub("[[:space:]]+", " ",
               paste(readLines(file.path(root, "README.md"), warn = FALSE), collapse = " "))

fails <- character(0)
note <- function(...) cat(" ", ..., "\n")

pick <- function(ds, model, n) {
  rows <- nfe[nfe$dataset == ds & nfe$model == model &
              nfe$sampler == "euler" & nfe$nfe == n, ]
  rows <- rows[order(rows$seed), ]
  rows$sliced_w2
}

# Exact permutation test over the 2^n sign flips of the paired differences.
# With three seeds the smallest attainable two sided p value is 0.25, so this
# can only ever say "not distinguishable", never "distinguishable". That is a
# property of the experiment, not of the test.
sign_flip_p <- function(d) {
  n <- length(d)
  signs <- as.matrix(expand.grid(rep(list(c(-1, 1)), n)))
  stats <- abs(signs %*% d / n)
  mean(stats >= abs(mean(d)) - 1e-15)
}

cat("R, the cost section at 128 NFE on 8 gaussians\n")
reflow <- pick("8gaussians", "2-rectified", 128)
teacher <- pick("8gaussians", "1-rectified", 128)
if (length(reflow) != 3 || length(teacher) != 3) {
  fails <- c(fails, sprintf("expected 3 seeds each, got %d and %d",
                            length(reflow), length(teacher)))
} else {
  quoted <- regmatches(readme, regexec(
    "seeds run ([0-9.]+) to ([0-9.]+) for it and ([0-9.]+) to ([0-9.]+) for its teacher",
    readme))[[1]]
  if (length(quoted) != 5) {
    fails <- c(fails, "could not find the quoted seed ranges in README.md")
  } else {
    want <- as.numeric(quoted[2:5])
    got <- c(min(reflow), max(reflow), min(teacher), max(teacher))
    labels <- c("reflow min", "reflow max", "teacher min", "teacher max")
    for (i in seq_along(want)) {
      shown <- sprintf("%.3f", got[i])
      note(sprintf("%-12s README %s   results/ %s", labels[i], quoted[i + 1], shown))
      if (shown != quoted[i + 1]) {
        fails <- c(fails, sprintf("%s: README says %s, results/ gives %s",
                                  labels[i], quoted[i + 1], shown))
      }
    }
  }
  d <- reflow - teacher
  p <- sign_flip_p(d)
  spread <- max(c(reflow, teacher)) - min(c(reflow, teacher))
  note(sprintf("paired differences per seed: %s", paste(sprintf("%+.5f", d), collapse = " ")))
  note(sprintf("mean paired difference %+.5f, seed to seed spread %.5f, ratio %.3f",
               mean(d), spread, abs(mean(d)) / spread))
  note(sprintf("exact sign flip permutation test on 3 pairs: p = %.3f", p))
  if (p <= 0.05) {
    fails <- c(fails, sprintf(
      "README calls the two level, but the paired test separates them at p = %.3f", p))
  }
}

cat("R, one step against 128 steps for the reflowed model on moons\n")
one <- pick("moons", "2-rectified", 1)
full <- pick("moons", "2-rectified", 128)
if (length(one) != 3 || length(full) != 3) {
  fails <- c(fails, "expected 3 seeds for the moons 1 against 128 comparison")
} else {
  d <- one - full
  p <- sign_flip_p(d)
  note(sprintf("paired differences per seed: %s", paste(sprintf("%+.5f", d), collapse = " ")))
  note(sprintf("median at 1 NFE %.5f, at 128 NFE %.5f, exact sign flip p = %.3f",
               median(one), median(full), p))
  if (p <= 0.05) {
    fails <- c(fails, sprintf(
      "README calls 1 NFE and 128 NFE indistinguishable here, but p = %.3f", p))
  }
}

if (length(fails) > 0) {
  cat("DISAGREEMENT:\n")
  for (f in fails) cat("  -", f, "\n")
  quit(status = 1)
}
quit(status = 0)
