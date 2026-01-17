
rm(list=ls())

# Base path
base_path <- "/scratch/oldhome/mhaque/quantized-model-code-quality/statistical-tests/per-instance-value/pass1"
output_path <- "/scratch/oldhome/mhaque/quantized-model-code-quality/statistical-tests/R-scripts-v2"

# ============== LOAD PYTHON DATA ==============
py_0_5b_qlora <- read.csv(paste0(base_path, "/py/r-pass1_cg_qwen0_5_mt_qlora.csv"), header=TRUE)
py_0_5b_fft <- read.csv(paste0(base_path, "/py/r-pass1_cg_qwen0_5_mt_fft.csv"), header=TRUE)
py_1_5b_qlora <- read.csv(paste0(base_path, "/py/r-pass1_cg_qwen1_5_mt_qlora.csv"), header=TRUE)
py_1_5b_fft <- read.csv(paste0(base_path, "/py/r-pass1_cg_qwen1_5_mt_fft.csv"), header=TRUE)
py_3b_qlora <- read.csv(paste0(base_path, "/py/r-pass1_cg_qwen3_mt_qlora.csv"), header=TRUE)
py_3b_fft <- read.csv(paste0(base_path, "/py/r-pass1_cg_qwen3_mt_fft.csv"), header=TRUE)

# ============== LOAD JAVA DATA ==============
java_0_5b_qlora <- read.csv(paste0(base_path, "/java/r-pass1_cg_qwen0_5_mt_qlora.csv"), header=TRUE)
java_0_5b_fft <- read.csv(paste0(base_path, "/java/r-pass1_cg_qwen0_5_mt_fft.csv"), header=TRUE)
java_1_5b_qlora <- read.csv(paste0(base_path, "/java/r-pass1_cg_qwen1_5_mt_qlora.csv"), header=TRUE)
java_1_5b_fft <- read.csv(paste0(base_path, "/java/r-pass1_cg_qwen1_5_mt_fft.csv"), header=TRUE)
java_3b_qlora <- read.csv(paste0(base_path, "/java/r-pass1_cg_qwen3_mt_qlora.csv"), header=TRUE)
java_3b_fft <- read.csv(paste0(base_path, "/java/r-pass1_cg_qwen3_mt_fft.csv"), header=TRUE)

# ============== RENAME COLUMN (handle special character @) ==============
rename_col <- function(df) {
  colnames(df)[2] <- "pass1"
  return(df)
}

py_0_5b_qlora <- rename_col(py_0_5b_qlora); py_0_5b_fft <- rename_col(py_0_5b_fft)
py_1_5b_qlora <- rename_col(py_1_5b_qlora); py_1_5b_fft <- rename_col(py_1_5b_fft)
py_3b_qlora <- rename_col(py_3b_qlora); py_3b_fft <- rename_col(py_3b_fft)

java_0_5b_qlora <- rename_col(java_0_5b_qlora); java_0_5b_fft <- rename_col(java_0_5b_fft)
java_1_5b_qlora <- rename_col(java_1_5b_qlora); java_1_5b_fft <- rename_col(java_1_5b_fft)
java_3b_qlora <- rename_col(java_3b_qlora); java_3b_fft <- rename_col(java_3b_fft)

# ============== ENSURE SAME LENGTH ==============
align_data <- function(df1, df2) {
  min_len <- min(nrow(df1), nrow(df2))
  return(list(df1[1:min_len, ], df2[1:min_len, ]))
}

# Python
aligned <- align_data(py_0_5b_qlora, py_0_5b_fft); py_0_5b_qlora <- aligned[[1]]; py_0_5b_fft <- aligned[[2]]
aligned <- align_data(py_1_5b_qlora, py_1_5b_fft); py_1_5b_qlora <- aligned[[1]]; py_1_5b_fft <- aligned[[2]]
aligned <- align_data(py_3b_qlora, py_3b_fft); py_3b_qlora <- aligned[[1]]; py_3b_fft <- aligned[[2]]

# Java
aligned <- align_data(java_0_5b_qlora, java_0_5b_fft); java_0_5b_qlora <- aligned[[1]]; java_0_5b_fft <- aligned[[2]]
aligned <- align_data(java_1_5b_qlora, java_1_5b_fft); java_1_5b_qlora <- aligned[[1]]; java_1_5b_fft <- aligned[[2]]
aligned <- align_data(java_3b_qlora, java_3b_fft); java_3b_qlora <- aligned[[1]]; java_3b_fft <- aligned[[2]]

# ============== HELPER FUNCTIONS ==============

# Run McNemar's test (CORRECTED VERSION)
run_mcnemar <- function(col1_pass, col2_pass) {
  # Build contingency table
  # col1 = QLoRA, col2 = FFT
  yes_yes <- sum(col1_pass == 1 & col2_pass == 1)  # Both pass
  yes_no <- sum(col1_pass == 1 & col2_pass == 0)   # QLoRA pass, FFT fail
  no_yes <- sum(col1_pass == 0 & col2_pass == 1)   # QLoRA fail, FFT pass
  no_no <- sum(col1_pass == 0 & col2_pass == 0)    # Both fail
  
  # Create matrix for mcnemar.test (NO adjustment to cells)
  mat <- matrix(c(yes_yes, yes_no, no_yes, no_no), nrow=2, byrow=TRUE)
  
  # McNemar's test (with continuity correction)
  result <- mcnemar.test(mat, correct=TRUE)
  
  # CORRECTED Odds ratio: ratio of discordant pairs
  # OR = (QLoRA pass, FFT fail) / (QLoRA fail, FFT pass)
  # Add 0.5 to avoid division by zero
  odds_ratio <- (yes_no + 0.5) / (no_yes + 0.5)
  
  return(list(
    p_value = result$p.value,
    odds_ratio = odds_ratio,
    yes_yes = yes_yes,
    yes_no = yes_no,
    no_yes = no_yes,
    no_no = no_no
  ))
}

# ============== RUN ALL TESTS ==============

# Python
res_py_0_5b <- run_mcnemar(py_0_5b_qlora$pass1, py_0_5b_fft$pass1)
res_py_1_5b <- run_mcnemar(py_1_5b_qlora$pass1, py_1_5b_fft$pass1)
res_py_3b <- run_mcnemar(py_3b_qlora$pass1, py_3b_fft$pass1)

# Java
res_java_0_5b <- run_mcnemar(java_0_5b_qlora$pass1, java_0_5b_fft$pass1)
res_java_1_5b <- run_mcnemar(java_1_5b_qlora$pass1, java_1_5b_fft$pass1)
res_java_3b <- run_mcnemar(java_3b_qlora$pass1, java_3b_fft$pass1)

# ============== HOLM-BONFERRONI CORRECTION ==============
all_p <- c(res_py_0_5b$p_value, res_py_1_5b$p_value, res_py_3b$p_value,
           res_java_0_5b$p_value, res_java_1_5b$p_value, res_java_3b$p_value)
adjusted_p <- p.adjust(all_p, method="holm")

# ============================================================================
# FILE 1: RAW OUTPUT (detailed)
# ============================================================================
sink(paste0(output_path, "/rq2_pass1_mcnemar_RAW.txt"))

cat("================================================================================\n")
cat("RQ2: Pass@1 McNemar Test - MT-QLoRA vs MT-FFT\n")
cat("================================================================================\n\n")

cat("(a) Python\n")
cat("-----------\n")
cat("Qwen 0.5B: both_pass=", res_py_0_5b$yes_yes, ", QLoRA_only=", res_py_0_5b$yes_no, 
    ", FFT_only=", res_py_0_5b$no_yes, ", both_fail=", res_py_0_5b$no_no,
    ", p=", sprintf("%.6f", res_py_0_5b$p_value), 
    ", adj_p=", sprintf("%.6f", adjusted_p[1]),
    ", OR=", sprintf("%.4f", res_py_0_5b$odds_ratio), "\n")

cat("Qwen 1.5B: both_pass=", res_py_1_5b$yes_yes, ", QLoRA_only=", res_py_1_5b$yes_no, 
    ", FFT_only=", res_py_1_5b$no_yes, ", both_fail=", res_py_1_5b$no_no,
    ", p=", sprintf("%.6f", res_py_1_5b$p_value), 
    ", adj_p=", sprintf("%.6f", adjusted_p[2]),
    ", OR=", sprintf("%.4f", res_py_1_5b$odds_ratio), "\n")

cat("Qwen 3B: both_pass=", res_py_3b$yes_yes, ", QLoRA_only=", res_py_3b$yes_no, 
    ", FFT_only=", res_py_3b$no_yes, ", both_fail=", res_py_3b$no_no,
    ", p=", sprintf("%.6f", res_py_3b$p_value), 
    ", adj_p=", sprintf("%.6f", adjusted_p[3]),
    ", OR=", sprintf("%.4f", res_py_3b$odds_ratio), "\n\n")

cat("(b) Java\n")
cat("---------\n")
cat("Qwen 0.5B: both_pass=", res_java_0_5b$yes_yes, ", QLoRA_only=", res_java_0_5b$yes_no, 
    ", FFT_only=", res_java_0_5b$no_yes, ", both_fail=", res_java_0_5b$no_no,
    ", p=", sprintf("%.6f", res_java_0_5b$p_value), 
    ", adj_p=", sprintf("%.6f", adjusted_p[4]),
    ", OR=", sprintf("%.4f", res_java_0_5b$odds_ratio), "\n")

cat("Qwen 1.5B: both_pass=", res_java_1_5b$yes_yes, ", QLoRA_only=", res_java_1_5b$yes_no, 
    ", FFT_only=", res_java_1_5b$no_yes, ", both_fail=", res_java_1_5b$no_no,
    ", p=", sprintf("%.6f", res_java_1_5b$p_value), 
    ", adj_p=", sprintf("%.6f", adjusted_p[5]),
    ", OR=", sprintf("%.4f", res_java_1_5b$odds_ratio), "\n")

cat("Qwen 3B: both_pass=", res_java_3b$yes_yes, ", QLoRA_only=", res_java_3b$yes_no, 
    ", FFT_only=", res_java_3b$no_yes, ", both_fail=", res_java_3b$no_no,
    ", p=", sprintf("%.6f", res_java_3b$p_value), 
    ", adj_p=", sprintf("%.6f", adjusted_p[6]),
    ", OR=", sprintf("%.4f", res_java_3b$odds_ratio), "\n")

sink()

# ============================================================================
# FILE 2: CLEAN SORTED OUTPUT (table format)
# ============================================================================
sink(paste0(output_path, "/rq2_pass1_mcnemar_CLEAN.txt"))

cat("================================================================================\n")
cat("RQ2: Pass@1 McNemar Test - MT-QLoRA vs MT-FFT (Code Generation)\n")
cat("================================================================================\n\n")

cat("(a) Python\n")
cat("------------------------------------------------------------------------------\n")
cat(sprintf("%-12s %12s %12s %12s %10s %10s\n", "Model", "P-value", "Adj P-value", "Odds Ratio", "QLoRA_only", "FFT_only"))
cat("------------------------------------------------------------------------------\n")
cat(sprintf("%-12s %12.6f %12.6f %12.4f %10d %10d\n", "Qwen 0.5B", res_py_0_5b$p_value, adjusted_p[1], res_py_0_5b$odds_ratio, res_py_0_5b$yes_no, res_py_0_5b$no_yes))
cat(sprintf("%-12s %12.6f %12.6f %12.4f %10d %10d\n", "Qwen 1.5B", res_py_1_5b$p_value, adjusted_p[2], res_py_1_5b$odds_ratio, res_py_1_5b$yes_no, res_py_1_5b$no_yes))
cat(sprintf("%-12s %12.6f %12.6f %12.4f %10d %10d\n", "Qwen 3B", res_py_3b$p_value, adjusted_p[3], res_py_3b$odds_ratio, res_py_3b$yes_no, res_py_3b$no_yes))
cat("\n")

cat("(b) Java\n")
cat("------------------------------------------------------------------------------\n")
cat(sprintf("%-12s %12s %12s %12s %10s %10s\n", "Model", "P-value", "Adj P-value", "Odds Ratio", "QLoRA_only", "FFT_only"))
cat("------------------------------------------------------------------------------\n")
cat(sprintf("%-12s %12.6f %12.6f %12.4f %10d %10d\n", "Qwen 0.5B", res_java_0_5b$p_value, adjusted_p[4], res_java_0_5b$odds_ratio, res_java_0_5b$yes_no, res_java_0_5b$no_yes))
cat(sprintf("%-12s %12.6f %12.6f %12.4f %10d %10d\n", "Qwen 1.5B", res_java_1_5b$p_value, adjusted_p[5], res_java_1_5b$odds_ratio, res_java_1_5b$yes_no, res_java_1_5b$no_yes))
cat(sprintf("%-12s %12.6f %12.6f %12.4f %10d %10d\n", "Qwen 3B", res_java_3b$p_value, adjusted_p[6], res_java_3b$odds_ratio, res_java_3b$yes_no, res_java_3b$no_yes))
cat("\n")

cat("================================================================================\n")
cat("Note: P-values are Holm-Bonferroni corrected across all 6 comparisons\n")
cat("Odds Ratio = (QLoRA_only + 0.5) / (FFT_only + 0.5)\n")
cat("  OR > 1: QLoRA passes more often when they disagree\n")
cat("  OR < 1: FFT passes more often when they disagree\n")
cat("  OR = 1: No difference in discordant pairs\n")
cat("================================================================================\n")

sink()

print("Results saved to:")
print(paste0(output_path, "/rq2_pass1_mcnemar_RAW.txt"))
print(paste0(output_path, "/rq2_pass1_mcnemar_CLEAN.txt"))

