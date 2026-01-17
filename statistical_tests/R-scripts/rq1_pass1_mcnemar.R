rm(list=ls())

# Base path
base_path <- "/scratch/oldhome/mhaque/quantized-model-code-quality/statistical-tests/per-instance-value/pass1"
output_path <- "/scratch/oldhome/mhaque/quantized-model-code-quality/statistical-tests/R-scripts-v2"

# ============== LOAD PYTHON DATA ==============
py_0_5b_st <- read.csv(paste0(base_path, "/py/r-pass1_cg_qwen0_5_st_qlora.csv"), header=TRUE)
py_0_5b_mt <- read.csv(paste0(base_path, "/py/r-pass1_cg_qwen0_5_mt_qlora.csv"), header=TRUE)
py_1_5b_st <- read.csv(paste0(base_path, "/py/r-pass1_cg_qwen1_5_st_qlora.csv"), header=TRUE)
py_1_5b_mt <- read.csv(paste0(base_path, "/py/r-pass1_cg_qwen1_5_mt_qlora.csv"), header=TRUE)
py_3b_st <- read.csv(paste0(base_path, "/py/r-pass1_cg_qwen3_st_qlora.csv"), header=TRUE)
py_3b_mt <- read.csv(paste0(base_path, "/py/r-pass1_cg_qwen3_mt_qlora.csv"), header=TRUE)

# ============== LOAD JAVA DATA ==============
java_0_5b_st <- read.csv(paste0(base_path, "/java/r-pass1_cg_qwen0_5_st_qlora.csv"), header=TRUE)
java_0_5b_mt <- read.csv(paste0(base_path, "/java/r-pass1_cg_qwen0_5_mt_qlora.csv"), header=TRUE)
java_1_5b_st <- read.csv(paste0(base_path, "/java/r-pass1_cg_qwen1_5_st_qlora.csv"), header=TRUE)
java_1_5b_mt <- read.csv(paste0(base_path, "/java/r-pass1_cg_qwen1_5_mt_qlora.csv"), header=TRUE)
java_3b_st <- read.csv(paste0(base_path, "/java/r-pass1_cg_qwen3_st_qlora.csv"), header=TRUE)
java_3b_mt <- read.csv(paste0(base_path, "/java/r-pass1_cg_qwen3_mt_qlora.csv"), header=TRUE)

# ============== RENAME COLUMN (handle special character @) ==============
rename_col <- function(df) {
  colnames(df)[2] <- "pass1"
  return(df)
}

py_0_5b_st <- rename_col(py_0_5b_st); py_0_5b_mt <- rename_col(py_0_5b_mt)
py_1_5b_st <- rename_col(py_1_5b_st); py_1_5b_mt <- rename_col(py_1_5b_mt)
py_3b_st <- rename_col(py_3b_st); py_3b_mt <- rename_col(py_3b_mt)

java_0_5b_st <- rename_col(java_0_5b_st); java_0_5b_mt <- rename_col(java_0_5b_mt)
java_1_5b_st <- rename_col(java_1_5b_st); java_1_5b_mt <- rename_col(java_1_5b_mt)
java_3b_st <- rename_col(java_3b_st); java_3b_mt <- rename_col(java_3b_mt)

# ============== ENSURE SAME LENGTH (merge by Task.ID) ==============
align_data <- function(df1, df2) {
  merged <- merge(df1, df2, by="Task.ID", suffixes=c("_st", "_mt"))
  return(merged)
}

# Python
py_0_5b <- align_data(py_0_5b_st, py_0_5b_mt)
py_1_5b <- align_data(py_1_5b_st, py_1_5b_mt)
py_3b <- align_data(py_3b_st, py_3b_mt)

# Java
java_0_5b <- align_data(java_0_5b_st, java_0_5b_mt)
java_1_5b <- align_data(java_1_5b_st, java_1_5b_mt)
java_3b <- align_data(java_3b_st, java_3b_mt)

# ============== HELPER FUNCTIONS ==============

# Run McNemar's test (CORRECTED VERSION)
run_mcnemar <- function(col1_pass, col2_pass) {
  # Build contingency table
  # col1 = ST, col2 = MT
  yes_yes <- sum(col1_pass == 1 & col2_pass == 1)  # Both pass
  yes_no <- sum(col1_pass == 1 & col2_pass == 0)   # ST pass, MT fail
  no_yes <- sum(col1_pass == 0 & col2_pass == 1)   # ST fail, MT pass
  no_no <- sum(col1_pass == 0 & col2_pass == 0)    # Both fail
  
  # Create matrix for mcnemar.test (NO adjustment to cells)
  # Matrix format:
  #              MT Pass  MT Fail
  # ST Pass    [ yes_yes, yes_no ]
  # ST Fail    [ no_yes,  no_no  ]
  mat <- matrix(c(yes_yes, yes_no, no_yes, no_no), nrow=2, byrow=TRUE)
  
  # McNemar's test (with continuity correction)
  result <- mcnemar.test(mat, correct=TRUE)
  
  # CORRECTED Odds ratio: ratio of discordant pairs
  # OR = (ST pass, MT fail) / (ST fail, MT pass)
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
res_py_0_5b <- run_mcnemar(py_0_5b$pass1_st, py_0_5b$pass1_mt)
res_py_1_5b <- run_mcnemar(py_1_5b$pass1_st, py_1_5b$pass1_mt)
res_py_3b <- run_mcnemar(py_3b$pass1_st, py_3b$pass1_mt)

# Java
res_java_0_5b <- run_mcnemar(java_0_5b$pass1_st, java_0_5b$pass1_mt)
res_java_1_5b <- run_mcnemar(java_1_5b$pass1_st, java_1_5b$pass1_mt)
res_java_3b <- run_mcnemar(java_3b$pass1_st, java_3b$pass1_mt)

# ============== BH-BONFERRONI CORRECTION ==============
all_p <- c(res_py_0_5b$p_value, res_py_1_5b$p_value, res_py_3b$p_value,
           res_java_0_5b$p_value, res_java_1_5b$p_value, res_java_3b$p_value)
adjusted_p <- p.adjust(all_p, method="BH")

# ============================================================================
# FILE 1: RAW OUTPUT (detailed)
# ============================================================================
sink(paste0(output_path, "/rq1_pass1_mcnemar_RAW.txt"))

cat("================================================================================\n")
cat("RQ1: Pass@1 McNemar Test - ST-QLoRA vs MT-QLoRA\n")
cat("================================================================================\n\n")

cat("Comparison: Python 0.5B ST-QLoRA vs MT-QLoRA\n")
cat("  Contingency: both_pass=", res_py_0_5b$yes_yes, ", ST_only=", res_py_0_5b$yes_no, 
    ", MT_only=", res_py_0_5b$no_yes, ", both_fail=", res_py_0_5b$no_no, "\n")
cat("  Original P-value:", sprintf("%.6f", res_py_0_5b$p_value), 
    ", Adjusted P-value:", sprintf("%.6f", adjusted_p[1]),
    ", Odds Ratio:", sprintf("%.4f", res_py_0_5b$odds_ratio), "\n\n")

cat("Comparison: Python 1.5B ST-QLoRA vs MT-QLoRA\n")
cat("  Contingency: both_pass=", res_py_1_5b$yes_yes, ", ST_only=", res_py_1_5b$yes_no, 
    ", MT_only=", res_py_1_5b$no_yes, ", both_fail=", res_py_1_5b$no_no, "\n")
cat("  Original P-value:", sprintf("%.6f", res_py_1_5b$p_value), 
    ", Adjusted P-value:", sprintf("%.6f", adjusted_p[2]),
    ", Odds Ratio:", sprintf("%.4f", res_py_1_5b$odds_ratio), "\n\n")

cat("Comparison: Python 3B ST-QLoRA vs MT-QLoRA\n")
cat("  Contingency: both_pass=", res_py_3b$yes_yes, ", ST_only=", res_py_3b$yes_no, 
    ", MT_only=", res_py_3b$no_yes, ", both_fail=", res_py_3b$no_no, "\n")
cat("  Original P-value:", sprintf("%.6f", res_py_3b$p_value), 
    ", Adjusted P-value:", sprintf("%.6f", adjusted_p[3]),
    ", Odds Ratio:", sprintf("%.4f", res_py_3b$odds_ratio), "\n\n")

cat("Comparison: Java 0.5B ST-QLoRA vs MT-QLoRA\n")
cat("  Contingency: both_pass=", res_java_0_5b$yes_yes, ", ST_only=", res_java_0_5b$yes_no, 
    ", MT_only=", res_java_0_5b$no_yes, ", both_fail=", res_java_0_5b$no_no, "\n")
cat("  Original P-value:", sprintf("%.6f", res_java_0_5b$p_value), 
    ", Adjusted P-value:", sprintf("%.6f", adjusted_p[4]),
    ", Odds Ratio:", sprintf("%.4f", res_java_0_5b$odds_ratio), "\n\n")

cat("Comparison: Java 1.5B ST-QLoRA vs MT-QLoRA\n")
cat("  Contingency: both_pass=", res_java_1_5b$yes_yes, ", ST_only=", res_java_1_5b$yes_no, 
    ", MT_only=", res_java_1_5b$no_yes, ", both_fail=", res_java_1_5b$no_no, "\n")
cat("  Original P-value:", sprintf("%.6f", res_java_1_5b$p_value), 
    ", Adjusted P-value:", sprintf("%.6f", adjusted_p[5]),
    ", Odds Ratio:", sprintf("%.4f", res_java_1_5b$odds_ratio), "\n\n")

cat("Comparison: Java 3B ST-QLoRA vs MT-QLoRA\n")
cat("  Contingency: both_pass=", res_java_3b$yes_yes, ", ST_only=", res_java_3b$yes_no, 
    ", MT_only=", res_java_3b$no_yes, ", both_fail=", res_java_3b$no_no, "\n")
cat("  Original P-value:", sprintf("%.6f", res_java_3b$p_value), 
    ", Adjusted P-value:", sprintf("%.6f", adjusted_p[6]),
    ", Odds Ratio:", sprintf("%.4f", res_java_3b$odds_ratio), "\n\n")

sink()

# ============================================================================
# FILE 2: CLEAN SORTED OUTPUT (table format)
# ============================================================================
sink(paste0(output_path, "/rq1_pass1_mcnemar_CLEAN.txt"))

cat("================================================================================\n")
cat("RQ1: Pass@1 McNemar Test - ST-QLoRA vs MT-QLoRA (Code Generation)\n")
cat("================================================================================\n\n")

cat("------------------------------------------------------------------------------\n")
cat("PYTHON: ST-QLoRA vs MT-QLoRA\n")
cat("------------------------------------------------------------------------------\n")
cat(sprintf("%-12s %12s %12s %12s %10s %10s\n", "Model", "P-value", "Adj P-value", "Odds Ratio", "ST_only", "MT_only"))
cat("------------------------------------------------------------------------------\n")
cat(sprintf("%-12s %12.6f %12.6f %12.4f %10d %10d\n", "Qwen 0.5B", res_py_0_5b$p_value, adjusted_p[1], res_py_0_5b$odds_ratio, res_py_0_5b$yes_no, res_py_0_5b$no_yes))
cat(sprintf("%-12s %12.6f %12.6f %12.4f %10d %10d\n", "Qwen 1.5B", res_py_1_5b$p_value, adjusted_p[2], res_py_1_5b$odds_ratio, res_py_1_5b$yes_no, res_py_1_5b$no_yes))
cat(sprintf("%-12s %12.6f %12.6f %12.4f %10d %10d\n", "Qwen 3B", res_py_3b$p_value, adjusted_p[3], res_py_3b$odds_ratio, res_py_3b$yes_no, res_py_3b$no_yes))
cat("\n")

cat("------------------------------------------------------------------------------\n")
cat("JAVA: ST-QLoRA vs MT-QLoRA\n")
cat("------------------------------------------------------------------------------\n")
cat(sprintf("%-12s %12s %12s %12s %10s %10s\n", "Model", "P-value", "Adj P-value", "Odds Ratio", "ST_only", "MT_only"))
cat("------------------------------------------------------------------------------\n")
cat(sprintf("%-12s %12.6f %12.6f %12.4f %10d %10d\n", "Qwen 0.5B", res_java_0_5b$p_value, adjusted_p[4], res_java_0_5b$odds_ratio, res_java_0_5b$yes_no, res_java_0_5b$no_yes))
cat(sprintf("%-12s %12.6f %12.6f %12.4f %10d %10d\n", "Qwen 1.5B", res_java_1_5b$p_value, adjusted_p[5], res_java_1_5b$odds_ratio, res_java_1_5b$yes_no, res_java_1_5b$no_yes))
cat(sprintf("%-12s %12.6f %12.6f %12.4f %10d %10d\n", "Qwen 3B", res_java_3b$p_value, adjusted_p[6], res_java_3b$odds_ratio, res_java_3b$yes_no, res_java_3b$no_yes))
cat("\n")

cat("================================================================================\n")
cat("Note: P-values are BH-Bonferroni corrected across all 6 comparisons\n")
cat("Odds Ratio = (ST_only + 0.5) / (MT_only + 0.5)\n")
cat("  OR > 1: ST passes more often when they disagree\n")
cat("  OR < 1: MT passes more often when they disagree\n")
cat("  OR = 1: No difference in discordant pairs\n")
cat("ST_only = cases where ST passes but MT fails\n")
cat("MT_only = cases where MT passes but ST fails\n")
cat("================================================================================\n")

sink()

print("Results saved to:")
print(paste0(output_path, "/rq1_pass1_mcnemar_RAW.txt"))
print(paste0(output_path, "/rq1_pass1_mcnemar_CLEAN.txt"))