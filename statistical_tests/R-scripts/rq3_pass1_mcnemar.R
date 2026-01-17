rm(list=ls())

# Base path
base_path <- "/scratch/oldhome/mhaque/quantized-model-code-quality/statistical-tests/per-instance-value/pass1"
output_path <- "/scratch/oldhome/mhaque/quantized-model-code-quality/statistical-tests/R-scripts-v2"

# ============== LOAD PYTHON DATA ==============
# 3-task (CG+CS+CT) - using 3B
py_3b_3task <- read.csv(paste0(base_path, "/py/r-pass1_cg_qwen3_mt_qlora.csv"), header=TRUE)
# 2-task combinations
py_3b_cg_cs <- read.csv(paste0(base_path, "/py/r-pass1_cg_qwen3_cg_cs_mt_qlora.csv"), header=TRUE)
py_3b_cg_ct <- read.csv(paste0(base_path, "/py/r-pass1_cg_qwen3_cg_ct_mt_qlora.csv"), header=TRUE)

# ============== LOAD JAVA DATA ==============
# 3-task (CG+CS+CT) - using 3B
java_3b_3task <- read.csv(paste0(base_path, "/java/r-pass1_cg_qwen3_mt_qlora.csv"), header=TRUE)
# 2-task combinations
java_3b_cg_cs <- read.csv(paste0(base_path, "/java/r-pass1_cg_qwen3_cg_cs_mt_qlora.csv"), header=TRUE)
java_3b_cg_ct <- read.csv(paste0(base_path, "/java/r-pass1_cg_qwen3_cg_ct_mt_qlora.csv"), header=TRUE)

# ============== RENAME COLUMN (handle special character @) ==============
rename_col <- function(df) {
  colnames(df)[2] <- "pass1"
  return(df)
}

py_3b_3task <- rename_col(py_3b_3task)
py_3b_cg_cs <- rename_col(py_3b_cg_cs)
py_3b_cg_ct <- rename_col(py_3b_cg_ct)

java_3b_3task <- rename_col(java_3b_3task)
java_3b_cg_cs <- rename_col(java_3b_cg_cs)
java_3b_cg_ct <- rename_col(java_3b_cg_ct)

# ============== ENSURE SAME LENGTH ==============
align_data <- function(df1, df2) {
  min_len <- min(nrow(df1), nrow(df2))
  return(list(df1[1:min_len, ], df2[1:min_len, ]))
}

# Python: 3-task vs CG+CS
aligned <- align_data(py_3b_3task, py_3b_cg_cs)
py_3b_3task_cs <- aligned[[1]]; py_3b_cg_cs <- aligned[[2]]

# Python: 3-task vs CG+CT
aligned <- align_data(py_3b_3task, py_3b_cg_ct)
py_3b_3task_ct <- aligned[[1]]; py_3b_cg_ct <- aligned[[2]]

# Java: 3-task vs CG+CS
aligned <- align_data(java_3b_3task, java_3b_cg_cs)
java_3b_3task_cs <- aligned[[1]]; java_3b_cg_cs <- aligned[[2]]

# Java: 3-task vs CG+CT
aligned <- align_data(java_3b_3task, java_3b_cg_ct)
java_3b_3task_ct <- aligned[[1]]; java_3b_cg_ct <- aligned[[2]]

# ============== HELPER FUNCTIONS ==============

# Run McNemar's test (CORRECTED VERSION)
run_mcnemar <- function(col1_pass, col2_pass) {
  # Build contingency table
  # col1 = 3-task, col2 = 2-task
  yes_yes <- sum(col1_pass == 1 & col2_pass == 1)  # Both pass
  yes_no <- sum(col1_pass == 1 & col2_pass == 0)   # 3-task pass, 2-task fail
  no_yes <- sum(col1_pass == 0 & col2_pass == 1)   # 3-task fail, 2-task pass
  no_no <- sum(col1_pass == 0 & col2_pass == 0)    # Both fail
  
  # Create matrix for mcnemar.test (NO adjustment to cells)
  mat <- matrix(c(yes_yes, yes_no, no_yes, no_no), nrow=2, byrow=TRUE)
  
  # McNemar's test (with continuity correction)
  result <- mcnemar.test(mat, correct=TRUE)
  
  # CORRECTED Odds ratio: ratio of discordant pairs
  # OR = (3-task pass, 2-task fail) / (3-task fail, 2-task pass)
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
res_py_3task_vs_cgcs <- run_mcnemar(py_3b_3task_cs$pass1, py_3b_cg_cs$pass1)
res_py_3task_vs_cgct <- run_mcnemar(py_3b_3task_ct$pass1, py_3b_cg_ct$pass1)

# Java
res_java_3task_vs_cgcs <- run_mcnemar(java_3b_3task_cs$pass1, java_3b_cg_cs$pass1)
res_java_3task_vs_cgct <- run_mcnemar(java_3b_3task_ct$pass1, java_3b_cg_ct$pass1)

# ============== HOLM-BONFERRONI CORRECTION ==============
all_p <- c(res_py_3task_vs_cgcs$p_value, res_py_3task_vs_cgct$p_value,
           res_java_3task_vs_cgcs$p_value, res_java_3task_vs_cgct$p_value)
adjusted_p <- p.adjust(all_p, method="holm")

# ============================================================================
# FILE 1: RAW OUTPUT (detailed)
# ============================================================================
sink(paste0(output_path, "/rq3_pass1_mcnemar_RAW.txt"))

cat("================================================================================\n")
cat("RQ3: Pass@1 McNemar Test - 3-task vs 2-task (Qwen 3B)\n")
cat("================================================================================\n\n")

cat("(a) Python\n")
cat("-----------\n\n")

cat("Comparison: 3-task vs CG+CS\n")
cat("  Contingency: both_pass=", res_py_3task_vs_cgcs$yes_yes, 
    ", 3task_only=", res_py_3task_vs_cgcs$yes_no,
    ", 2task_only=", res_py_3task_vs_cgcs$no_yes, 
    ", both_fail=", res_py_3task_vs_cgcs$no_no, "\n")
cat("  Original P-value:", sprintf("%.6f", res_py_3task_vs_cgcs$p_value), 
    ", Adjusted P-value:", sprintf("%.6f", adjusted_p[1]),
    ", Odds Ratio:", sprintf("%.4f", res_py_3task_vs_cgcs$odds_ratio), "\n\n")

cat("Comparison: 3-task vs CG+CT\n")
cat("  Contingency: both_pass=", res_py_3task_vs_cgct$yes_yes, 
    ", 3task_only=", res_py_3task_vs_cgct$yes_no,
    ", 2task_only=", res_py_3task_vs_cgct$no_yes, 
    ", both_fail=", res_py_3task_vs_cgct$no_no, "\n")
cat("  Original P-value:", sprintf("%.6f", res_py_3task_vs_cgct$p_value), 
    ", Adjusted P-value:", sprintf("%.6f", adjusted_p[2]),
    ", Odds Ratio:", sprintf("%.4f", res_py_3task_vs_cgct$odds_ratio), "\n\n")

cat("(b) Java\n")
cat("-----------\n\n")

cat("Comparison: 3-task vs CG+CS\n")
cat("  Contingency: both_pass=", res_java_3task_vs_cgcs$yes_yes, 
    ", 3task_only=", res_java_3task_vs_cgcs$yes_no,
    ", 2task_only=", res_java_3task_vs_cgcs$no_yes, 
    ", both_fail=", res_java_3task_vs_cgcs$no_no, "\n")
cat("  Original P-value:", sprintf("%.6f", res_java_3task_vs_cgcs$p_value), 
    ", Adjusted P-value:", sprintf("%.6f", adjusted_p[3]),
    ", Odds Ratio:", sprintf("%.4f", res_java_3task_vs_cgcs$odds_ratio), "\n\n")

cat("Comparison: 3-task vs CG+CT\n")
cat("  Contingency: both_pass=", res_java_3task_vs_cgct$yes_yes, 
    ", 3task_only=", res_java_3task_vs_cgct$yes_no,
    ", 2task_only=", res_java_3task_vs_cgct$no_yes, 
    ", both_fail=", res_java_3task_vs_cgct$no_no, "\n")
cat("  Original P-value:", sprintf("%.6f", res_java_3task_vs_cgct$p_value), 
    ", Adjusted P-value:", sprintf("%.6f", adjusted_p[4]),
    ", Odds Ratio:", sprintf("%.4f", res_java_3task_vs_cgct$odds_ratio), "\n\n")

sink()

# ============================================================================
# FILE 2: CLEAN SORTED OUTPUT (table format)
# ============================================================================
sink(paste0(output_path, "/rq3_pass1_mcnemar_CLEAN.txt"))

cat("================================================================================\n")
cat("RQ3: Pass@1 McNemar Test - 3-task vs 2-task (Code Generation, Qwen 3B)\n")
cat("================================================================================\n\n")

cat("------------------------------------------------------------------------------\n")
cat("(a) Python: 3-task vs 2-task\n")
cat("------------------------------------------------------------------------------\n")
cat(sprintf("%-20s %12s %12s %12s %10s %10s\n", "Comparison", "P-value", "Adj P-value", "Odds Ratio", "3task_only", "2task_only"))
cat("------------------------------------------------------------------------------\n")
cat(sprintf("%-20s %12.6f %12.6f %12.4f %10d %10d\n", "3-task vs CG+CS", 
            res_py_3task_vs_cgcs$p_value, adjusted_p[1], res_py_3task_vs_cgcs$odds_ratio,
            res_py_3task_vs_cgcs$yes_no, res_py_3task_vs_cgcs$no_yes))
cat(sprintf("%-20s %12.6f %12.6f %12.4f %10d %10d\n", "3-task vs CG+CT", 
            res_py_3task_vs_cgct$p_value, adjusted_p[2], res_py_3task_vs_cgct$odds_ratio,
            res_py_3task_vs_cgct$yes_no, res_py_3task_vs_cgct$no_yes))
cat("\n")

cat("------------------------------------------------------------------------------\n")
cat("(b) Java: 3-task vs 2-task\n")
cat("------------------------------------------------------------------------------\n")
cat(sprintf("%-20s %12s %12s %12s %10s %10s\n", "Comparison", "P-value", "Adj P-value", "Odds Ratio", "3task_only", "2task_only"))
cat("------------------------------------------------------------------------------\n")
cat(sprintf("%-20s %12.6f %12.6f %12.4f %10d %10d\n", "3-task vs CG+CS", 
            res_java_3task_vs_cgcs$p_value, adjusted_p[3], res_java_3task_vs_cgcs$odds_ratio,
            res_java_3task_vs_cgcs$yes_no, res_java_3task_vs_cgcs$no_yes))
cat(sprintf("%-20s %12.6f %12.6f %12.4f %10d %10d\n", "3-task vs CG+CT", 
            res_java_3task_vs_cgct$p_value, adjusted_p[4], res_java_3task_vs_cgct$odds_ratio,
            res_java_3task_vs_cgct$yes_no, res_java_3task_vs_cgct$no_yes))
cat("\n")

cat("================================================================================\n")
cat("Note: P-values are Holm-Bonferroni corrected across all 4 comparisons\n")
cat("Odds Ratio = (3task_only + 0.5) / (2task_only + 0.5)\n")
cat("  OR > 1: 3-task passes more often when they disagree\n")
cat("  OR < 1: 2-task passes more often when they disagree\n")
cat("  OR = 1: No difference in discordant pairs\n")
cat("3task_only = cases where 3-task passes but 2-task fails\n")
cat("2task_only = cases where 2-task passes but 3-task fails\n")
cat("================================================================================\n")

sink()

print("Results saved to:")
print(paste0(output_path, "/rq3_pass1_mcnemar_RAW.txt"))
print(paste0(output_path, "/rq3_pass1_mcnemar_CLEAN.txt"))