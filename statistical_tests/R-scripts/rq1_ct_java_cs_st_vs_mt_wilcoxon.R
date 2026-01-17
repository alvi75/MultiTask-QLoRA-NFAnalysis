rm(list=ls())

if (!require("effsize")) install.packages("effsize")
library(effsize)

# Base path
base_path <- "/scratch/oldhome/mhaque/quantized-model-code-quality/statistical-tests/per-instance-value"
output_path <- "/scratch/oldhome/mhaque/quantized-model-code-quality/statistical-tests/R-scripts-v2"

# ============== LOAD LIZARD DATA ==============
lizard_0_5b_st <- read.csv(paste0(base_path, "/lizard/translation/java_cs/r-lizard_ct_0_5b-st-qlora.csv"), header=TRUE)
lizard_0_5b_mt <- read.csv(paste0(base_path, "/lizard/translation/java_cs/r-lizard_ct_0_5b-mt-qlora.csv"), header=TRUE)
lizard_1_5b_st <- read.csv(paste0(base_path, "/lizard/translation/java_cs/r-lizard_ct_1_5b-st-qlora.csv"), header=TRUE)
lizard_1_5b_mt <- read.csv(paste0(base_path, "/lizard/translation/java_cs/r-lizard_ct_1_5b-mt-qlora.csv"), header=TRUE)
lizard_3b_st <- read.csv(paste0(base_path, "/lizard/translation/java_cs/r-lizard_ct_3b-st-qlora.csv"), header=TRUE)
lizard_3b_mt <- read.csv(paste0(base_path, "/lizard/translation/java_cs/r-lizard_ct_3b-mt-qlora.csv"), header=TRUE)

# ============== LOAD CODEBLEU DATA ==============
codebleu_0_5b_st <- read.csv(paste0(base_path, "/codebleu/translation/java_cs/ct_qwen0_5_st_qlora.csv"), header=TRUE)
codebleu_0_5b_mt <- read.csv(paste0(base_path, "/codebleu/translation/java_cs/ct_qwen0_5_mt_qlora.csv"), header=TRUE)
codebleu_1_5b_st <- read.csv(paste0(base_path, "/codebleu/translation/java_cs/ct_qwen1_5_st_qlora.csv"), header=TRUE)
codebleu_1_5b_mt <- read.csv(paste0(base_path, "/codebleu/translation/java_cs/ct_qwen1_5_mt_qlora.csv"), header=TRUE)
codebleu_3b_st <- read.csv(paste0(base_path, "/codebleu/translation/java_cs/ct_qwen3_st_qlora.csv"), header=TRUE)
codebleu_3b_mt <- read.csv(paste0(base_path, "/codebleu/translation/java_cs/ct_qwen3_mt_qlora.csv"), header=TRUE)

# ============== LOAD ROSLYN DATA ==============
roslyn_0_5b_st <- read.csv(paste0(base_path, "/roslyn/java_cs/r-roslyn_ct_0_5b-st-qlora.csv"), header=TRUE)
roslyn_0_5b_mt <- read.csv(paste0(base_path, "/roslyn/java_cs/r-roslyn_ct_0_5b-mt-qlora.csv"), header=TRUE)
roslyn_1_5b_st <- read.csv(paste0(base_path, "/roslyn/java_cs/r-roslyn_ct_1_5b-st-qlora.csv"), header=TRUE)
roslyn_1_5b_mt <- read.csv(paste0(base_path, "/roslyn/java_cs/r-roslyn_ct_1_5b-mt-qlora.csv"), header=TRUE)
roslyn_3b_st <- read.csv(paste0(base_path, "/roslyn/java_cs/r-roslyn_ct_3b-st-qlora.csv"), header=TRUE)
roslyn_3b_mt <- read.csv(paste0(base_path, "/roslyn/java_cs/r-roslyn_ct_3b-mt-qlora.csv"), header=TRUE)

# ============== ENSURE SAME LENGTH ==============
min_len <- min(nrow(lizard_0_5b_st), nrow(lizard_0_5b_mt))
lizard_0_5b_st <- lizard_0_5b_st[1:min_len, ]; lizard_0_5b_mt <- lizard_0_5b_mt[1:min_len, ]
min_len <- min(nrow(lizard_1_5b_st), nrow(lizard_1_5b_mt))
lizard_1_5b_st <- lizard_1_5b_st[1:min_len, ]; lizard_1_5b_mt <- lizard_1_5b_mt[1:min_len, ]
min_len <- min(nrow(lizard_3b_st), nrow(lizard_3b_mt))
lizard_3b_st <- lizard_3b_st[1:min_len, ]; lizard_3b_mt <- lizard_3b_mt[1:min_len, ]

min_len <- min(nrow(codebleu_0_5b_st), nrow(codebleu_0_5b_mt))
codebleu_0_5b_st <- codebleu_0_5b_st[1:min_len, ]; codebleu_0_5b_mt <- codebleu_0_5b_mt[1:min_len, ]
min_len <- min(nrow(codebleu_1_5b_st), nrow(codebleu_1_5b_mt))
codebleu_1_5b_st <- codebleu_1_5b_st[1:min_len, ]; codebleu_1_5b_mt <- codebleu_1_5b_mt[1:min_len, ]
min_len <- min(nrow(codebleu_3b_st), nrow(codebleu_3b_mt))
codebleu_3b_st <- codebleu_3b_st[1:min_len, ]; codebleu_3b_mt <- codebleu_3b_mt[1:min_len, ]

min_len <- min(nrow(roslyn_0_5b_st), nrow(roslyn_0_5b_mt))
roslyn_0_5b_st <- roslyn_0_5b_st[1:min_len, ]; roslyn_0_5b_mt <- roslyn_0_5b_mt[1:min_len, ]
min_len <- min(nrow(roslyn_1_5b_st), nrow(roslyn_1_5b_mt))
roslyn_1_5b_st <- roslyn_1_5b_st[1:min_len, ]; roslyn_1_5b_mt <- roslyn_1_5b_mt[1:min_len, ]
min_len <- min(nrow(roslyn_3b_st), nrow(roslyn_3b_mt))
roslyn_3b_st <- roslyn_3b_st[1:min_len, ]; roslyn_3b_mt <- roslyn_3b_mt[1:min_len, ]

# ============== HELPER FUNCTIONS ==============
is_constant <- function(x) {
  return(length(unique(x)) == 1)
}

wilcox_test <- function(x, y) {
  if (is_constant(x) & is_constant(y)) {
    return(NA)
  }
  return(wilcox.test(x, y, alternative="two.side", paired=TRUE, exact=FALSE, correct=FALSE)$p.value)
}

get_delta <- function(x, y) {
  return(cliff.delta(x, y)$estimate)
}

# ============================================================================
# FILE 1: RAW OUTPUT (detailed)
# ============================================================================
sink(paste0(output_path, "/rq1_ct_java_cs_st_vs_mt_wilcoxon_RAW.txt"))

# ============== QWEN 0.5B ==============
print("********************** QWEN 0.5B: ST-QLoRA vs MT-QLoRA (Java to C#) *********************************")

res_0_5b = list(Wilcoxon.p = c())

# CODEBLEU
print("CodeBLEU - ST vs MT (0.5B)")
res_0_5b$Wilcoxon.p = append(res_0_5b$Wilcoxon.p, wilcox_test(codebleu_0_5b_st$codebleu, codebleu_0_5b_mt$codebleu))
cliff.delta(codebleu_0_5b_st$codebleu, codebleu_0_5b_mt$codebleu)

# LIZARD
print("Lines_of_Code - ST vs MT (0.5B)")
res_0_5b$Wilcoxon.p = append(res_0_5b$Wilcoxon.p, wilcox_test(lizard_0_5b_st$Lines_of_Code, lizard_0_5b_mt$Lines_of_Code))
cliff.delta(lizard_0_5b_st$Lines_of_Code, lizard_0_5b_mt$Lines_of_Code)

print("Token_Count - ST vs MT (0.5B)")
res_0_5b$Wilcoxon.p = append(res_0_5b$Wilcoxon.p, wilcox_test(lizard_0_5b_st$Token_Count, lizard_0_5b_mt$Token_Count))
cliff.delta(lizard_0_5b_st$Token_Count, lizard_0_5b_mt$Token_Count)

print("Detection_Rate - ST vs MT (0.5B)")
res_0_5b$Wilcoxon.p = append(res_0_5b$Wilcoxon.p, wilcox_test(lizard_0_5b_st$Detection_Rate, lizard_0_5b_mt$Detection_Rate))
cliff.delta(lizard_0_5b_st$Detection_Rate, lizard_0_5b_mt$Detection_Rate)

print("Cyclomatic_Complexity - ST vs MT (0.5B)")
res_0_5b$Wilcoxon.p = append(res_0_5b$Wilcoxon.p, wilcox_test(lizard_0_5b_st$Cyclomatic_Complexity, lizard_0_5b_mt$Cyclomatic_Complexity))
cliff.delta(lizard_0_5b_st$Cyclomatic_Complexity, lizard_0_5b_mt$Cyclomatic_Complexity)

# ROSLYN
print("Syntax_Errors - ST vs MT (0.5B)")
res_0_5b$Wilcoxon.p = append(res_0_5b$Wilcoxon.p, wilcox_test(roslyn_0_5b_st$Syntax_Errors, roslyn_0_5b_mt$Syntax_Errors))
cliff.delta(roslyn_0_5b_st$Syntax_Errors, roslyn_0_5b_mt$Syntax_Errors)

print("Maintainability - ST vs MT (0.5B)")
res_0_5b$Wilcoxon.p = append(res_0_5b$Wilcoxon.p, wilcox_test(roslyn_0_5b_st$Maintainability, roslyn_0_5b_mt$Maintainability))
cliff.delta(roslyn_0_5b_st$Maintainability, roslyn_0_5b_mt$Maintainability)

# ============== QWEN 1.5B ==============
print("********************** QWEN 1.5B: ST-QLoRA vs MT-QLoRA (Java to C#) *********************************")

res_1_5b = list(Wilcoxon.p = c())

# CODEBLEU
print("CodeBLEU - ST vs MT (1.5B)")
res_1_5b$Wilcoxon.p = append(res_1_5b$Wilcoxon.p, wilcox_test(codebleu_1_5b_st$codebleu, codebleu_1_5b_mt$codebleu))
cliff.delta(codebleu_1_5b_st$codebleu, codebleu_1_5b_mt$codebleu)

# LIZARD
print("Lines_of_Code - ST vs MT (1.5B)")
res_1_5b$Wilcoxon.p = append(res_1_5b$Wilcoxon.p, wilcox_test(lizard_1_5b_st$Lines_of_Code, lizard_1_5b_mt$Lines_of_Code))
cliff.delta(lizard_1_5b_st$Lines_of_Code, lizard_1_5b_mt$Lines_of_Code)

print("Token_Count - ST vs MT (1.5B)")
res_1_5b$Wilcoxon.p = append(res_1_5b$Wilcoxon.p, wilcox_test(lizard_1_5b_st$Token_Count, lizard_1_5b_mt$Token_Count))
cliff.delta(lizard_1_5b_st$Token_Count, lizard_1_5b_mt$Token_Count)

print("Detection_Rate - ST vs MT (1.5B)")
res_1_5b$Wilcoxon.p = append(res_1_5b$Wilcoxon.p, wilcox_test(lizard_1_5b_st$Detection_Rate, lizard_1_5b_mt$Detection_Rate))
cliff.delta(lizard_1_5b_st$Detection_Rate, lizard_1_5b_mt$Detection_Rate)

print("Cyclomatic_Complexity - ST vs MT (1.5B)")
res_1_5b$Wilcoxon.p = append(res_1_5b$Wilcoxon.p, wilcox_test(lizard_1_5b_st$Cyclomatic_Complexity, lizard_1_5b_mt$Cyclomatic_Complexity))
cliff.delta(lizard_1_5b_st$Cyclomatic_Complexity, lizard_1_5b_mt$Cyclomatic_Complexity)

# ROSLYN
print("Syntax_Errors - ST vs MT (1.5B)")
res_1_5b$Wilcoxon.p = append(res_1_5b$Wilcoxon.p, wilcox_test(roslyn_1_5b_st$Syntax_Errors, roslyn_1_5b_mt$Syntax_Errors))
cliff.delta(roslyn_1_5b_st$Syntax_Errors, roslyn_1_5b_mt$Syntax_Errors)

print("Maintainability - ST vs MT (1.5B)")
res_1_5b$Wilcoxon.p = append(res_1_5b$Wilcoxon.p, wilcox_test(roslyn_1_5b_st$Maintainability, roslyn_1_5b_mt$Maintainability))
cliff.delta(roslyn_1_5b_st$Maintainability, roslyn_1_5b_mt$Maintainability)

# ============== QWEN 3B ==============
print("********************** QWEN 3B: ST-QLoRA vs MT-QLoRA (Java to C#) *********************************")

res_3b = list(Wilcoxon.p = c())

# CODEBLEU
print("CodeBLEU - ST vs MT (3B)")
res_3b$Wilcoxon.p = append(res_3b$Wilcoxon.p, wilcox_test(codebleu_3b_st$codebleu, codebleu_3b_mt$codebleu))
cliff.delta(codebleu_3b_st$codebleu, codebleu_3b_mt$codebleu)

# LIZARD
print("Lines_of_Code - ST vs MT (3B)")
res_3b$Wilcoxon.p = append(res_3b$Wilcoxon.p, wilcox_test(lizard_3b_st$Lines_of_Code, lizard_3b_mt$Lines_of_Code))
cliff.delta(lizard_3b_st$Lines_of_Code, lizard_3b_mt$Lines_of_Code)

print("Token_Count - ST vs MT (3B)")
res_3b$Wilcoxon.p = append(res_3b$Wilcoxon.p, wilcox_test(lizard_3b_st$Token_Count, lizard_3b_mt$Token_Count))
cliff.delta(lizard_3b_st$Token_Count, lizard_3b_mt$Token_Count)

print("Detection_Rate - ST vs MT (3B)")
res_3b$Wilcoxon.p = append(res_3b$Wilcoxon.p, wilcox_test(lizard_3b_st$Detection_Rate, lizard_3b_mt$Detection_Rate))
cliff.delta(lizard_3b_st$Detection_Rate, lizard_3b_mt$Detection_Rate)

print("Cyclomatic_Complexity - ST vs MT (3B)")
res_3b$Wilcoxon.p = append(res_3b$Wilcoxon.p, wilcox_test(lizard_3b_st$Cyclomatic_Complexity, lizard_3b_mt$Cyclomatic_Complexity))
cliff.delta(lizard_3b_st$Cyclomatic_Complexity, lizard_3b_mt$Cyclomatic_Complexity)

# ROSLYN
print("Syntax_Errors - ST vs MT (3B)")
res_3b$Wilcoxon.p = append(res_3b$Wilcoxon.p, wilcox_test(roslyn_3b_st$Syntax_Errors, roslyn_3b_mt$Syntax_Errors))
cliff.delta(roslyn_3b_st$Syntax_Errors, roslyn_3b_mt$Syntax_Errors)

print("Maintainability - ST vs MT (3B)")
res_3b$Wilcoxon.p = append(res_3b$Wilcoxon.p, wilcox_test(roslyn_3b_st$Maintainability, roslyn_3b_mt$Maintainability))
cliff.delta(roslyn_3b_st$Maintainability, roslyn_3b_mt$Maintainability)

# ============== PRINT RAW RESULTS ==============
print("***************************************************************************")

res_df_0_5b = data.frame(Wilcoxon.p = res_0_5b$Wilcoxon.p)
res_df_0_5b$Wilcoxon.p = p.adjust(res_df_0_5b$Wilcoxon.p, method="holm")
print("Qwen 0.5B Results (Adjusted P-values):")
print(res_df_0_5b)

res_df_1_5b = data.frame(Wilcoxon.p = res_1_5b$Wilcoxon.p)
res_df_1_5b$Wilcoxon.p = p.adjust(res_df_1_5b$Wilcoxon.p, method="holm")
print("Qwen 1.5B Results (Adjusted P-values):")
print(res_df_1_5b)

res_df_3b = data.frame(Wilcoxon.p = res_3b$Wilcoxon.p)
res_df_3b$Wilcoxon.p = p.adjust(res_df_3b$Wilcoxon.p, method="holm")
print("Qwen 3B Results (Adjusted P-values):")
print(res_df_3b)

sink()

# ============================================================================
# FILE 2: CLEAN SORTED OUTPUT (table format)
# ============================================================================
sink(paste0(output_path, "/rq1_ct_java_cs_st_vs_mt_wilcoxon_CLEAN.txt"))

cat("================================================================================\n")
cat("RQ1: Code Translation - Java to C# (ST-QLoRA vs MT-QLoRA) - Wilcoxon Signed-Rank Test\n")
cat("================================================================================\n\n")

# Metric labels (CodeBLEU + Lizard + Roslyn)
metric_labels <- c("CodeBLEU",
                   "Lizard_LoC", "Lizard_ToK", "Lizard_DR", "Lizard_CyC",
                   "Roslyn_SynErr", "Roslyn_Main")

# Get delta values for 0.5B
delta_0_5b <- c(
  get_delta(codebleu_0_5b_st$codebleu, codebleu_0_5b_mt$codebleu),
  get_delta(lizard_0_5b_st$Lines_of_Code, lizard_0_5b_mt$Lines_of_Code),
  get_delta(lizard_0_5b_st$Token_Count, lizard_0_5b_mt$Token_Count),
  get_delta(lizard_0_5b_st$Detection_Rate, lizard_0_5b_mt$Detection_Rate),
  get_delta(lizard_0_5b_st$Cyclomatic_Complexity, lizard_0_5b_mt$Cyclomatic_Complexity),
  get_delta(roslyn_0_5b_st$Syntax_Errors, roslyn_0_5b_mt$Syntax_Errors),
  get_delta(roslyn_0_5b_st$Maintainability, roslyn_0_5b_mt$Maintainability)
)

# Get delta values for 1.5B
delta_1_5b <- c(
  get_delta(codebleu_1_5b_st$codebleu, codebleu_1_5b_mt$codebleu),
  get_delta(lizard_1_5b_st$Lines_of_Code, lizard_1_5b_mt$Lines_of_Code),
  get_delta(lizard_1_5b_st$Token_Count, lizard_1_5b_mt$Token_Count),
  get_delta(lizard_1_5b_st$Detection_Rate, lizard_1_5b_mt$Detection_Rate),
  get_delta(lizard_1_5b_st$Cyclomatic_Complexity, lizard_1_5b_mt$Cyclomatic_Complexity),
  get_delta(roslyn_1_5b_st$Syntax_Errors, roslyn_1_5b_mt$Syntax_Errors),
  get_delta(roslyn_1_5b_st$Maintainability, roslyn_1_5b_mt$Maintainability)
)

# Get delta values for 3B
delta_3b <- c(
  get_delta(codebleu_3b_st$codebleu, codebleu_3b_mt$codebleu),
  get_delta(lizard_3b_st$Lines_of_Code, lizard_3b_mt$Lines_of_Code),
  get_delta(lizard_3b_st$Token_Count, lizard_3b_mt$Token_Count),
  get_delta(lizard_3b_st$Detection_Rate, lizard_3b_mt$Detection_Rate),
  get_delta(lizard_3b_st$Cyclomatic_Complexity, lizard_3b_mt$Cyclomatic_Complexity),
  get_delta(roslyn_3b_st$Syntax_Errors, roslyn_3b_mt$Syntax_Errors),
  get_delta(roslyn_3b_st$Maintainability, roslyn_3b_mt$Maintainability)
)

# ============== QWEN 0.5B TABLE ==============
cat("------------------------------------------------------------------------------\n")
cat("QWEN 0.5B: ST-QLoRA vs MT-QLoRA\n")
cat("------------------------------------------------------------------------------\n")
cat(sprintf("%-15s %12s %12s %12s\n", "Metric", "p-value", "Cliff's d", "Significant"))
cat("------------------------------------------------------------------------------\n")

for (i in 1:length(metric_labels)) {
  p_val <- res_df_0_5b$Wilcoxon.p[i]
  d_val <- delta_0_5b[i]
  sig <- ifelse(is.na(p_val), "NA", ifelse(p_val < 0.05, "Yes", "No"))
  p_display <- ifelse(is.na(p_val), "NA", sprintf("%.4f", p_val))
  d_display <- sprintf("%.4f", d_val)
  cat(sprintf("%-15s %12s %12s %12s\n", metric_labels[i], p_display, d_display, sig))
}
cat("\n")

# ============== QWEN 1.5B TABLE ==============
cat("------------------------------------------------------------------------------\n")
cat("QWEN 1.5B: ST-QLoRA vs MT-QLoRA\n")
cat("------------------------------------------------------------------------------\n")
cat(sprintf("%-15s %12s %12s %12s\n", "Metric", "p-value", "Cliff's d", "Significant"))
cat("------------------------------------------------------------------------------\n")

for (i in 1:length(metric_labels)) {
  p_val <- res_df_1_5b$Wilcoxon.p[i]
  d_val <- delta_1_5b[i]
  sig <- ifelse(is.na(p_val), "NA", ifelse(p_val < 0.05, "Yes", "No"))
  p_display <- ifelse(is.na(p_val), "NA", sprintf("%.4f", p_val))
  d_display <- sprintf("%.4f", d_val)
  cat(sprintf("%-15s %12s %12s %12s\n", metric_labels[i], p_display, d_display, sig))
}
cat("\n")

# ============== QWEN 3B TABLE ==============
cat("------------------------------------------------------------------------------\n")
cat("QWEN 3B: ST-QLoRA vs MT-QLoRA\n")
cat("------------------------------------------------------------------------------\n")
cat(sprintf("%-15s %12s %12s %12s\n", "Metric", "p-value", "Cliff's d", "Significant"))
cat("------------------------------------------------------------------------------\n")

for (i in 1:length(metric_labels)) {
  p_val <- res_df_3b$Wilcoxon.p[i]
  d_val <- delta_3b[i]
  sig <- ifelse(is.na(p_val), "NA", ifelse(p_val < 0.05, "Yes", "No"))
  p_display <- ifelse(is.na(p_val), "NA", sprintf("%.4f", p_val))
  d_display <- sprintf("%.4f", d_val)
  cat(sprintf("%-15s %12s %12s %12s\n", metric_labels[i], p_display, d_display, sig))
}
cat("\n")

cat("================================================================================\n")
cat("Note: p-values are Holm-Bonferroni corrected\n")
cat("Cliff's Delta interpretation: |d| < 0.147 (N), < 0.33 (S), < 0.474 (M), >= 0.474 (L)\n")
cat("================================================================================\n")

sink()

print("Results saved to:")
print(paste0(output_path, "/rq1_ct_java_cs_st_vs_mt_wilcoxon_RAW.txt"))
print(paste0(output_path, "/rq1_ct_java_cs_st_vs_mt_wilcoxon_CLEAN.txt"))