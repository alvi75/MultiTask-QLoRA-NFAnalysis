rm(list=ls())

if (!require("effsize")) install.packages("effsize")
library(effsize)

# Base path
base_path <- "/scratch/oldhome/mhaque/quantized-model-code-quality/statistical-tests/per-instance-value"
output_path <- "/scratch/oldhome/mhaque/quantized-model-code-quality/statistical-tests/R-scripts-v2"

# ============== LOAD CODEBLEU DATA ==============
# 3-task (CG+CS+CT)
codebleu_3b_3task <- read.csv(paste0(base_path, "/codebleu/translation/java_cs/ct_qwen3_mt_qlora.csv"), header=TRUE)
# 2-task combinations
codebleu_3b_cg_ct <- read.csv(paste0(base_path, "/codebleu/translation/java_cs/ct_qwen3_cg_ct_mt_qlora.csv"), header=TRUE)
codebleu_3b_cs_ct <- read.csv(paste0(base_path, "/codebleu/translation/java_cs/ct_qwen3_cs_ct_mt_qlora.csv"), header=TRUE)

# ============== LOAD LIZARD DATA ==============
# 3-task (CG+CS+CT)
lizard_3b_3task <- read.csv(paste0(base_path, "/lizard/translation/java_cs/r-lizard_ct_3b-mt-qlora.csv"), header=TRUE)
# 2-task combinations
lizard_3b_cg_ct <- read.csv(paste0(base_path, "/lizard/translation/java_cs/r-lizard_ct_3b_cg_ct-mt-qlora.csv"), header=TRUE)
lizard_3b_cs_ct <- read.csv(paste0(base_path, "/lizard/translation/java_cs/r-lizard_ct_3b_cs_ct-mt-qlora.csv"), header=TRUE)

# ============== LOAD ROSLYN DATA ==============
# 3-task (CG+CS+CT)
roslyn_3b_3task <- read.csv(paste0(base_path, "/roslyn/java_cs/r-roslyn_ct_3b-mt-qlora.csv"), header=TRUE)
# 2-task combinations
roslyn_3b_cg_ct <- read.csv(paste0(base_path, "/roslyn/java_cs/r-roslyn_ct_3b_cg_ct-mt-qlora.csv"), header=TRUE)
roslyn_3b_cs_ct <- read.csv(paste0(base_path, "/roslyn/java_cs/r-roslyn_ct_3b_cs_ct-mt-qlora.csv"), header=TRUE)

# ============== ENSURE SAME LENGTH ==============
# 3-task vs CG+CT
min_len <- min(nrow(codebleu_3b_3task), nrow(codebleu_3b_cg_ct))
codebleu_3b_3task_cgct <- codebleu_3b_3task[1:min_len, ]; codebleu_3b_cg_ct <- codebleu_3b_cg_ct[1:min_len, ]

min_len <- min(nrow(lizard_3b_3task), nrow(lizard_3b_cg_ct))
lizard_3b_3task_cgct <- lizard_3b_3task[1:min_len, ]; lizard_3b_cg_ct <- lizard_3b_cg_ct[1:min_len, ]

min_len <- min(nrow(roslyn_3b_3task), nrow(roslyn_3b_cg_ct))
roslyn_3b_3task_cgct <- roslyn_3b_3task[1:min_len, ]; roslyn_3b_cg_ct <- roslyn_3b_cg_ct[1:min_len, ]

# 3-task vs CS+CT
min_len <- min(nrow(codebleu_3b_3task), nrow(codebleu_3b_cs_ct))
codebleu_3b_3task_csct <- codebleu_3b_3task[1:min_len, ]; codebleu_3b_cs_ct <- codebleu_3b_cs_ct[1:min_len, ]

min_len <- min(nrow(lizard_3b_3task), nrow(lizard_3b_cs_ct))
lizard_3b_3task_csct <- lizard_3b_3task[1:min_len, ]; lizard_3b_cs_ct <- lizard_3b_cs_ct[1:min_len, ]

min_len <- min(nrow(roslyn_3b_3task), nrow(roslyn_3b_cs_ct))
roslyn_3b_3task_csct <- roslyn_3b_3task[1:min_len, ]; roslyn_3b_cs_ct <- roslyn_3b_cs_ct[1:min_len, ]

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
sink(paste0(output_path, "/rq3_ct_java_cs_3task_vs_2task_wilcoxon_RAW.txt"))

# ============== COMPARISON 1: 3-task vs CG+CT ==============
print("********************** QWEN 3B: MT-QLoRA (3-task) vs MT-QLoRA (CG+CT) (Java to C#) *********************************")

res_3task_vs_cgct = list(Wilcoxon.p = c())

# CODEBLEU
print("CodeBLEU - 3-task vs CG+CT (3B)")
res_3task_vs_cgct$Wilcoxon.p = append(res_3task_vs_cgct$Wilcoxon.p, wilcox_test(codebleu_3b_3task_cgct$codebleu, codebleu_3b_cg_ct$codebleu))
cliff.delta(codebleu_3b_3task_cgct$codebleu, codebleu_3b_cg_ct$codebleu)

# LIZARD
print("Lines_of_Code - 3-task vs CG+CT (3B)")
res_3task_vs_cgct$Wilcoxon.p = append(res_3task_vs_cgct$Wilcoxon.p, wilcox_test(lizard_3b_3task_cgct$Lines_of_Code, lizard_3b_cg_ct$Lines_of_Code))
cliff.delta(lizard_3b_3task_cgct$Lines_of_Code, lizard_3b_cg_ct$Lines_of_Code)

print("Token_Count - 3-task vs CG+CT (3B)")
res_3task_vs_cgct$Wilcoxon.p = append(res_3task_vs_cgct$Wilcoxon.p, wilcox_test(lizard_3b_3task_cgct$Token_Count, lizard_3b_cg_ct$Token_Count))
cliff.delta(lizard_3b_3task_cgct$Token_Count, lizard_3b_cg_ct$Token_Count)

print("Detection_Rate - 3-task vs CG+CT (3B)")
res_3task_vs_cgct$Wilcoxon.p = append(res_3task_vs_cgct$Wilcoxon.p, wilcox_test(lizard_3b_3task_cgct$Detection_Rate, lizard_3b_cg_ct$Detection_Rate))
cliff.delta(lizard_3b_3task_cgct$Detection_Rate, lizard_3b_cg_ct$Detection_Rate)

print("Cyclomatic_Complexity - 3-task vs CG+CT (3B)")
res_3task_vs_cgct$Wilcoxon.p = append(res_3task_vs_cgct$Wilcoxon.p, wilcox_test(lizard_3b_3task_cgct$Cyclomatic_Complexity, lizard_3b_cg_ct$Cyclomatic_Complexity))
cliff.delta(lizard_3b_3task_cgct$Cyclomatic_Complexity, lizard_3b_cg_ct$Cyclomatic_Complexity)

# ROSLYN
print("Syntax_Errors - 3-task vs CG+CT (3B)")
res_3task_vs_cgct$Wilcoxon.p = append(res_3task_vs_cgct$Wilcoxon.p, wilcox_test(roslyn_3b_3task_cgct$Syntax_Errors, roslyn_3b_cg_ct$Syntax_Errors))
cliff.delta(roslyn_3b_3task_cgct$Syntax_Errors, roslyn_3b_cg_ct$Syntax_Errors)

print("Maintainability - 3-task vs CG+CT (3B)")
res_3task_vs_cgct$Wilcoxon.p = append(res_3task_vs_cgct$Wilcoxon.p, wilcox_test(roslyn_3b_3task_cgct$Maintainability, roslyn_3b_cg_ct$Maintainability))
cliff.delta(roslyn_3b_3task_cgct$Maintainability, roslyn_3b_cg_ct$Maintainability)

# ============== COMPARISON 2: 3-task vs CS+CT ==============
print("********************** QWEN 3B: MT-QLoRA (3-task) vs MT-QLoRA (CS+CT) (Java to C#) *********************************")

res_3task_vs_csct = list(Wilcoxon.p = c())

# CODEBLEU
print("CodeBLEU - 3-task vs CS+CT (3B)")
res_3task_vs_csct$Wilcoxon.p = append(res_3task_vs_csct$Wilcoxon.p, wilcox_test(codebleu_3b_3task_csct$codebleu, codebleu_3b_cs_ct$codebleu))
cliff.delta(codebleu_3b_3task_csct$codebleu, codebleu_3b_cs_ct$codebleu)

# LIZARD
print("Lines_of_Code - 3-task vs CS+CT (3B)")
res_3task_vs_csct$Wilcoxon.p = append(res_3task_vs_csct$Wilcoxon.p, wilcox_test(lizard_3b_3task_csct$Lines_of_Code, lizard_3b_cs_ct$Lines_of_Code))
cliff.delta(lizard_3b_3task_csct$Lines_of_Code, lizard_3b_cs_ct$Lines_of_Code)

print("Token_Count - 3-task vs CS+CT (3B)")
res_3task_vs_csct$Wilcoxon.p = append(res_3task_vs_csct$Wilcoxon.p, wilcox_test(lizard_3b_3task_csct$Token_Count, lizard_3b_cs_ct$Token_Count))
cliff.delta(lizard_3b_3task_csct$Token_Count, lizard_3b_cs_ct$Token_Count)

print("Detection_Rate - 3-task vs CS+CT (3B)")
res_3task_vs_csct$Wilcoxon.p = append(res_3task_vs_csct$Wilcoxon.p, wilcox_test(lizard_3b_3task_csct$Detection_Rate, lizard_3b_cs_ct$Detection_Rate))
cliff.delta(lizard_3b_3task_csct$Detection_Rate, lizard_3b_cs_ct$Detection_Rate)

print("Cyclomatic_Complexity - 3-task vs CS+CT (3B)")
res_3task_vs_csct$Wilcoxon.p = append(res_3task_vs_csct$Wilcoxon.p, wilcox_test(lizard_3b_3task_csct$Cyclomatic_Complexity, lizard_3b_cs_ct$Cyclomatic_Complexity))
cliff.delta(lizard_3b_3task_csct$Cyclomatic_Complexity, lizard_3b_cs_ct$Cyclomatic_Complexity)

# ROSLYN
print("Syntax_Errors - 3-task vs CS+CT (3B)")
res_3task_vs_csct$Wilcoxon.p = append(res_3task_vs_csct$Wilcoxon.p, wilcox_test(roslyn_3b_3task_csct$Syntax_Errors, roslyn_3b_cs_ct$Syntax_Errors))
cliff.delta(roslyn_3b_3task_csct$Syntax_Errors, roslyn_3b_cs_ct$Syntax_Errors)

print("Maintainability - 3-task vs CS+CT (3B)")
res_3task_vs_csct$Wilcoxon.p = append(res_3task_vs_csct$Wilcoxon.p, wilcox_test(roslyn_3b_3task_csct$Maintainability, roslyn_3b_cs_ct$Maintainability))
cliff.delta(roslyn_3b_3task_csct$Maintainability, roslyn_3b_cs_ct$Maintainability)

# ============== PRINT RAW RESULTS ==============
print("***************************************************************************")

res_df_3task_vs_cgct = data.frame(Wilcoxon.p = res_3task_vs_cgct$Wilcoxon.p)
res_df_3task_vs_cgct$Wilcoxon.p = p.adjust(res_df_3task_vs_cgct$Wilcoxon.p, method="holm")
print("3-task vs CG+CT Results (Adjusted P-values):")
print(res_df_3task_vs_cgct)

res_df_3task_vs_csct = data.frame(Wilcoxon.p = res_3task_vs_csct$Wilcoxon.p)
res_df_3task_vs_csct$Wilcoxon.p = p.adjust(res_df_3task_vs_csct$Wilcoxon.p, method="holm")
print("3-task vs CS+CT Results (Adjusted P-values):")
print(res_df_3task_vs_csct)

sink()

# ============================================================================
# FILE 2: CLEAN SORTED OUTPUT (table format)
# ============================================================================
sink(paste0(output_path, "/rq3_ct_java_cs_3task_vs_2task_wilcoxon_CLEAN.txt"))

cat("================================================================================\n")
cat("RQ3: Code Translation - Java to C# (3-task vs 2-task) - Wilcoxon Signed-Rank Test\n")
cat("================================================================================\n\n")

# Metric labels (CodeBLEU + Lizard + Roslyn)
metric_labels <- c("CodeBLEU",
                   "Lizard_LoC", "Lizard_ToK", "Lizard_DR", "Lizard_CyC",
                   "Roslyn_SynErr", "Roslyn_Main")

# Get delta values for 3-task vs CG+CT
delta_3task_vs_cgct <- c(
  get_delta(codebleu_3b_3task_cgct$codebleu, codebleu_3b_cg_ct$codebleu),
  get_delta(lizard_3b_3task_cgct$Lines_of_Code, lizard_3b_cg_ct$Lines_of_Code),
  get_delta(lizard_3b_3task_cgct$Token_Count, lizard_3b_cg_ct$Token_Count),
  get_delta(lizard_3b_3task_cgct$Detection_Rate, lizard_3b_cg_ct$Detection_Rate),
  get_delta(lizard_3b_3task_cgct$Cyclomatic_Complexity, lizard_3b_cg_ct$Cyclomatic_Complexity),
  get_delta(roslyn_3b_3task_cgct$Syntax_Errors, roslyn_3b_cg_ct$Syntax_Errors),
  get_delta(roslyn_3b_3task_cgct$Maintainability, roslyn_3b_cg_ct$Maintainability)
)

# Get delta values for 3-task vs CS+CT
delta_3task_vs_csct <- c(
  get_delta(codebleu_3b_3task_csct$codebleu, codebleu_3b_cs_ct$codebleu),
  get_delta(lizard_3b_3task_csct$Lines_of_Code, lizard_3b_cs_ct$Lines_of_Code),
  get_delta(lizard_3b_3task_csct$Token_Count, lizard_3b_cs_ct$Token_Count),
  get_delta(lizard_3b_3task_csct$Detection_Rate, lizard_3b_cs_ct$Detection_Rate),
  get_delta(lizard_3b_3task_csct$Cyclomatic_Complexity, lizard_3b_cs_ct$Cyclomatic_Complexity),
  get_delta(roslyn_3b_3task_csct$Syntax_Errors, roslyn_3b_cs_ct$Syntax_Errors),
  get_delta(roslyn_3b_3task_csct$Maintainability, roslyn_3b_cs_ct$Maintainability)
)

# ============== 3-task vs CG+CT TABLE ==============
cat("------------------------------------------------------------------------------\n")
cat("QWEN 3B: MT-QLoRA (3-task) vs MT-QLoRA (CG+CT)\n")
cat("------------------------------------------------------------------------------\n")
cat(sprintf("%-15s %12s %12s %12s\n", "Metric", "p-value", "Cliff's d", "Significant"))
cat("------------------------------------------------------------------------------\n")

for (i in 1:length(metric_labels)) {
  p_val <- res_df_3task_vs_cgct$Wilcoxon.p[i]
  d_val <- delta_3task_vs_cgct[i]
  sig <- ifelse(is.na(p_val), "NA", ifelse(p_val < 0.05, "Yes", "No"))
  p_display <- ifelse(is.na(p_val), "NA", sprintf("%.4f", p_val))
  d_display <- sprintf("%.4f", d_val)
  cat(sprintf("%-15s %12s %12s %12s\n", metric_labels[i], p_display, d_display, sig))
}
cat("\n")

# ============== 3-task vs CS+CT TABLE ==============
cat("------------------------------------------------------------------------------\n")
cat("QWEN 3B: MT-QLoRA (3-task) vs MT-QLoRA (CS+CT)\n")
cat("------------------------------------------------------------------------------\n")
cat(sprintf("%-15s %12s %12s %12s\n", "Metric", "p-value", "Cliff's d", "Significant"))
cat("------------------------------------------------------------------------------\n")

for (i in 1:length(metric_labels)) {
  p_val <- res_df_3task_vs_csct$Wilcoxon.p[i]
  d_val <- delta_3task_vs_csct[i]
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
print(paste0(output_path, "/rq3_ct_java_cs_3task_vs_2task_wilcoxon_RAW.txt"))
print(paste0(output_path, "/rq3_ct_java_cs_3task_vs_2task_wilcoxon_CLEAN.txt"))