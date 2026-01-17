rm(list=ls())

if (!require("effsize")) install.packages("effsize")
library(effsize)

# Base path
base_path <- "/scratch/oldhome/mhaque/quantized-model-code-quality/statistical-tests/per-instance-value"
output_path <- "/scratch/oldhome/mhaque/quantized-model-code-quality/statistical-tests/R-scripts-v2"

# ============== LOAD LIZARD DATA ==============
# 3-task (CG+CS+CT) - using 3B
lizard_3b_3task <- read.csv(paste0(base_path, "/lizard/generation/java/r-lizard_cg_3b-mt-qlora.csv"), header=TRUE)
# 2-task combinations
lizard_3b_cg_cs <- read.csv(paste0(base_path, "/lizard/generation/java/r-lizard_cg_3b_cg_cs-mt-qlora.csv"), header=TRUE)
lizard_3b_cg_ct <- read.csv(paste0(base_path, "/lizard/generation/java/r-lizard_cg_3b_cg_ct-mt-qlora.csv"), header=TRUE)

# ============== LOAD PMD DATA ==============
# 3-task (CG+CS+CT) - using 3B
pmd_3b_3task <- read.csv(paste0(base_path, "/pmd/generation/r-pmd_cg_3b-mt-qlora.csv"), header=TRUE)
# 2-task combinations
pmd_3b_cg_cs <- read.csv(paste0(base_path, "/pmd/generation/r-pmd_cg_3b_cg_cs-mt-qlora.csv"), header=TRUE)
pmd_3b_cg_ct <- read.csv(paste0(base_path, "/pmd/generation/r-pmd_cg_3b_cg_ct-mt-qlora.csv"), header=TRUE)

# ============== LOAD SONARCLOUD DATA ==============
# 3-task (CG+CS+CT) - using 3B
sonar_3b_3task <- read.csv(paste0(base_path, "/sonarcloud/generation/java/r-sonarcloud_cg_qwen3_mt_qlora.csv"), header=TRUE)
# 2-task combinations
sonar_3b_cg_cs <- read.csv(paste0(base_path, "/sonarcloud/generation/java/r-sonarcloud_cg_qwen3_cg_cs_mt_qlora.csv"), header=TRUE)
sonar_3b_cg_ct <- read.csv(paste0(base_path, "/sonarcloud/generation/java/r-sonarcloud_cg_qwen3_cg_ct_mt_qlora.csv"), header=TRUE)

# ============== ENSURE SAME LENGTH ==============
# 3-task vs CG+CS
min_len <- min(nrow(lizard_3b_3task), nrow(lizard_3b_cg_cs))
lizard_3b_3task_cs <- lizard_3b_3task[1:min_len, ]; lizard_3b_cg_cs <- lizard_3b_cg_cs[1:min_len, ]

min_len <- min(nrow(pmd_3b_3task), nrow(pmd_3b_cg_cs))
pmd_3b_3task_cs <- pmd_3b_3task[1:min_len, ]; pmd_3b_cg_cs <- pmd_3b_cg_cs[1:min_len, ]

min_len <- min(nrow(sonar_3b_3task), nrow(sonar_3b_cg_cs))
sonar_3b_3task_cs <- sonar_3b_3task[1:min_len, ]; sonar_3b_cg_cs <- sonar_3b_cg_cs[1:min_len, ]

# 3-task vs CG+CT
min_len <- min(nrow(lizard_3b_3task), nrow(lizard_3b_cg_ct))
lizard_3b_3task_ct <- lizard_3b_3task[1:min_len, ]; lizard_3b_cg_ct <- lizard_3b_cg_ct[1:min_len, ]

min_len <- min(nrow(pmd_3b_3task), nrow(pmd_3b_cg_ct))
pmd_3b_3task_ct <- pmd_3b_3task[1:min_len, ]; pmd_3b_cg_ct <- pmd_3b_cg_ct[1:min_len, ]

min_len <- min(nrow(sonar_3b_3task), nrow(sonar_3b_cg_ct))
sonar_3b_3task_ct <- sonar_3b_3task[1:min_len, ]; sonar_3b_cg_ct <- sonar_3b_cg_ct[1:min_len, ]

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
sink(paste0(output_path, "/rq3_cg_java_3task_vs_2task_wilcoxon_RAW.txt"))

# ============== COMPARISON 1: 3-task vs CG+CS ==============
print("********************** QWEN 3B: MT-QLoRA (3-task) vs MT-QLoRA (CG+CS) (Java) *********************************")

res_3task_vs_cgcs = list(Wilcoxon.p = c())

# LIZARD
print("Lines_of_Code - 3-task vs CG+CS (3B)")
res_3task_vs_cgcs$Wilcoxon.p = append(res_3task_vs_cgcs$Wilcoxon.p, wilcox_test(lizard_3b_3task_cs$Lines_of_Code, lizard_3b_cg_cs$Lines_of_Code))
cliff.delta(lizard_3b_3task_cs$Lines_of_Code, lizard_3b_cg_cs$Lines_of_Code)

print("Token_Count - 3-task vs CG+CS (3B)")
res_3task_vs_cgcs$Wilcoxon.p = append(res_3task_vs_cgcs$Wilcoxon.p, wilcox_test(lizard_3b_3task_cs$Token_Count, lizard_3b_cg_cs$Token_Count))
cliff.delta(lizard_3b_3task_cs$Token_Count, lizard_3b_cg_cs$Token_Count)

print("Detection_Rate - 3-task vs CG+CS (3B)")
res_3task_vs_cgcs$Wilcoxon.p = append(res_3task_vs_cgcs$Wilcoxon.p, wilcox_test(lizard_3b_3task_cs$Detection_Rate, lizard_3b_cg_cs$Detection_Rate))
cliff.delta(lizard_3b_3task_cs$Detection_Rate, lizard_3b_cg_cs$Detection_Rate)

print("Cyclomatic_Complexity - 3-task vs CG+CS (3B)")
res_3task_vs_cgcs$Wilcoxon.p = append(res_3task_vs_cgcs$Wilcoxon.p, wilcox_test(lizard_3b_3task_cs$Cyclomatic_Complexity, lizard_3b_cg_cs$Cyclomatic_Complexity))
cliff.delta(lizard_3b_3task_cs$Cyclomatic_Complexity, lizard_3b_cg_cs$Cyclomatic_Complexity)

# PMD
print("bestpractices - 3-task vs CG+CS (3B)")
res_3task_vs_cgcs$Wilcoxon.p = append(res_3task_vs_cgcs$Wilcoxon.p, wilcox_test(pmd_3b_3task_cs$bestpractices, pmd_3b_cg_cs$bestpractices))
cliff.delta(pmd_3b_3task_cs$bestpractices, pmd_3b_cg_cs$bestpractices)

print("codestyle - 3-task vs CG+CS (3B)")
res_3task_vs_cgcs$Wilcoxon.p = append(res_3task_vs_cgcs$Wilcoxon.p, wilcox_test(pmd_3b_3task_cs$codestyle, pmd_3b_cg_cs$codestyle))
cliff.delta(pmd_3b_3task_cs$codestyle, pmd_3b_cg_cs$codestyle)

print("design - 3-task vs CG+CS (3B)")
res_3task_vs_cgcs$Wilcoxon.p = append(res_3task_vs_cgcs$Wilcoxon.p, wilcox_test(pmd_3b_3task_cs$design, pmd_3b_cg_cs$design))
cliff.delta(pmd_3b_3task_cs$design, pmd_3b_cg_cs$design)

print("errorprone - 3-task vs CG+CS (3B)")
res_3task_vs_cgcs$Wilcoxon.p = append(res_3task_vs_cgcs$Wilcoxon.p, wilcox_test(pmd_3b_3task_cs$errorprone, pmd_3b_cg_cs$errorprone))
cliff.delta(pmd_3b_3task_cs$errorprone, pmd_3b_cg_cs$errorprone)

print("multithreading - 3-task vs CG+CS (3B)")
res_3task_vs_cgcs$Wilcoxon.p = append(res_3task_vs_cgcs$Wilcoxon.p, wilcox_test(pmd_3b_3task_cs$multithreading, pmd_3b_cg_cs$multithreading))
cliff.delta(pmd_3b_3task_cs$multithreading, pmd_3b_cg_cs$multithreading)

print("performance - 3-task vs CG+CS (3B)")
res_3task_vs_cgcs$Wilcoxon.p = append(res_3task_vs_cgcs$Wilcoxon.p, wilcox_test(pmd_3b_3task_cs$performance, pmd_3b_cg_cs$performance))
cliff.delta(pmd_3b_3task_cs$performance, pmd_3b_cg_cs$performance)

# SONARCLOUD
print("Security_Hotspots - 3-task vs CG+CS (3B)")
res_3task_vs_cgcs$Wilcoxon.p = append(res_3task_vs_cgcs$Wilcoxon.p, wilcox_test(sonar_3b_3task_cs$Security_Hotspots, sonar_3b_cg_cs$Security_Hotspots))
cliff.delta(sonar_3b_3task_cs$Security_Hotspots, sonar_3b_cg_cs$Security_Hotspots)

print("Reliability - 3-task vs CG+CS (3B)")
res_3task_vs_cgcs$Wilcoxon.p = append(res_3task_vs_cgcs$Wilcoxon.p, wilcox_test(sonar_3b_3task_cs$Reliability, sonar_3b_cg_cs$Reliability))
cliff.delta(sonar_3b_3task_cs$Reliability, sonar_3b_cg_cs$Reliability)

print("Maintainability - 3-task vs CG+CS (3B)")
res_3task_vs_cgcs$Wilcoxon.p = append(res_3task_vs_cgcs$Wilcoxon.p, wilcox_test(sonar_3b_3task_cs$Maintainability, sonar_3b_cg_cs$Maintainability))
cliff.delta(sonar_3b_3task_cs$Maintainability, sonar_3b_cg_cs$Maintainability)

print("CyC - 3-task vs CG+CS (3B)")
res_3task_vs_cgcs$Wilcoxon.p = append(res_3task_vs_cgcs$Wilcoxon.p, wilcox_test(sonar_3b_3task_cs$CyC, sonar_3b_cg_cs$CyC))
cliff.delta(sonar_3b_3task_cs$CyC, sonar_3b_cg_cs$CyC)

print("CoC - 3-task vs CG+CS (3B)")
res_3task_vs_cgcs$Wilcoxon.p = append(res_3task_vs_cgcs$Wilcoxon.p, wilcox_test(sonar_3b_3task_cs$CoC, sonar_3b_cg_cs$CoC))
cliff.delta(sonar_3b_3task_cs$CoC, sonar_3b_cg_cs$CoC)

# ============== COMPARISON 2: 3-task vs CG+CT ==============
print("********************** QWEN 3B: MT-QLoRA (3-task) vs MT-QLoRA (CG+CT) (Java) *********************************")

res_3task_vs_cgct = list(Wilcoxon.p = c())

# LIZARD
print("Lines_of_Code - 3-task vs CG+CT (3B)")
res_3task_vs_cgct$Wilcoxon.p = append(res_3task_vs_cgct$Wilcoxon.p, wilcox_test(lizard_3b_3task_ct$Lines_of_Code, lizard_3b_cg_ct$Lines_of_Code))
cliff.delta(lizard_3b_3task_ct$Lines_of_Code, lizard_3b_cg_ct$Lines_of_Code)

print("Token_Count - 3-task vs CG+CT (3B)")
res_3task_vs_cgct$Wilcoxon.p = append(res_3task_vs_cgct$Wilcoxon.p, wilcox_test(lizard_3b_3task_ct$Token_Count, lizard_3b_cg_ct$Token_Count))
cliff.delta(lizard_3b_3task_ct$Token_Count, lizard_3b_cg_ct$Token_Count)

print("Detection_Rate - 3-task vs CG+CT (3B)")
res_3task_vs_cgct$Wilcoxon.p = append(res_3task_vs_cgct$Wilcoxon.p, wilcox_test(lizard_3b_3task_ct$Detection_Rate, lizard_3b_cg_ct$Detection_Rate))
cliff.delta(lizard_3b_3task_ct$Detection_Rate, lizard_3b_cg_ct$Detection_Rate)

print("Cyclomatic_Complexity - 3-task vs CG+CT (3B)")
res_3task_vs_cgct$Wilcoxon.p = append(res_3task_vs_cgct$Wilcoxon.p, wilcox_test(lizard_3b_3task_ct$Cyclomatic_Complexity, lizard_3b_cg_ct$Cyclomatic_Complexity))
cliff.delta(lizard_3b_3task_ct$Cyclomatic_Complexity, lizard_3b_cg_ct$Cyclomatic_Complexity)

# PMD
print("bestpractices - 3-task vs CG+CT (3B)")
res_3task_vs_cgct$Wilcoxon.p = append(res_3task_vs_cgct$Wilcoxon.p, wilcox_test(pmd_3b_3task_ct$bestpractices, pmd_3b_cg_ct$bestpractices))
cliff.delta(pmd_3b_3task_ct$bestpractices, pmd_3b_cg_ct$bestpractices)

print("codestyle - 3-task vs CG+CT (3B)")
res_3task_vs_cgct$Wilcoxon.p = append(res_3task_vs_cgct$Wilcoxon.p, wilcox_test(pmd_3b_3task_ct$codestyle, pmd_3b_cg_ct$codestyle))
cliff.delta(pmd_3b_3task_ct$codestyle, pmd_3b_cg_ct$codestyle)

print("design - 3-task vs CG+CT (3B)")
res_3task_vs_cgct$Wilcoxon.p = append(res_3task_vs_cgct$Wilcoxon.p, wilcox_test(pmd_3b_3task_ct$design, pmd_3b_cg_ct$design))
cliff.delta(pmd_3b_3task_ct$design, pmd_3b_cg_ct$design)

print("errorprone - 3-task vs CG+CT (3B)")
res_3task_vs_cgct$Wilcoxon.p = append(res_3task_vs_cgct$Wilcoxon.p, wilcox_test(pmd_3b_3task_ct$errorprone, pmd_3b_cg_ct$errorprone))
cliff.delta(pmd_3b_3task_ct$errorprone, pmd_3b_cg_ct$errorprone)

print("multithreading - 3-task vs CG+CT (3B)")
res_3task_vs_cgct$Wilcoxon.p = append(res_3task_vs_cgct$Wilcoxon.p, wilcox_test(pmd_3b_3task_ct$multithreading, pmd_3b_cg_ct$multithreading))
cliff.delta(pmd_3b_3task_ct$multithreading, pmd_3b_cg_ct$multithreading)

print("performance - 3-task vs CG+CT (3B)")
res_3task_vs_cgct$Wilcoxon.p = append(res_3task_vs_cgct$Wilcoxon.p, wilcox_test(pmd_3b_3task_ct$performance, pmd_3b_cg_ct$performance))
cliff.delta(pmd_3b_3task_ct$performance, pmd_3b_cg_ct$performance)

# SONARCLOUD
print("Security_Hotspots - 3-task vs CG+CT (3B)")
res_3task_vs_cgct$Wilcoxon.p = append(res_3task_vs_cgct$Wilcoxon.p, wilcox_test(sonar_3b_3task_ct$Security_Hotspots, sonar_3b_cg_ct$Security_Hotspots))
cliff.delta(sonar_3b_3task_ct$Security_Hotspots, sonar_3b_cg_ct$Security_Hotspots)

print("Reliability - 3-task vs CG+CT (3B)")
res_3task_vs_cgct$Wilcoxon.p = append(res_3task_vs_cgct$Wilcoxon.p, wilcox_test(sonar_3b_3task_ct$Reliability, sonar_3b_cg_ct$Reliability))
cliff.delta(sonar_3b_3task_ct$Reliability, sonar_3b_cg_ct$Reliability)

print("Maintainability - 3-task vs CG+CT (3B)")
res_3task_vs_cgct$Wilcoxon.p = append(res_3task_vs_cgct$Wilcoxon.p, wilcox_test(sonar_3b_3task_ct$Maintainability, sonar_3b_cg_ct$Maintainability))
cliff.delta(sonar_3b_3task_ct$Maintainability, sonar_3b_cg_ct$Maintainability)

print("CyC - 3-task vs CG+CT (3B)")
res_3task_vs_cgct$Wilcoxon.p = append(res_3task_vs_cgct$Wilcoxon.p, wilcox_test(sonar_3b_3task_ct$CyC, sonar_3b_cg_ct$CyC))
cliff.delta(sonar_3b_3task_ct$CyC, sonar_3b_cg_ct$CyC)

print("CoC - 3-task vs CG+CT (3B)")
res_3task_vs_cgct$Wilcoxon.p = append(res_3task_vs_cgct$Wilcoxon.p, wilcox_test(sonar_3b_3task_ct$CoC, sonar_3b_cg_ct$CoC))
cliff.delta(sonar_3b_3task_ct$CoC, sonar_3b_cg_ct$CoC)

# ============== PRINT RAW RESULTS ==============
print("***************************************************************************")

res_df_3task_vs_cgcs = data.frame(Wilcoxon.p = res_3task_vs_cgcs$Wilcoxon.p)
res_df_3task_vs_cgcs$Wilcoxon.p = p.adjust(res_df_3task_vs_cgcs$Wilcoxon.p, method="holm")
print("3-task vs CG+CS Results (Adjusted P-values):")
print(res_df_3task_vs_cgcs)

res_df_3task_vs_cgct = data.frame(Wilcoxon.p = res_3task_vs_cgct$Wilcoxon.p)
res_df_3task_vs_cgct$Wilcoxon.p = p.adjust(res_df_3task_vs_cgct$Wilcoxon.p, method="holm")
print("3-task vs CG+CT Results (Adjusted P-values):")
print(res_df_3task_vs_cgct)

sink()

# ============================================================================
# FILE 2: CLEAN SORTED OUTPUT (table format)
# ============================================================================
sink(paste0(output_path, "/rq3_cg_java_3task_vs_2task_wilcoxon_CLEAN.txt"))

cat("================================================================================\n")
cat("RQ3: Code Generation - Java (3-task vs 2-task) - Wilcoxon Signed-Rank Test\n")
cat("================================================================================\n\n")

# Metric labels (Lizard + PMD + SonarCloud)
metric_labels <- c("Lizard_LoC", "Lizard_ToK", "Lizard_DR", "Lizard_CyC",
                   "PMD_BestPrac", "PMD_CS", "PMD_Design", "PMD_EP", "PMD_MT", "PMD_Perf",
                   "Sonar_Sec", "Sonar_Rel", "Sonar_Main", "Sonar_CyC", "Sonar_CoC")

# Get delta values for 3-task vs CG+CS
delta_3task_vs_cgcs <- c(
  get_delta(lizard_3b_3task_cs$Lines_of_Code, lizard_3b_cg_cs$Lines_of_Code),
  get_delta(lizard_3b_3task_cs$Token_Count, lizard_3b_cg_cs$Token_Count),
  get_delta(lizard_3b_3task_cs$Detection_Rate, lizard_3b_cg_cs$Detection_Rate),
  get_delta(lizard_3b_3task_cs$Cyclomatic_Complexity, lizard_3b_cg_cs$Cyclomatic_Complexity),
  get_delta(pmd_3b_3task_cs$bestpractices, pmd_3b_cg_cs$bestpractices),
  get_delta(pmd_3b_3task_cs$codestyle, pmd_3b_cg_cs$codestyle),
  get_delta(pmd_3b_3task_cs$design, pmd_3b_cg_cs$design),
  get_delta(pmd_3b_3task_cs$errorprone, pmd_3b_cg_cs$errorprone),
  get_delta(pmd_3b_3task_cs$multithreading, pmd_3b_cg_cs$multithreading),
  get_delta(pmd_3b_3task_cs$performance, pmd_3b_cg_cs$performance),
  get_delta(sonar_3b_3task_cs$Security_Hotspots, sonar_3b_cg_cs$Security_Hotspots),
  get_delta(sonar_3b_3task_cs$Reliability, sonar_3b_cg_cs$Reliability),
  get_delta(sonar_3b_3task_cs$Maintainability, sonar_3b_cg_cs$Maintainability),
  get_delta(sonar_3b_3task_cs$CyC, sonar_3b_cg_cs$CyC),
  get_delta(sonar_3b_3task_cs$CoC, sonar_3b_cg_cs$CoC)
)

# Get delta values for 3-task vs CG+CT
delta_3task_vs_cgct <- c(
  get_delta(lizard_3b_3task_ct$Lines_of_Code, lizard_3b_cg_ct$Lines_of_Code),
  get_delta(lizard_3b_3task_ct$Token_Count, lizard_3b_cg_ct$Token_Count),
  get_delta(lizard_3b_3task_ct$Detection_Rate, lizard_3b_cg_ct$Detection_Rate),
  get_delta(lizard_3b_3task_ct$Cyclomatic_Complexity, lizard_3b_cg_ct$Cyclomatic_Complexity),
  get_delta(pmd_3b_3task_ct$bestpractices, pmd_3b_cg_ct$bestpractices),
  get_delta(pmd_3b_3task_ct$codestyle, pmd_3b_cg_ct$codestyle),
  get_delta(pmd_3b_3task_ct$design, pmd_3b_cg_ct$design),
  get_delta(pmd_3b_3task_ct$errorprone, pmd_3b_cg_ct$errorprone),
  get_delta(pmd_3b_3task_ct$multithreading, pmd_3b_cg_ct$multithreading),
  get_delta(pmd_3b_3task_ct$performance, pmd_3b_cg_ct$performance),
  get_delta(sonar_3b_3task_ct$Security_Hotspots, sonar_3b_cg_ct$Security_Hotspots),
  get_delta(sonar_3b_3task_ct$Reliability, sonar_3b_cg_ct$Reliability),
  get_delta(sonar_3b_3task_ct$Maintainability, sonar_3b_cg_ct$Maintainability),
  get_delta(sonar_3b_3task_ct$CyC, sonar_3b_cg_ct$CyC),
  get_delta(sonar_3b_3task_ct$CoC, sonar_3b_cg_ct$CoC)
)

# ============== 3-task vs CG+CS TABLE ==============
cat("------------------------------------------------------------------------------\n")
cat("QWEN 3B: MT-QLoRA (3-task) vs MT-QLoRA (CG+CS)\n")
cat("------------------------------------------------------------------------------\n")
cat(sprintf("%-15s %12s %12s %12s\n", "Metric", "p-value", "Cliff's d", "Significant"))
cat("------------------------------------------------------------------------------\n")

for (i in 1:length(metric_labels)) {
  p_val <- res_df_3task_vs_cgcs$Wilcoxon.p[i]
  d_val <- delta_3task_vs_cgcs[i]
  sig <- ifelse(is.na(p_val), "NA", ifelse(p_val < 0.05, "Yes", "No"))
  p_display <- ifelse(is.na(p_val), "NA", sprintf("%.4f", p_val))
  d_display <- sprintf("%.4f", d_val)
  cat(sprintf("%-15s %12s %12s %12s\n", metric_labels[i], p_display, d_display, sig))
}
cat("\n")

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

cat("================================================================================\n")
cat("Note: p-values are Holm-Bonferroni corrected\n")
cat("Cliff's Delta interpretation: |d| < 0.147 (N), < 0.33 (S), < 0.474 (M), >= 0.474 (L)\n")
cat("================================================================================\n")

sink()

print("Results saved to:")
print(paste0(output_path, "/rq3_cg_java_3task_vs_2task_wilcoxon_RAW.txt"))
print(paste0(output_path, "/rq3_cg_java_3task_vs_2task_wilcoxon_CLEAN.txt"))