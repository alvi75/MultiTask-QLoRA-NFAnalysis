rm(list=ls())

if (!require("effsize")) install.packages("effsize")
library(effsize)

# Base path
base_path <- "/scratch/oldhome/mhaque/quantized-model-code-quality/statistical-tests/per-instance-value"
output_path <- "/scratch/oldhome/mhaque/quantized-model-code-quality/statistical-tests/R-scripts-v2"

# ============== LOAD CODEBLEU DATA ==============
# 3-task (CG+CS+CT)
codebleu_3b_3task <- read.csv(paste0(base_path, "/codebleu/translation/cs_java/ct_qwen3_mt_qlora.csv"), header=TRUE)
# 2-task combinations
codebleu_3b_cg_ct <- read.csv(paste0(base_path, "/codebleu/translation/cs_java/ct_qwen3_cg_ct_mt_qlora.csv"), header=TRUE)
codebleu_3b_cs_ct <- read.csv(paste0(base_path, "/codebleu/translation/cs_java/ct_qwen3_cs_ct_mt_qlora.csv"), header=TRUE)

# ============== LOAD LIZARD DATA ==============
# 3-task (CG+CS+CT)
lizard_3b_3task <- read.csv(paste0(base_path, "/lizard/translation/cs_java/r-lizard_ct_3b-mt-qlora.csv"), header=TRUE)
# 2-task combinations
lizard_3b_cg_ct <- read.csv(paste0(base_path, "/lizard/translation/cs_java/r-lizard_ct_3b_cg_ct-mt-qlora.csv"), header=TRUE)
lizard_3b_cs_ct <- read.csv(paste0(base_path, "/lizard/translation/cs_java/r-lizard_ct_3b_cs_ct-mt-qlora.csv"), header=TRUE)

# ============== LOAD PMD DATA ==============
# 3-task (CG+CS+CT)
pmd_3b_3task <- read.csv(paste0(base_path, "/pmd/translation/cs_java/r-pmd_ct_3b-mt-qlora.csv"), header=TRUE)
# 2-task combinations
pmd_3b_cg_ct <- read.csv(paste0(base_path, "/pmd/translation/cs_java/r-pmd_ct_3b_cg_ct-mt-qlora.csv"), header=TRUE)
pmd_3b_cs_ct <- read.csv(paste0(base_path, "/pmd/translation/cs_java/r-pmd_ct_3b_cg_cs-mt-qlora.csv"), header=TRUE)  # Note: file says cg_cs

# ============== LOAD SONARCLOUD DATA ==============
# 3-task (CG+CS+CT)
sonar_3b_3task <- read.csv(paste0(base_path, "/sonarcloud/translation/cs_java/r-sonarcloud_ct_qwen3_mt_qlora.csv"), header=TRUE)
# 2-task combinations
sonar_3b_cg_ct <- read.csv(paste0(base_path, "/sonarcloud/translation/cs_java/r-sonarcloud_ct_qwen3_cg_ct_mt_qlora.csv"), header=TRUE)
sonar_3b_cs_ct <- read.csv(paste0(base_path, "/sonarcloud/translation/cs_java/r-sonarcloud_ct_qwen3_cs_ct_mt_qlora.csv"), header=TRUE)

# ============== ENSURE SAME LENGTH ==============
# 3-task vs CG+CT
min_len <- min(nrow(codebleu_3b_3task), nrow(codebleu_3b_cg_ct))
codebleu_3b_3task_cgct <- codebleu_3b_3task[1:min_len, ]; codebleu_3b_cg_ct <- codebleu_3b_cg_ct[1:min_len, ]

min_len <- min(nrow(lizard_3b_3task), nrow(lizard_3b_cg_ct))
lizard_3b_3task_cgct <- lizard_3b_3task[1:min_len, ]; lizard_3b_cg_ct <- lizard_3b_cg_ct[1:min_len, ]

min_len <- min(nrow(pmd_3b_3task), nrow(pmd_3b_cg_ct))
pmd_3b_3task_cgct <- pmd_3b_3task[1:min_len, ]; pmd_3b_cg_ct <- pmd_3b_cg_ct[1:min_len, ]

min_len <- min(nrow(sonar_3b_3task), nrow(sonar_3b_cg_ct))
sonar_3b_3task_cgct <- sonar_3b_3task[1:min_len, ]; sonar_3b_cg_ct <- sonar_3b_cg_ct[1:min_len, ]

# 3-task vs CS+CT
min_len <- min(nrow(codebleu_3b_3task), nrow(codebleu_3b_cs_ct))
codebleu_3b_3task_csct <- codebleu_3b_3task[1:min_len, ]; codebleu_3b_cs_ct <- codebleu_3b_cs_ct[1:min_len, ]

min_len <- min(nrow(lizard_3b_3task), nrow(lizard_3b_cs_ct))
lizard_3b_3task_csct <- lizard_3b_3task[1:min_len, ]; lizard_3b_cs_ct <- lizard_3b_cs_ct[1:min_len, ]

min_len <- min(nrow(pmd_3b_3task), nrow(pmd_3b_cs_ct))
pmd_3b_3task_csct <- pmd_3b_3task[1:min_len, ]; pmd_3b_cs_ct <- pmd_3b_cs_ct[1:min_len, ]

min_len <- min(nrow(sonar_3b_3task), nrow(sonar_3b_cs_ct))
sonar_3b_3task_csct <- sonar_3b_3task[1:min_len, ]; sonar_3b_cs_ct <- sonar_3b_cs_ct[1:min_len, ]

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
sink(paste0(output_path, "/rq3_ct_cs_java_3task_vs_2task_wilcoxon_RAW.txt"))

# ============== COMPARISON 1: 3-task vs CG+CT ==============
print("********************** QWEN 3B: MT-QLoRA (3-task) vs MT-QLoRA (CG+CT) (C# to Java) *********************************")

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

# PMD
print("bestpractices - 3-task vs CG+CT (3B)")
res_3task_vs_cgct$Wilcoxon.p = append(res_3task_vs_cgct$Wilcoxon.p, wilcox_test(pmd_3b_3task_cgct$bestpractices, pmd_3b_cg_ct$bestpractices))
cliff.delta(pmd_3b_3task_cgct$bestpractices, pmd_3b_cg_ct$bestpractices)

print("codestyle - 3-task vs CG+CT (3B)")
res_3task_vs_cgct$Wilcoxon.p = append(res_3task_vs_cgct$Wilcoxon.p, wilcox_test(pmd_3b_3task_cgct$codestyle, pmd_3b_cg_ct$codestyle))
cliff.delta(pmd_3b_3task_cgct$codestyle, pmd_3b_cg_ct$codestyle)

print("design - 3-task vs CG+CT (3B)")
res_3task_vs_cgct$Wilcoxon.p = append(res_3task_vs_cgct$Wilcoxon.p, wilcox_test(pmd_3b_3task_cgct$design, pmd_3b_cg_ct$design))
cliff.delta(pmd_3b_3task_cgct$design, pmd_3b_cg_ct$design)

print("errorprone - 3-task vs CG+CT (3B)")
res_3task_vs_cgct$Wilcoxon.p = append(res_3task_vs_cgct$Wilcoxon.p, wilcox_test(pmd_3b_3task_cgct$errorprone, pmd_3b_cg_ct$errorprone))
cliff.delta(pmd_3b_3task_cgct$errorprone, pmd_3b_cg_ct$errorprone)

print("multithreading - 3-task vs CG+CT (3B)")
res_3task_vs_cgct$Wilcoxon.p = append(res_3task_vs_cgct$Wilcoxon.p, wilcox_test(pmd_3b_3task_cgct$multithreading, pmd_3b_cg_ct$multithreading))
cliff.delta(pmd_3b_3task_cgct$multithreading, pmd_3b_cg_ct$multithreading)

print("performance - 3-task vs CG+CT (3B)")
res_3task_vs_cgct$Wilcoxon.p = append(res_3task_vs_cgct$Wilcoxon.p, wilcox_test(pmd_3b_3task_cgct$performance, pmd_3b_cg_ct$performance))
cliff.delta(pmd_3b_3task_cgct$performance, pmd_3b_cg_ct$performance)

# SONARCLOUD
print("Security_Hotspots - 3-task vs CG+CT (3B)")
res_3task_vs_cgct$Wilcoxon.p = append(res_3task_vs_cgct$Wilcoxon.p, wilcox_test(sonar_3b_3task_cgct$Security_Hotspots, sonar_3b_cg_ct$Security_Hotspots))
cliff.delta(sonar_3b_3task_cgct$Security_Hotspots, sonar_3b_cg_ct$Security_Hotspots)

print("Reliability - 3-task vs CG+CT (3B)")
res_3task_vs_cgct$Wilcoxon.p = append(res_3task_vs_cgct$Wilcoxon.p, wilcox_test(sonar_3b_3task_cgct$Reliability, sonar_3b_cg_ct$Reliability))
cliff.delta(sonar_3b_3task_cgct$Reliability, sonar_3b_cg_ct$Reliability)

print("Maintainability - 3-task vs CG+CT (3B)")
res_3task_vs_cgct$Wilcoxon.p = append(res_3task_vs_cgct$Wilcoxon.p, wilcox_test(sonar_3b_3task_cgct$Maintainability, sonar_3b_cg_ct$Maintainability))
cliff.delta(sonar_3b_3task_cgct$Maintainability, sonar_3b_cg_ct$Maintainability)

print("CyC - 3-task vs CG+CT (3B)")
res_3task_vs_cgct$Wilcoxon.p = append(res_3task_vs_cgct$Wilcoxon.p, wilcox_test(sonar_3b_3task_cgct$CyC, sonar_3b_cg_ct$CyC))
cliff.delta(sonar_3b_3task_cgct$CyC, sonar_3b_cg_ct$CyC)

print("CoC - 3-task vs CG+CT (3B)")
res_3task_vs_cgct$Wilcoxon.p = append(res_3task_vs_cgct$Wilcoxon.p, wilcox_test(sonar_3b_3task_cgct$CoC, sonar_3b_cg_ct$CoC))
cliff.delta(sonar_3b_3task_cgct$CoC, sonar_3b_cg_ct$CoC)

# ============== COMPARISON 2: 3-task vs CS+CT ==============
print("********************** QWEN 3B: MT-QLoRA (3-task) vs MT-QLoRA (CS+CT) (C# to Java) *********************************")

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

# PMD (Note: file says cg_cs but using for CS+CT comparison)
print("bestpractices - 3-task vs CS+CT (3B)")
res_3task_vs_csct$Wilcoxon.p = append(res_3task_vs_csct$Wilcoxon.p, wilcox_test(pmd_3b_3task_csct$bestpractices, pmd_3b_cs_ct$bestpractices))
cliff.delta(pmd_3b_3task_csct$bestpractices, pmd_3b_cs_ct$bestpractices)

print("codestyle - 3-task vs CS+CT (3B)")
res_3task_vs_csct$Wilcoxon.p = append(res_3task_vs_csct$Wilcoxon.p, wilcox_test(pmd_3b_3task_csct$codestyle, pmd_3b_cs_ct$codestyle))
cliff.delta(pmd_3b_3task_csct$codestyle, pmd_3b_cs_ct$codestyle)

print("design - 3-task vs CS+CT (3B)")
res_3task_vs_csct$Wilcoxon.p = append(res_3task_vs_csct$Wilcoxon.p, wilcox_test(pmd_3b_3task_csct$design, pmd_3b_cs_ct$design))
cliff.delta(pmd_3b_3task_csct$design, pmd_3b_cs_ct$design)

print("errorprone - 3-task vs CS+CT (3B)")
res_3task_vs_csct$Wilcoxon.p = append(res_3task_vs_csct$Wilcoxon.p, wilcox_test(pmd_3b_3task_csct$errorprone, pmd_3b_cs_ct$errorprone))
cliff.delta(pmd_3b_3task_csct$errorprone, pmd_3b_cs_ct$errorprone)

print("multithreading - 3-task vs CS+CT (3B)")
res_3task_vs_csct$Wilcoxon.p = append(res_3task_vs_csct$Wilcoxon.p, wilcox_test(pmd_3b_3task_csct$multithreading, pmd_3b_cs_ct$multithreading))
cliff.delta(pmd_3b_3task_csct$multithreading, pmd_3b_cs_ct$multithreading)

print("performance - 3-task vs CS+CT (3B)")
res_3task_vs_csct$Wilcoxon.p = append(res_3task_vs_csct$Wilcoxon.p, wilcox_test(pmd_3b_3task_csct$performance, pmd_3b_cs_ct$performance))
cliff.delta(pmd_3b_3task_csct$performance, pmd_3b_cs_ct$performance)

# SONARCLOUD
print("Security_Hotspots - 3-task vs CS+CT (3B)")
res_3task_vs_csct$Wilcoxon.p = append(res_3task_vs_csct$Wilcoxon.p, wilcox_test(sonar_3b_3task_csct$Security_Hotspots, sonar_3b_cs_ct$Security_Hotspots))
cliff.delta(sonar_3b_3task_csct$Security_Hotspots, sonar_3b_cs_ct$Security_Hotspots)

print("Reliability - 3-task vs CS+CT (3B)")
res_3task_vs_csct$Wilcoxon.p = append(res_3task_vs_csct$Wilcoxon.p, wilcox_test(sonar_3b_3task_csct$Reliability, sonar_3b_cs_ct$Reliability))
cliff.delta(sonar_3b_3task_csct$Reliability, sonar_3b_cs_ct$Reliability)

print("Maintainability - 3-task vs CS+CT (3B)")
res_3task_vs_csct$Wilcoxon.p = append(res_3task_vs_csct$Wilcoxon.p, wilcox_test(sonar_3b_3task_csct$Maintainability, sonar_3b_cs_ct$Maintainability))
cliff.delta(sonar_3b_3task_csct$Maintainability, sonar_3b_cs_ct$Maintainability)

print("CyC - 3-task vs CS+CT (3B)")
res_3task_vs_csct$Wilcoxon.p = append(res_3task_vs_csct$Wilcoxon.p, wilcox_test(sonar_3b_3task_csct$CyC, sonar_3b_cs_ct$CyC))
cliff.delta(sonar_3b_3task_csct$CyC, sonar_3b_cs_ct$CyC)

print("CoC - 3-task vs CS+CT (3B)")
res_3task_vs_csct$Wilcoxon.p = append(res_3task_vs_csct$Wilcoxon.p, wilcox_test(sonar_3b_3task_csct$CoC, sonar_3b_cs_ct$CoC))
cliff.delta(sonar_3b_3task_csct$CoC, sonar_3b_cs_ct$CoC)

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
sink(paste0(output_path, "/rq3_ct_cs_java_3task_vs_2task_wilcoxon_CLEAN.txt"))

cat("================================================================================\n")
cat("RQ3: Code Translation - C# to Java (3-task vs 2-task) - Wilcoxon Signed-Rank Test\n")
cat("================================================================================\n\n")

# Metric labels (CodeBLEU + Lizard + PMD + SonarCloud)
metric_labels <- c("CodeBLEU",
                   "Lizard_LoC", "Lizard_ToK", "Lizard_DR", "Lizard_CyC",
                   "PMD_BestPrac", "PMD_CS", "PMD_Design", "PMD_EP", "PMD_MT", "PMD_Perf",
                   "Sonar_Sec", "Sonar_Rel", "Sonar_Main", "Sonar_CyC", "Sonar_CoC")

# Get delta values for 3-task vs CG+CT
delta_3task_vs_cgct <- c(
  get_delta(codebleu_3b_3task_cgct$codebleu, codebleu_3b_cg_ct$codebleu),
  get_delta(lizard_3b_3task_cgct$Lines_of_Code, lizard_3b_cg_ct$Lines_of_Code),
  get_delta(lizard_3b_3task_cgct$Token_Count, lizard_3b_cg_ct$Token_Count),
  get_delta(lizard_3b_3task_cgct$Detection_Rate, lizard_3b_cg_ct$Detection_Rate),
  get_delta(lizard_3b_3task_cgct$Cyclomatic_Complexity, lizard_3b_cg_ct$Cyclomatic_Complexity),
  get_delta(pmd_3b_3task_cgct$bestpractices, pmd_3b_cg_ct$bestpractices),
  get_delta(pmd_3b_3task_cgct$codestyle, pmd_3b_cg_ct$codestyle),
  get_delta(pmd_3b_3task_cgct$design, pmd_3b_cg_ct$design),
  get_delta(pmd_3b_3task_cgct$errorprone, pmd_3b_cg_ct$errorprone),
  get_delta(pmd_3b_3task_cgct$multithreading, pmd_3b_cg_ct$multithreading),
  get_delta(pmd_3b_3task_cgct$performance, pmd_3b_cg_ct$performance),
  get_delta(sonar_3b_3task_cgct$Security_Hotspots, sonar_3b_cg_ct$Security_Hotspots),
  get_delta(sonar_3b_3task_cgct$Reliability, sonar_3b_cg_ct$Reliability),
  get_delta(sonar_3b_3task_cgct$Maintainability, sonar_3b_cg_ct$Maintainability),
  get_delta(sonar_3b_3task_cgct$CyC, sonar_3b_cg_ct$CyC),
  get_delta(sonar_3b_3task_cgct$CoC, sonar_3b_cg_ct$CoC)
)

# Get delta values for 3-task vs CS+CT
delta_3task_vs_csct <- c(
  get_delta(codebleu_3b_3task_csct$codebleu, codebleu_3b_cs_ct$codebleu),
  get_delta(lizard_3b_3task_csct$Lines_of_Code, lizard_3b_cs_ct$Lines_of_Code),
  get_delta(lizard_3b_3task_csct$Token_Count, lizard_3b_cs_ct$Token_Count),
  get_delta(lizard_3b_3task_csct$Detection_Rate, lizard_3b_cs_ct$Detection_Rate),
  get_delta(lizard_3b_3task_csct$Cyclomatic_Complexity, lizard_3b_cs_ct$Cyclomatic_Complexity),
  get_delta(pmd_3b_3task_csct$bestpractices, pmd_3b_cs_ct$bestpractices),
  get_delta(pmd_3b_3task_csct$codestyle, pmd_3b_cs_ct$codestyle),
  get_delta(pmd_3b_3task_csct$design, pmd_3b_cs_ct$design),
  get_delta(pmd_3b_3task_csct$errorprone, pmd_3b_cs_ct$errorprone),
  get_delta(pmd_3b_3task_csct$multithreading, pmd_3b_cs_ct$multithreading),
  get_delta(pmd_3b_3task_csct$performance, pmd_3b_cs_ct$performance),
  get_delta(sonar_3b_3task_csct$Security_Hotspots, sonar_3b_cs_ct$Security_Hotspots),
  get_delta(sonar_3b_3task_csct$Reliability, sonar_3b_cs_ct$Reliability),
  get_delta(sonar_3b_3task_csct$Maintainability, sonar_3b_cs_ct$Maintainability),
  get_delta(sonar_3b_3task_csct$CyC, sonar_3b_cs_ct$CyC),
  get_delta(sonar_3b_3task_csct$CoC, sonar_3b_cs_ct$CoC)
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
cat("Note: PMD CS+CT comparison uses file: r-pmd_ct_3b_cg_cs-mt-qlora.csv\n")
cat("Cliff's Delta interpretation: |d| < 0.147 (N), < 0.33 (S), < 0.474 (M), >= 0.474 (L)\n")
cat("================================================================================\n")

sink()

print("Results saved to:")
print(paste0(output_path, "/rq3_ct_cs_java_3task_vs_2task_wilcoxon_RAW.txt"))
print(paste0(output_path, "/rq3_ct_cs_java_3task_vs_2task_wilcoxon_CLEAN.txt"))