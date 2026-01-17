rm(list=ls())

if (!require("effsize")) install.packages("effsize")
library(effsize)

# Base path
base_path <- "/scratch/oldhome/mhaque/quantized-model-code-quality/statistical-tests/per-instance-value"
output_path <- "/scratch/oldhome/mhaque/quantized-model-code-quality/statistical-tests/R-scripts-v2"

# ============== LOAD LIZARD DATA ==============
lizard_0_5b_st <- read.csv(paste0(base_path, "/lizard/generation/python/r-lizard_cg_0_5b-st-qlora.csv"), header=TRUE)
lizard_0_5b_mt <- read.csv(paste0(base_path, "/lizard/generation/python/r-lizard_cg_0_5b-mt-qlora.csv"), header=TRUE)
lizard_1_5b_st <- read.csv(paste0(base_path, "/lizard/generation/python/r-lizard_cg_1_5b-st-qlora.csv"), header=TRUE)
lizard_1_5b_mt <- read.csv(paste0(base_path, "/lizard/generation/python/r-lizard_cg_1_5b-mt-qlora.csv"), header=TRUE)
lizard_3b_st <- read.csv(paste0(base_path, "/lizard/generation/python/r-lizard_cg_3b-st-qlora.csv"), header=TRUE)
lizard_3b_mt <- read.csv(paste0(base_path, "/lizard/generation/python/r-lizard_cg_3b-mt-qlora.csv"), header=TRUE)

# ============== LOAD PYLINT DATA ==============
pylint_0_5b_st <- read.csv(paste0(base_path, "/pylint/generation/r-pylint_cg_0.5b-st-qlora.csv"), header=TRUE)
pylint_0_5b_mt <- read.csv(paste0(base_path, "/pylint/generation/r-pylint_cg_0_5b-mt-qlora.csv"), header=TRUE)
pylint_1_5b_st <- read.csv(paste0(base_path, "/pylint/generation/r-pylint_cg_1.5b-st-qlora.csv"), header=TRUE)
pylint_1_5b_mt <- read.csv(paste0(base_path, "/pylint/generation/r-pylint_cg_1_5b-mt-qlora.csv"), header=TRUE)
pylint_3b_st <- read.csv(paste0(base_path, "/pylint/generation/r-pylint_cg_3b-st-qlora.csv"), header=TRUE)
pylint_3b_mt <- read.csv(paste0(base_path, "/pylint/generation/r-pylint_cg_3b-mt-qlora.csv"), header=TRUE)

# ============== LOAD SONARCLOUD DATA ==============
sonar_0_5b_st <- read.csv(paste0(base_path, "/sonarcloud/generation/py/r-sonarcloud_cg_qwen0_5_st_qlora.csv"), header=TRUE)
sonar_0_5b_mt <- read.csv(paste0(base_path, "/sonarcloud/generation/py/r-sonarcloud_cg_qwen0_5_mt_qlora.csv"), header=TRUE)
sonar_1_5b_st <- read.csv(paste0(base_path, "/sonarcloud/generation/py/r-sonarcloud_cg_qwen1_5_st_qlora.csv"), header=TRUE)
sonar_1_5b_mt <- read.csv(paste0(base_path, "/sonarcloud/generation/py/r-sonarcloud_cg_qwen1_5_mt_qlora.csv"), header=TRUE)
sonar_3b_st <- read.csv(paste0(base_path, "/sonarcloud/generation/py/r-sonarcloud_cg_qwen3_st_qlora.csv"), header=TRUE)
sonar_3b_mt <- read.csv(paste0(base_path, "/sonarcloud/generation/py/r-sonarcloud_cg_qwen3_mt_qlora.csv"), header=TRUE)

# ============== ENSURE SAME LENGTH ==============
min_len <- min(nrow(lizard_0_5b_st), nrow(lizard_0_5b_mt))
lizard_0_5b_st <- lizard_0_5b_st[1:min_len, ]; lizard_0_5b_mt <- lizard_0_5b_mt[1:min_len, ]
min_len <- min(nrow(lizard_1_5b_st), nrow(lizard_1_5b_mt))
lizard_1_5b_st <- lizard_1_5b_st[1:min_len, ]; lizard_1_5b_mt <- lizard_1_5b_mt[1:min_len, ]
min_len <- min(nrow(lizard_3b_st), nrow(lizard_3b_mt))
lizard_3b_st <- lizard_3b_st[1:min_len, ]; lizard_3b_mt <- lizard_3b_mt[1:min_len, ]

min_len <- min(nrow(pylint_0_5b_st), nrow(pylint_0_5b_mt))
pylint_0_5b_st <- pylint_0_5b_st[1:min_len, ]; pylint_0_5b_mt <- pylint_0_5b_mt[1:min_len, ]
min_len <- min(nrow(pylint_1_5b_st), nrow(pylint_1_5b_mt))
pylint_1_5b_st <- pylint_1_5b_st[1:min_len, ]; pylint_1_5b_mt <- pylint_1_5b_mt[1:min_len, ]
min_len <- min(nrow(pylint_3b_st), nrow(pylint_3b_mt))
pylint_3b_st <- pylint_3b_st[1:min_len, ]; pylint_3b_mt <- pylint_3b_mt[1:min_len, ]

min_len <- min(nrow(sonar_0_5b_st), nrow(sonar_0_5b_mt))
sonar_0_5b_st <- sonar_0_5b_st[1:min_len, ]; sonar_0_5b_mt <- sonar_0_5b_mt[1:min_len, ]
min_len <- min(nrow(sonar_1_5b_st), nrow(sonar_1_5b_mt))
sonar_1_5b_st <- sonar_1_5b_st[1:min_len, ]; sonar_1_5b_mt <- sonar_1_5b_mt[1:min_len, ]
min_len <- min(nrow(sonar_3b_st), nrow(sonar_3b_mt))
sonar_3b_st <- sonar_3b_st[1:min_len, ]; sonar_3b_mt <- sonar_3b_mt[1:min_len, ]

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
sink(paste0(output_path, "/rq1_cg_python_st_vs_mt_wilcoxon_RAW.txt"))

# ============== QWEN 0.5B ==============
print("********************** QWEN 0.5B: ST-QLoRA vs MT-QLoRA (Python) *********************************")

res_0_5b = list(Wilcoxon.p = c())

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

print("Error - ST vs MT (0.5B)")
res_0_5b$Wilcoxon.p = append(res_0_5b$Wilcoxon.p, wilcox_test(pylint_0_5b_st$Error, pylint_0_5b_mt$Error))
cliff.delta(pylint_0_5b_st$Error, pylint_0_5b_mt$Error)

print("Warning - ST vs MT (0.5B)")
res_0_5b$Wilcoxon.p = append(res_0_5b$Wilcoxon.p, wilcox_test(pylint_0_5b_st$Warning, pylint_0_5b_mt$Warning))
cliff.delta(pylint_0_5b_st$Warning, pylint_0_5b_mt$Warning)

print("Convention - ST vs MT (0.5B)")
res_0_5b$Wilcoxon.p = append(res_0_5b$Wilcoxon.p, wilcox_test(pylint_0_5b_st$Convention, pylint_0_5b_mt$Convention))
cliff.delta(pylint_0_5b_st$Convention, pylint_0_5b_mt$Convention)

print("Refactor - ST vs MT (0.5B)")
res_0_5b$Wilcoxon.p = append(res_0_5b$Wilcoxon.p, wilcox_test(pylint_0_5b_st$Refactor, pylint_0_5b_mt$Refactor))
cliff.delta(pylint_0_5b_st$Refactor, pylint_0_5b_mt$Refactor)

print("Security_Hotspots - ST vs MT (0.5B)")
res_0_5b$Wilcoxon.p = append(res_0_5b$Wilcoxon.p, wilcox_test(sonar_0_5b_st$Security_Hotspots, sonar_0_5b_mt$Security_Hotspots))
cliff.delta(sonar_0_5b_st$Security_Hotspots, sonar_0_5b_mt$Security_Hotspots)

print("Reliability - ST vs MT (0.5B)")
res_0_5b$Wilcoxon.p = append(res_0_5b$Wilcoxon.p, wilcox_test(sonar_0_5b_st$Reliability, sonar_0_5b_mt$Reliability))
cliff.delta(sonar_0_5b_st$Reliability, sonar_0_5b_mt$Reliability)

print("Maintainability - ST vs MT (0.5B)")
res_0_5b$Wilcoxon.p = append(res_0_5b$Wilcoxon.p, wilcox_test(sonar_0_5b_st$Maintainability, sonar_0_5b_mt$Maintainability))
cliff.delta(sonar_0_5b_st$Maintainability, sonar_0_5b_mt$Maintainability)

print("CyC - ST vs MT (0.5B)")
res_0_5b$Wilcoxon.p = append(res_0_5b$Wilcoxon.p, wilcox_test(sonar_0_5b_st$CyC, sonar_0_5b_mt$CyC))
cliff.delta(sonar_0_5b_st$CyC, sonar_0_5b_mt$CyC)

print("CoC - ST vs MT (0.5B)")
res_0_5b$Wilcoxon.p = append(res_0_5b$Wilcoxon.p, wilcox_test(sonar_0_5b_st$CoC, sonar_0_5b_mt$CoC))
cliff.delta(sonar_0_5b_st$CoC, sonar_0_5b_mt$CoC)

# ============== QWEN 1.5B ==============
print("********************** QWEN 1.5B: ST-QLoRA vs MT-QLoRA (Python) *********************************")

res_1_5b = list(Wilcoxon.p = c())

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

print("Error - ST vs MT (1.5B)")
res_1_5b$Wilcoxon.p = append(res_1_5b$Wilcoxon.p, wilcox_test(pylint_1_5b_st$Error, pylint_1_5b_mt$Error))
cliff.delta(pylint_1_5b_st$Error, pylint_1_5b_mt$Error)

print("Warning - ST vs MT (1.5B)")
res_1_5b$Wilcoxon.p = append(res_1_5b$Wilcoxon.p, wilcox_test(pylint_1_5b_st$Warning, pylint_1_5b_mt$Warning))
cliff.delta(pylint_1_5b_st$Warning, pylint_1_5b_mt$Warning)

print("Convention - ST vs MT (1.5B)")
res_1_5b$Wilcoxon.p = append(res_1_5b$Wilcoxon.p, wilcox_test(pylint_1_5b_st$Convention, pylint_1_5b_mt$Convention))
cliff.delta(pylint_1_5b_st$Convention, pylint_1_5b_mt$Convention)

print("Refactor - ST vs MT (1.5B)")
res_1_5b$Wilcoxon.p = append(res_1_5b$Wilcoxon.p, wilcox_test(pylint_1_5b_st$Refactor, pylint_1_5b_mt$Refactor))
cliff.delta(pylint_1_5b_st$Refactor, pylint_1_5b_mt$Refactor)

print("Security_Hotspots - ST vs MT (1.5B)")
res_1_5b$Wilcoxon.p = append(res_1_5b$Wilcoxon.p, wilcox_test(sonar_1_5b_st$Security_Hotspots, sonar_1_5b_mt$Security_Hotspots))
cliff.delta(sonar_1_5b_st$Security_Hotspots, sonar_1_5b_mt$Security_Hotspots)

print("Reliability - ST vs MT (1.5B)")
res_1_5b$Wilcoxon.p = append(res_1_5b$Wilcoxon.p, wilcox_test(sonar_1_5b_st$Reliability, sonar_1_5b_mt$Reliability))
cliff.delta(sonar_1_5b_st$Reliability, sonar_1_5b_mt$Reliability)

print("Maintainability - ST vs MT (1.5B)")
res_1_5b$Wilcoxon.p = append(res_1_5b$Wilcoxon.p, wilcox_test(sonar_1_5b_st$Maintainability, sonar_1_5b_mt$Maintainability))
cliff.delta(sonar_1_5b_st$Maintainability, sonar_1_5b_mt$Maintainability)

print("CyC - ST vs MT (1.5B)")
res_1_5b$Wilcoxon.p = append(res_1_5b$Wilcoxon.p, wilcox_test(sonar_1_5b_st$CyC, sonar_1_5b_mt$CyC))
cliff.delta(sonar_1_5b_st$CyC, sonar_1_5b_mt$CyC)

print("CoC - ST vs MT (1.5B)")
res_1_5b$Wilcoxon.p = append(res_1_5b$Wilcoxon.p, wilcox_test(sonar_1_5b_st$CoC, sonar_1_5b_mt$CoC))
cliff.delta(sonar_1_5b_st$CoC, sonar_1_5b_mt$CoC)

# ============== QWEN 3B ==============
print("********************** QWEN 3B: ST-QLoRA vs MT-QLoRA (Python) *********************************")

res_3b = list(Wilcoxon.p = c())

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

print("Error - ST vs MT (3B)")
res_3b$Wilcoxon.p = append(res_3b$Wilcoxon.p, wilcox_test(pylint_3b_st$Error, pylint_3b_mt$Error))
cliff.delta(pylint_3b_st$Error, pylint_3b_mt$Error)

print("Warning - ST vs MT (3B)")
res_3b$Wilcoxon.p = append(res_3b$Wilcoxon.p, wilcox_test(pylint_3b_st$Warning, pylint_3b_mt$Warning))
cliff.delta(pylint_3b_st$Warning, pylint_3b_mt$Warning)

print("Convention - ST vs MT (3B)")
res_3b$Wilcoxon.p = append(res_3b$Wilcoxon.p, wilcox_test(pylint_3b_st$Convention, pylint_3b_mt$Convention))
cliff.delta(pylint_3b_st$Convention, pylint_3b_mt$Convention)

print("Refactor - ST vs MT (3B)")
res_3b$Wilcoxon.p = append(res_3b$Wilcoxon.p, wilcox_test(pylint_3b_st$Refactor, pylint_3b_mt$Refactor))
cliff.delta(pylint_3b_st$Refactor, pylint_3b_mt$Refactor)

print("Security_Hotspots - ST vs MT (3B)")
res_3b$Wilcoxon.p = append(res_3b$Wilcoxon.p, wilcox_test(sonar_3b_st$Security_Hotspots, sonar_3b_mt$Security_Hotspots))
cliff.delta(sonar_3b_st$Security_Hotspots, sonar_3b_mt$Security_Hotspots)

print("Reliability - ST vs MT (3B)")
res_3b$Wilcoxon.p = append(res_3b$Wilcoxon.p, wilcox_test(sonar_3b_st$Reliability, sonar_3b_mt$Reliability))
cliff.delta(sonar_3b_st$Reliability, sonar_3b_mt$Reliability)

print("Maintainability - ST vs MT (3B)")
res_3b$Wilcoxon.p = append(res_3b$Wilcoxon.p, wilcox_test(sonar_3b_st$Maintainability, sonar_3b_mt$Maintainability))
cliff.delta(sonar_3b_st$Maintainability, sonar_3b_mt$Maintainability)

print("CyC - ST vs MT (3B)")
res_3b$Wilcoxon.p = append(res_3b$Wilcoxon.p, wilcox_test(sonar_3b_st$CyC, sonar_3b_mt$CyC))
cliff.delta(sonar_3b_st$CyC, sonar_3b_mt$CyC)

print("CoC - ST vs MT (3B)")
res_3b$Wilcoxon.p = append(res_3b$Wilcoxon.p, wilcox_test(sonar_3b_st$CoC, sonar_3b_mt$CoC))
cliff.delta(sonar_3b_st$CoC, sonar_3b_mt$CoC)

# ============== PRINT RAW RESULTS ==============
print("***************************************************************************")

res_df_0_5b = data.frame(Wilcoxon.p = res_0_5b$Wilcoxon.p)
res_df_0_5b$Wilcoxon.p = p.adjust(res_df_0_5b$Wilcoxon.p, method="BH")
print("Qwen 0.5B Results (Adjusted P-values):")
print(res_df_0_5b)

res_df_1_5b = data.frame(Wilcoxon.p = res_1_5b$Wilcoxon.p)
res_df_1_5b$Wilcoxon.p = p.adjust(res_df_1_5b$Wilcoxon.p, method="BH")
print("Qwen 1.5B Results (Adjusted P-values):")
print(res_df_1_5b)

res_df_3b = data.frame(Wilcoxon.p = res_3b$Wilcoxon.p)
res_df_3b$Wilcoxon.p = p.adjust(res_df_3b$Wilcoxon.p, method="BH")
print("Qwen 3B Results (Adjusted P-values):")
print(res_df_3b)

sink()

# ============================================================================
# FILE 2: CLEAN SORTED OUTPUT (table format)
# ============================================================================
sink(paste0(output_path, "/rq1_cg_python_st_vs_mt_wilcoxon_CLEAN.txt"))

cat("================================================================================\n")
cat("RQ1: Code Generation - Python (ST-QLoRA vs MT-QLoRA) - Wilcoxon Signed-Rank Test\n")
cat("================================================================================\n\n")

# Metric labels
metric_labels <- c("Lizard_LoC", "Lizard_ToK", "Lizard_DR", "Lizard_CyC",
                   "Pylint_Err", "Pylint_War", "Pylint_Conv", "Pylint_Ref",
                   "Sonar_SecHot", "Sonar_Rel", "Sonar_Main", "Sonar_CyC", "Sonar_CoC")

# Get delta values for 0.5B
delta_0_5b <- c(
  get_delta(lizard_0_5b_st$Lines_of_Code, lizard_0_5b_mt$Lines_of_Code),
  get_delta(lizard_0_5b_st$Token_Count, lizard_0_5b_mt$Token_Count),
  get_delta(lizard_0_5b_st$Detection_Rate, lizard_0_5b_mt$Detection_Rate),
  get_delta(lizard_0_5b_st$Cyclomatic_Complexity, lizard_0_5b_mt$Cyclomatic_Complexity),
  get_delta(pylint_0_5b_st$Error, pylint_0_5b_mt$Error),
  get_delta(pylint_0_5b_st$Warning, pylint_0_5b_mt$Warning),
  get_delta(pylint_0_5b_st$Convention, pylint_0_5b_mt$Convention),
  get_delta(pylint_0_5b_st$Refactor, pylint_0_5b_mt$Refactor),
  get_delta(sonar_0_5b_st$Security_Hotspots, sonar_0_5b_mt$Security_Hotspots),
  get_delta(sonar_0_5b_st$Reliability, sonar_0_5b_mt$Reliability),
  get_delta(sonar_0_5b_st$Maintainability, sonar_0_5b_mt$Maintainability),
  get_delta(sonar_0_5b_st$CyC, sonar_0_5b_mt$CyC),
  get_delta(sonar_0_5b_st$CoC, sonar_0_5b_mt$CoC)
)

# Get delta values for 1.5B
delta_1_5b <- c(
  get_delta(lizard_1_5b_st$Lines_of_Code, lizard_1_5b_mt$Lines_of_Code),
  get_delta(lizard_1_5b_st$Token_Count, lizard_1_5b_mt$Token_Count),
  get_delta(lizard_1_5b_st$Detection_Rate, lizard_1_5b_mt$Detection_Rate),
  get_delta(lizard_1_5b_st$Cyclomatic_Complexity, lizard_1_5b_mt$Cyclomatic_Complexity),
  get_delta(pylint_1_5b_st$Error, pylint_1_5b_mt$Error),
  get_delta(pylint_1_5b_st$Warning, pylint_1_5b_mt$Warning),
  get_delta(pylint_1_5b_st$Convention, pylint_1_5b_mt$Convention),
  get_delta(pylint_1_5b_st$Refactor, pylint_1_5b_mt$Refactor),
  get_delta(sonar_1_5b_st$Security_Hotspots, sonar_1_5b_mt$Security_Hotspots),
  get_delta(sonar_1_5b_st$Reliability, sonar_1_5b_mt$Reliability),
  get_delta(sonar_1_5b_st$Maintainability, sonar_1_5b_mt$Maintainability),
  get_delta(sonar_1_5b_st$CyC, sonar_1_5b_mt$CyC),
  get_delta(sonar_1_5b_st$CoC, sonar_1_5b_mt$CoC)
)

# Get delta values for 3B
delta_3b <- c(
  get_delta(lizard_3b_st$Lines_of_Code, lizard_3b_mt$Lines_of_Code),
  get_delta(lizard_3b_st$Token_Count, lizard_3b_mt$Token_Count),
  get_delta(lizard_3b_st$Detection_Rate, lizard_3b_mt$Detection_Rate),
  get_delta(lizard_3b_st$Cyclomatic_Complexity, lizard_3b_mt$Cyclomatic_Complexity),
  get_delta(pylint_3b_st$Error, pylint_3b_mt$Error),
  get_delta(pylint_3b_st$Warning, pylint_3b_mt$Warning),
  get_delta(pylint_3b_st$Convention, pylint_3b_mt$Convention),
  get_delta(pylint_3b_st$Refactor, pylint_3b_mt$Refactor),
  get_delta(sonar_3b_st$Security_Hotspots, sonar_3b_mt$Security_Hotspots),
  get_delta(sonar_3b_st$Reliability, sonar_3b_mt$Reliability),
  get_delta(sonar_3b_st$Maintainability, sonar_3b_mt$Maintainability),
  get_delta(sonar_3b_st$CyC, sonar_3b_mt$CyC),
  get_delta(sonar_3b_st$CoC, sonar_3b_mt$CoC)
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
cat("Note: p-values are BH-Bonferroni corrected\n")
cat("Cliff's Delta interpretation: |d| < 0.147 (N), < 0.33 (S), < 0.474 (M), >= 0.474 (L)\n")
cat("================================================================================\n")

sink()

print("Results saved to:")
print(paste0(output_path, "/rq1_cg_python_st_vs_mt_wilcoxon_RAW.txt"))
print(paste0(output_path, "/rq1_cg_python_st_vs_mt_wilcoxon_CLEAN.txt"))