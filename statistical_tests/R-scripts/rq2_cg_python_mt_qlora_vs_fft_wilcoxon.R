rm(list=ls())

if (!require("effsize")) install.packages("effsize")
library(effsize)

# Base path
base_path <- "/scratch/oldhome/mhaque/quantized-model-code-quality/statistical-tests/per-instance-value"
output_path <- "/scratch/oldhome/mhaque/quantized-model-code-quality/statistical-tests/R-scripts-v2"

# ============== LOAD LIZARD DATA ==============
lizard_0_5b_qlora <- read.csv(paste0(base_path, "/lizard/generation/python/r-lizard_cg_0_5b-mt-qlora.csv"), header=TRUE)
lizard_0_5b_fft <- read.csv(paste0(base_path, "/lizard/generation/python/r-lizard_cg_0_5b-mt-fft.csv"), header=TRUE)
lizard_1_5b_qlora <- read.csv(paste0(base_path, "/lizard/generation/python/r-lizard_cg_1_5b-mt-qlora.csv"), header=TRUE)
lizard_1_5b_fft <- read.csv(paste0(base_path, "/lizard/generation/python/r-lizard_cg_1_5b-mt-fft.csv"), header=TRUE)
lizard_3b_qlora <- read.csv(paste0(base_path, "/lizard/generation/python/r-lizard_cg_3b-mt-qlora.csv"), header=TRUE)
lizard_3b_fft <- read.csv(paste0(base_path, "/lizard/generation/python/r-lizard_cg_3b-mt-fft.csv"), header=TRUE)

# ============== LOAD PYLINT DATA ==============
pylint_0_5b_qlora <- read.csv(paste0(base_path, "/pylint/generation/r-pylint_cg_0_5b-mt-qlora.csv"), header=TRUE)
pylint_0_5b_fft <- read.csv(paste0(base_path, "/pylint/generation/r-pylint_cg_0_5b-mt-fft.csv"), header=TRUE)
pylint_1_5b_qlora <- read.csv(paste0(base_path, "/pylint/generation/r-pylint_cg_1_5b-mt-qlora.csv"), header=TRUE)
pylint_1_5b_fft <- read.csv(paste0(base_path, "/pylint/generation/r-pylint_cg_1_5b-mt-fft.csv"), header=TRUE)
pylint_3b_qlora <- read.csv(paste0(base_path, "/pylint/generation/r-pylint_cg_3b-mt-qlora.csv"), header=TRUE)
pylint_3b_fft <- read.csv(paste0(base_path, "/pylint/generation/r-pylint_cg_3b-mt-fft.csv"), header=TRUE)

# ============== LOAD SONARCLOUD DATA ==============
sonar_0_5b_qlora <- read.csv(paste0(base_path, "/sonarcloud/generation/py/r-sonarcloud_cg_qwen0_5_mt_qlora.csv"), header=TRUE)
sonar_0_5b_fft <- read.csv(paste0(base_path, "/sonarcloud/generation/py/r-sonarcloud_cg_qwen0_5_mt_fft.csv"), header=TRUE)
sonar_1_5b_qlora <- read.csv(paste0(base_path, "/sonarcloud/generation/py/r-sonarcloud_cg_qwen1_5_mt_qlora.csv"), header=TRUE)
sonar_1_5b_fft <- read.csv(paste0(base_path, "/sonarcloud/generation/py/r-sonarcloud_cg_qwen1_5_mt_fft.csv"), header=TRUE)
sonar_3b_qlora <- read.csv(paste0(base_path, "/sonarcloud/generation/py/r-sonarcloud_cg_qwen3_mt_qlora.csv"), header=TRUE)
sonar_3b_fft <- read.csv(paste0(base_path, "/sonarcloud/generation/py/r-sonarcloud_cg_qwen3_mt_fft.csv"), header=TRUE)

# ============== ENSURE SAME LENGTH ==============
min_len <- min(nrow(lizard_0_5b_qlora), nrow(lizard_0_5b_fft))
lizard_0_5b_qlora <- lizard_0_5b_qlora[1:min_len, ]; lizard_0_5b_fft <- lizard_0_5b_fft[1:min_len, ]
min_len <- min(nrow(lizard_1_5b_qlora), nrow(lizard_1_5b_fft))
lizard_1_5b_qlora <- lizard_1_5b_qlora[1:min_len, ]; lizard_1_5b_fft <- lizard_1_5b_fft[1:min_len, ]
min_len <- min(nrow(lizard_3b_qlora), nrow(lizard_3b_fft))
lizard_3b_qlora <- lizard_3b_qlora[1:min_len, ]; lizard_3b_fft <- lizard_3b_fft[1:min_len, ]

min_len <- min(nrow(pylint_0_5b_qlora), nrow(pylint_0_5b_fft))
pylint_0_5b_qlora <- pylint_0_5b_qlora[1:min_len, ]; pylint_0_5b_fft <- pylint_0_5b_fft[1:min_len, ]
min_len <- min(nrow(pylint_1_5b_qlora), nrow(pylint_1_5b_fft))
pylint_1_5b_qlora <- pylint_1_5b_qlora[1:min_len, ]; pylint_1_5b_fft <- pylint_1_5b_fft[1:min_len, ]
min_len <- min(nrow(pylint_3b_qlora), nrow(pylint_3b_fft))
pylint_3b_qlora <- pylint_3b_qlora[1:min_len, ]; pylint_3b_fft <- pylint_3b_fft[1:min_len, ]

min_len <- min(nrow(sonar_0_5b_qlora), nrow(sonar_0_5b_fft))
sonar_0_5b_qlora <- sonar_0_5b_qlora[1:min_len, ]; sonar_0_5b_fft <- sonar_0_5b_fft[1:min_len, ]
min_len <- min(nrow(sonar_1_5b_qlora), nrow(sonar_1_5b_fft))
sonar_1_5b_qlora <- sonar_1_5b_qlora[1:min_len, ]; sonar_1_5b_fft <- sonar_1_5b_fft[1:min_len, ]
min_len <- min(nrow(sonar_3b_qlora), nrow(sonar_3b_fft))
sonar_3b_qlora <- sonar_3b_qlora[1:min_len, ]; sonar_3b_fft <- sonar_3b_fft[1:min_len, ]

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
sink(paste0(output_path, "/rq2_cg_python_mt_qlora_vs_fft_wilcoxon_RAW.txt"))

# ============== QWEN 0.5B ==============
print("********************** QWEN 0.5B: MT-QLoRA vs MT-FFT (Python) *********************************")

res_0_5b = list(Wilcoxon.p = c())

# LIZARD
print("Lines_of_Code - QLoRA vs FFT (0.5B)")
res_0_5b$Wilcoxon.p = append(res_0_5b$Wilcoxon.p, wilcox_test(lizard_0_5b_qlora$Lines_of_Code, lizard_0_5b_fft$Lines_of_Code))
cliff.delta(lizard_0_5b_qlora$Lines_of_Code, lizard_0_5b_fft$Lines_of_Code)

print("Token_Count - QLoRA vs FFT (0.5B)")
res_0_5b$Wilcoxon.p = append(res_0_5b$Wilcoxon.p, wilcox_test(lizard_0_5b_qlora$Token_Count, lizard_0_5b_fft$Token_Count))
cliff.delta(lizard_0_5b_qlora$Token_Count, lizard_0_5b_fft$Token_Count)

print("Detection_Rate - QLoRA vs FFT (0.5B)")
res_0_5b$Wilcoxon.p = append(res_0_5b$Wilcoxon.p, wilcox_test(lizard_0_5b_qlora$Detection_Rate, lizard_0_5b_fft$Detection_Rate))
cliff.delta(lizard_0_5b_qlora$Detection_Rate, lizard_0_5b_fft$Detection_Rate)

print("Cyclomatic_Complexity - QLoRA vs FFT (0.5B)")
res_0_5b$Wilcoxon.p = append(res_0_5b$Wilcoxon.p, wilcox_test(lizard_0_5b_qlora$Cyclomatic_Complexity, lizard_0_5b_fft$Cyclomatic_Complexity))
cliff.delta(lizard_0_5b_qlora$Cyclomatic_Complexity, lizard_0_5b_fft$Cyclomatic_Complexity)

# PYLINT
print("Error - QLoRA vs FFT (0.5B)")
res_0_5b$Wilcoxon.p = append(res_0_5b$Wilcoxon.p, wilcox_test(pylint_0_5b_qlora$Error, pylint_0_5b_fft$Error))
cliff.delta(pylint_0_5b_qlora$Error, pylint_0_5b_fft$Error)

print("Warning - QLoRA vs FFT (0.5B)")
res_0_5b$Wilcoxon.p = append(res_0_5b$Wilcoxon.p, wilcox_test(pylint_0_5b_qlora$Warning, pylint_0_5b_fft$Warning))
cliff.delta(pylint_0_5b_qlora$Warning, pylint_0_5b_fft$Warning)

print("Convention - QLoRA vs FFT (0.5B)")
res_0_5b$Wilcoxon.p = append(res_0_5b$Wilcoxon.p, wilcox_test(pylint_0_5b_qlora$Convention, pylint_0_5b_fft$Convention))
cliff.delta(pylint_0_5b_qlora$Convention, pylint_0_5b_fft$Convention)

print("Refactor - QLoRA vs FFT (0.5B)")
res_0_5b$Wilcoxon.p = append(res_0_5b$Wilcoxon.p, wilcox_test(pylint_0_5b_qlora$Refactor, pylint_0_5b_fft$Refactor))
cliff.delta(pylint_0_5b_qlora$Refactor, pylint_0_5b_fft$Refactor)

# SONARCLOUD (ignoring LoC)
print("Security_Hotspots - QLoRA vs FFT (0.5B)")
res_0_5b$Wilcoxon.p = append(res_0_5b$Wilcoxon.p, wilcox_test(sonar_0_5b_qlora$Security_Hotspots, sonar_0_5b_fft$Security_Hotspots))
cliff.delta(sonar_0_5b_qlora$Security_Hotspots, sonar_0_5b_fft$Security_Hotspots)

print("Reliability - QLoRA vs FFT (0.5B)")
res_0_5b$Wilcoxon.p = append(res_0_5b$Wilcoxon.p, wilcox_test(sonar_0_5b_qlora$Reliability, sonar_0_5b_fft$Reliability))
cliff.delta(sonar_0_5b_qlora$Reliability, sonar_0_5b_fft$Reliability)

print("Maintainability - QLoRA vs FFT (0.5B)")
res_0_5b$Wilcoxon.p = append(res_0_5b$Wilcoxon.p, wilcox_test(sonar_0_5b_qlora$Maintainability, sonar_0_5b_fft$Maintainability))
cliff.delta(sonar_0_5b_qlora$Maintainability, sonar_0_5b_fft$Maintainability)

print("CyC - QLoRA vs FFT (0.5B)")
res_0_5b$Wilcoxon.p = append(res_0_5b$Wilcoxon.p, wilcox_test(sonar_0_5b_qlora$CyC, sonar_0_5b_fft$CyC))
cliff.delta(sonar_0_5b_qlora$CyC, sonar_0_5b_fft$CyC)

print("CoC - QLoRA vs FFT (0.5B)")
res_0_5b$Wilcoxon.p = append(res_0_5b$Wilcoxon.p, wilcox_test(sonar_0_5b_qlora$CoC, sonar_0_5b_fft$CoC))
cliff.delta(sonar_0_5b_qlora$CoC, sonar_0_5b_fft$CoC)

# ============== QWEN 1.5B ==============
print("********************** QWEN 1.5B: MT-QLoRA vs MT-FFT (Python) *********************************")

res_1_5b = list(Wilcoxon.p = c())

# LIZARD
print("Lines_of_Code - QLoRA vs FFT (1.5B)")
res_1_5b$Wilcoxon.p = append(res_1_5b$Wilcoxon.p, wilcox_test(lizard_1_5b_qlora$Lines_of_Code, lizard_1_5b_fft$Lines_of_Code))
cliff.delta(lizard_1_5b_qlora$Lines_of_Code, lizard_1_5b_fft$Lines_of_Code)

print("Token_Count - QLoRA vs FFT (1.5B)")
res_1_5b$Wilcoxon.p = append(res_1_5b$Wilcoxon.p, wilcox_test(lizard_1_5b_qlora$Token_Count, lizard_1_5b_fft$Token_Count))
cliff.delta(lizard_1_5b_qlora$Token_Count, lizard_1_5b_fft$Token_Count)

print("Detection_Rate - QLoRA vs FFT (1.5B)")
res_1_5b$Wilcoxon.p = append(res_1_5b$Wilcoxon.p, wilcox_test(lizard_1_5b_qlora$Detection_Rate, lizard_1_5b_fft$Detection_Rate))
cliff.delta(lizard_1_5b_qlora$Detection_Rate, lizard_1_5b_fft$Detection_Rate)

print("Cyclomatic_Complexity - QLoRA vs FFT (1.5B)")
res_1_5b$Wilcoxon.p = append(res_1_5b$Wilcoxon.p, wilcox_test(lizard_1_5b_qlora$Cyclomatic_Complexity, lizard_1_5b_fft$Cyclomatic_Complexity))
cliff.delta(lizard_1_5b_qlora$Cyclomatic_Complexity, lizard_1_5b_fft$Cyclomatic_Complexity)

# PYLINT
print("Error - QLoRA vs FFT (1.5B)")
res_1_5b$Wilcoxon.p = append(res_1_5b$Wilcoxon.p, wilcox_test(pylint_1_5b_qlora$Error, pylint_1_5b_fft$Error))
cliff.delta(pylint_1_5b_qlora$Error, pylint_1_5b_fft$Error)

print("Warning - QLoRA vs FFT (1.5B)")
res_1_5b$Wilcoxon.p = append(res_1_5b$Wilcoxon.p, wilcox_test(pylint_1_5b_qlora$Warning, pylint_1_5b_fft$Warning))
cliff.delta(pylint_1_5b_qlora$Warning, pylint_1_5b_fft$Warning)

print("Convention - QLoRA vs FFT (1.5B)")
res_1_5b$Wilcoxon.p = append(res_1_5b$Wilcoxon.p, wilcox_test(pylint_1_5b_qlora$Convention, pylint_1_5b_fft$Convention))
cliff.delta(pylint_1_5b_qlora$Convention, pylint_1_5b_fft$Convention)

print("Refactor - QLoRA vs FFT (1.5B)")
res_1_5b$Wilcoxon.p = append(res_1_5b$Wilcoxon.p, wilcox_test(pylint_1_5b_qlora$Refactor, pylint_1_5b_fft$Refactor))
cliff.delta(pylint_1_5b_qlora$Refactor, pylint_1_5b_fft$Refactor)

# SONARCLOUD
print("Security_Hotspots - QLoRA vs FFT (1.5B)")
res_1_5b$Wilcoxon.p = append(res_1_5b$Wilcoxon.p, wilcox_test(sonar_1_5b_qlora$Security_Hotspots, sonar_1_5b_fft$Security_Hotspots))
cliff.delta(sonar_1_5b_qlora$Security_Hotspots, sonar_1_5b_fft$Security_Hotspots)

print("Reliability - QLoRA vs FFT (1.5B)")
res_1_5b$Wilcoxon.p = append(res_1_5b$Wilcoxon.p, wilcox_test(sonar_1_5b_qlora$Reliability, sonar_1_5b_fft$Reliability))
cliff.delta(sonar_1_5b_qlora$Reliability, sonar_1_5b_fft$Reliability)

print("Maintainability - QLoRA vs FFT (1.5B)")
res_1_5b$Wilcoxon.p = append(res_1_5b$Wilcoxon.p, wilcox_test(sonar_1_5b_qlora$Maintainability, sonar_1_5b_fft$Maintainability))
cliff.delta(sonar_1_5b_qlora$Maintainability, sonar_1_5b_fft$Maintainability)

print("CyC - QLoRA vs FFT (1.5B)")
res_1_5b$Wilcoxon.p = append(res_1_5b$Wilcoxon.p, wilcox_test(sonar_1_5b_qlora$CyC, sonar_1_5b_fft$CyC))
cliff.delta(sonar_1_5b_qlora$CyC, sonar_1_5b_fft$CyC)

print("CoC - QLoRA vs FFT (1.5B)")
res_1_5b$Wilcoxon.p = append(res_1_5b$Wilcoxon.p, wilcox_test(sonar_1_5b_qlora$CoC, sonar_1_5b_fft$CoC))
cliff.delta(sonar_1_5b_qlora$CoC, sonar_1_5b_fft$CoC)

# ============== QWEN 3B ==============
print("********************** QWEN 3B: MT-QLoRA vs MT-FFT (Python) *********************************")

res_3b = list(Wilcoxon.p = c())

# LIZARD
print("Lines_of_Code - QLoRA vs FFT (3B)")
res_3b$Wilcoxon.p = append(res_3b$Wilcoxon.p, wilcox_test(lizard_3b_qlora$Lines_of_Code, lizard_3b_fft$Lines_of_Code))
cliff.delta(lizard_3b_qlora$Lines_of_Code, lizard_3b_fft$Lines_of_Code)

print("Token_Count - QLoRA vs FFT (3B)")
res_3b$Wilcoxon.p = append(res_3b$Wilcoxon.p, wilcox_test(lizard_3b_qlora$Token_Count, lizard_3b_fft$Token_Count))
cliff.delta(lizard_3b_qlora$Token_Count, lizard_3b_fft$Token_Count)

print("Detection_Rate - QLoRA vs FFT (3B)")
res_3b$Wilcoxon.p = append(res_3b$Wilcoxon.p, wilcox_test(lizard_3b_qlora$Detection_Rate, lizard_3b_fft$Detection_Rate))
cliff.delta(lizard_3b_qlora$Detection_Rate, lizard_3b_fft$Detection_Rate)

print("Cyclomatic_Complexity - QLoRA vs FFT (3B)")
res_3b$Wilcoxon.p = append(res_3b$Wilcoxon.p, wilcox_test(lizard_3b_qlora$Cyclomatic_Complexity, lizard_3b_fft$Cyclomatic_Complexity))
cliff.delta(lizard_3b_qlora$Cyclomatic_Complexity, lizard_3b_fft$Cyclomatic_Complexity)

# PYLINT
print("Error - QLoRA vs FFT (3B)")
res_3b$Wilcoxon.p = append(res_3b$Wilcoxon.p, wilcox_test(pylint_3b_qlora$Error, pylint_3b_fft$Error))
cliff.delta(pylint_3b_qlora$Error, pylint_3b_fft$Error)

print("Warning - QLoRA vs FFT (3B)")
res_3b$Wilcoxon.p = append(res_3b$Wilcoxon.p, wilcox_test(pylint_3b_qlora$Warning, pylint_3b_fft$Warning))
cliff.delta(pylint_3b_qlora$Warning, pylint_3b_fft$Warning)

print("Convention - QLoRA vs FFT (3B)")
res_3b$Wilcoxon.p = append(res_3b$Wilcoxon.p, wilcox_test(pylint_3b_qlora$Convention, pylint_3b_fft$Convention))
cliff.delta(pylint_3b_qlora$Convention, pylint_3b_fft$Convention)

print("Refactor - QLoRA vs FFT (3B)")
res_3b$Wilcoxon.p = append(res_3b$Wilcoxon.p, wilcox_test(pylint_3b_qlora$Refactor, pylint_3b_fft$Refactor))
cliff.delta(pylint_3b_qlora$Refactor, pylint_3b_fft$Refactor)

# SONARCLOUD
print("Security_Hotspots - QLoRA vs FFT (3B)")
res_3b$Wilcoxon.p = append(res_3b$Wilcoxon.p, wilcox_test(sonar_3b_qlora$Security_Hotspots, sonar_3b_fft$Security_Hotspots))
cliff.delta(sonar_3b_qlora$Security_Hotspots, sonar_3b_fft$Security_Hotspots)

print("Reliability - QLoRA vs FFT (3B)")
res_3b$Wilcoxon.p = append(res_3b$Wilcoxon.p, wilcox_test(sonar_3b_qlora$Reliability, sonar_3b_fft$Reliability))
cliff.delta(sonar_3b_qlora$Reliability, sonar_3b_fft$Reliability)

print("Maintainability - QLoRA vs FFT (3B)")
res_3b$Wilcoxon.p = append(res_3b$Wilcoxon.p, wilcox_test(sonar_3b_qlora$Maintainability, sonar_3b_fft$Maintainability))
cliff.delta(sonar_3b_qlora$Maintainability, sonar_3b_fft$Maintainability)

print("CyC - QLoRA vs FFT (3B)")
res_3b$Wilcoxon.p = append(res_3b$Wilcoxon.p, wilcox_test(sonar_3b_qlora$CyC, sonar_3b_fft$CyC))
cliff.delta(sonar_3b_qlora$CyC, sonar_3b_fft$CyC)

print("CoC - QLoRA vs FFT (3B)")
res_3b$Wilcoxon.p = append(res_3b$Wilcoxon.p, wilcox_test(sonar_3b_qlora$CoC, sonar_3b_fft$CoC))
cliff.delta(sonar_3b_qlora$CoC, sonar_3b_fft$CoC)

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
sink(paste0(output_path, "/rq2_cg_python_mt_qlora_vs_fft_wilcoxon_CLEAN.txt"))

cat("================================================================================\n")
cat("RQ2: Code Generation - Python (MT-QLoRA vs MT-FFT) - Wilcoxon Signed-Rank Test\n")
cat("================================================================================\n\n")

# Metric labels
metric_labels <- c("Lizard_LoC", "Lizard_ToK", "Lizard_DR", "Lizard_CyC",
                   "Pylint_Err", "Pylint_War", "Pylint_Conv", "Pylint_Ref",
                   "Sonar_SecHot", "Sonar_Rel", "Sonar_Main", "Sonar_CyC", "Sonar_CoC")

# Get delta values for 0.5B
delta_0_5b <- c(
  get_delta(lizard_0_5b_qlora$Lines_of_Code, lizard_0_5b_fft$Lines_of_Code),
  get_delta(lizard_0_5b_qlora$Token_Count, lizard_0_5b_fft$Token_Count),
  get_delta(lizard_0_5b_qlora$Detection_Rate, lizard_0_5b_fft$Detection_Rate),
  get_delta(lizard_0_5b_qlora$Cyclomatic_Complexity, lizard_0_5b_fft$Cyclomatic_Complexity),
  get_delta(pylint_0_5b_qlora$Error, pylint_0_5b_fft$Error),
  get_delta(pylint_0_5b_qlora$Warning, pylint_0_5b_fft$Warning),
  get_delta(pylint_0_5b_qlora$Convention, pylint_0_5b_fft$Convention),
  get_delta(pylint_0_5b_qlora$Refactor, pylint_0_5b_fft$Refactor),
  get_delta(sonar_0_5b_qlora$Security_Hotspots, sonar_0_5b_fft$Security_Hotspots),
  get_delta(sonar_0_5b_qlora$Reliability, sonar_0_5b_fft$Reliability),
  get_delta(sonar_0_5b_qlora$Maintainability, sonar_0_5b_fft$Maintainability),
  get_delta(sonar_0_5b_qlora$CyC, sonar_0_5b_fft$CyC),
  get_delta(sonar_0_5b_qlora$CoC, sonar_0_5b_fft$CoC)
)

# Get delta values for 1.5B
delta_1_5b <- c(
  get_delta(lizard_1_5b_qlora$Lines_of_Code, lizard_1_5b_fft$Lines_of_Code),
  get_delta(lizard_1_5b_qlora$Token_Count, lizard_1_5b_fft$Token_Count),
  get_delta(lizard_1_5b_qlora$Detection_Rate, lizard_1_5b_fft$Detection_Rate),
  get_delta(lizard_1_5b_qlora$Cyclomatic_Complexity, lizard_1_5b_fft$Cyclomatic_Complexity),
  get_delta(pylint_1_5b_qlora$Error, pylint_1_5b_fft$Error),
  get_delta(pylint_1_5b_qlora$Warning, pylint_1_5b_fft$Warning),
  get_delta(pylint_1_5b_qlora$Convention, pylint_1_5b_fft$Convention),
  get_delta(pylint_1_5b_qlora$Refactor, pylint_1_5b_fft$Refactor),
  get_delta(sonar_1_5b_qlora$Security_Hotspots, sonar_1_5b_fft$Security_Hotspots),
  get_delta(sonar_1_5b_qlora$Reliability, sonar_1_5b_fft$Reliability),
  get_delta(sonar_1_5b_qlora$Maintainability, sonar_1_5b_fft$Maintainability),
  get_delta(sonar_1_5b_qlora$CyC, sonar_1_5b_fft$CyC),
  get_delta(sonar_1_5b_qlora$CoC, sonar_1_5b_fft$CoC)
)

# Get delta values for 3B
delta_3b <- c(
  get_delta(lizard_3b_qlora$Lines_of_Code, lizard_3b_fft$Lines_of_Code),
  get_delta(lizard_3b_qlora$Token_Count, lizard_3b_fft$Token_Count),
  get_delta(lizard_3b_qlora$Detection_Rate, lizard_3b_fft$Detection_Rate),
  get_delta(lizard_3b_qlora$Cyclomatic_Complexity, lizard_3b_fft$Cyclomatic_Complexity),
  get_delta(pylint_3b_qlora$Error, pylint_3b_fft$Error),
  get_delta(pylint_3b_qlora$Warning, pylint_3b_fft$Warning),
  get_delta(pylint_3b_qlora$Convention, pylint_3b_fft$Convention),
  get_delta(pylint_3b_qlora$Refactor, pylint_3b_fft$Refactor),
  get_delta(sonar_3b_qlora$Security_Hotspots, sonar_3b_fft$Security_Hotspots),
  get_delta(sonar_3b_qlora$Reliability, sonar_3b_fft$Reliability),
  get_delta(sonar_3b_qlora$Maintainability, sonar_3b_fft$Maintainability),
  get_delta(sonar_3b_qlora$CyC, sonar_3b_fft$CyC),
  get_delta(sonar_3b_qlora$CoC, sonar_3b_fft$CoC)
)

# ============== QWEN 0.5B TABLE ==============
cat("------------------------------------------------------------------------------\n")
cat("QWEN 0.5B: MT-QLoRA vs MT-FFT\n")
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
cat("QWEN 1.5B: MT-QLoRA vs MT-FFT\n")
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
cat("QWEN 3B: MT-QLoRA vs MT-FFT\n")
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
print(paste0(output_path, "/rq2_cg_python_mt_qlora_vs_fft_wilcoxon_RAW.txt"))
print(paste0(output_path, "/rq2_cg_python_mt_qlora_vs_fft_wilcoxon_CLEAN.txt"))