rm(list=ls())

if (!require("effsize")) install.packages("effsize")
library(effsize)

# Base path
base_path <- "/scratch/oldhome/mhaque/quantized-model-code-quality/statistical-tests/per-instance-value/llm_judge/py"
output_path <- "/scratch/oldhome/mhaque/quantized-model-code-quality/statistical-tests/R-scripts-v2"

# ============== LOAD DATA ==============
data_0_5b_qlora <- read.csv(paste0(base_path, "/qwen0_5_mt_qlora.csv"), header=TRUE)
data_0_5b_fft <- read.csv(paste0(base_path, "/qwen0_5_mt_fft.csv"), header=TRUE)
data_1_5b_qlora <- read.csv(paste0(base_path, "/qwen1_5_mt_qlora.csv"), header=TRUE)
data_1_5b_fft <- read.csv(paste0(base_path, "/qwen1_5_mt_fft.csv"), header=TRUE)
data_3b_qlora <- read.csv(paste0(base_path, "/qwen3_mt_qlora.csv"), header=TRUE)
data_3b_fft <- read.csv(paste0(base_path, "/qwen3_mt_fft.csv"), header=TRUE)

# ============== ENSURE SAME LENGTH ==============
min_len <- min(nrow(data_0_5b_qlora), nrow(data_0_5b_fft))
data_0_5b_qlora <- data_0_5b_qlora[1:min_len, ]; data_0_5b_fft <- data_0_5b_fft[1:min_len, ]
min_len <- min(nrow(data_1_5b_qlora), nrow(data_1_5b_fft))
data_1_5b_qlora <- data_1_5b_qlora[1:min_len, ]; data_1_5b_fft <- data_1_5b_fft[1:min_len, ]
min_len <- min(nrow(data_3b_qlora), nrow(data_3b_fft))
data_3b_qlora <- data_3b_qlora[1:min_len, ]; data_3b_fft <- data_3b_fft[1:min_len, ]

# ============== HELPER FUNCTIONS ==============
is_constant <- function(x) { return(length(unique(x)) == 1) }

wilcox_test <- function(x, y) {
  if (is_constant(x) & is_constant(y)) return(NA)
  return(wilcox.test(x, y, alternative="two.side", paired=TRUE, exact=FALSE, correct=FALSE)$p.value)
}

get_delta <- function(x, y) { return(cliff.delta(x, y)$estimate) }

# ============== RAW OUTPUT ==============
sink(paste0(output_path, "/rq2_cs_python_llm_judge_mt_qlora_vs_fft_wilcoxon_RAW.txt"))

# QWEN 0.5B
print("********************** QWEN 0.5B: MT-QLoRA vs MT-FFT (Python LLM Judge) **********************")
res_0_5b = list(Wilcoxon.p = c())

print("CA - QLoRA vs FFT (0.5B)")
res_0_5b$Wilcoxon.p = append(res_0_5b$Wilcoxon.p, wilcox_test(data_0_5b_qlora$CA, data_0_5b_fft$CA))
cliff.delta(data_0_5b_qlora$CA, data_0_5b_fft$CA)

print("Conciseness - QLoRA vs FFT (0.5B)")
res_0_5b$Wilcoxon.p = append(res_0_5b$Wilcoxon.p, wilcox_test(data_0_5b_qlora$Conciseness, data_0_5b_fft$Conciseness))
cliff.delta(data_0_5b_qlora$Conciseness, data_0_5b_fft$Conciseness)

print("Fluency - QLoRA vs FFT (0.5B)")
res_0_5b$Wilcoxon.p = append(res_0_5b$Wilcoxon.p, wilcox_test(data_0_5b_qlora$Fluency, data_0_5b_fft$Fluency))
cliff.delta(data_0_5b_qlora$Fluency, data_0_5b_fft$Fluency)

# QWEN 1.5B
print("********************** QWEN 1.5B: MT-QLoRA vs MT-FFT (Python LLM Judge) **********************")
res_1_5b = list(Wilcoxon.p = c())

print("CA - QLoRA vs FFT (1.5B)")
res_1_5b$Wilcoxon.p = append(res_1_5b$Wilcoxon.p, wilcox_test(data_1_5b_qlora$CA, data_1_5b_fft$CA))
cliff.delta(data_1_5b_qlora$CA, data_1_5b_fft$CA)

print("Conciseness - QLoRA vs FFT (1.5B)")
res_1_5b$Wilcoxon.p = append(res_1_5b$Wilcoxon.p, wilcox_test(data_1_5b_qlora$Conciseness, data_1_5b_fft$Conciseness))
cliff.delta(data_1_5b_qlora$Conciseness, data_1_5b_fft$Conciseness)

print("Fluency - QLoRA vs FFT (1.5B)")
res_1_5b$Wilcoxon.p = append(res_1_5b$Wilcoxon.p, wilcox_test(data_1_5b_qlora$Fluency, data_1_5b_fft$Fluency))
cliff.delta(data_1_5b_qlora$Fluency, data_1_5b_fft$Fluency)

# QWEN 3B
print("********************** QWEN 3B: MT-QLoRA vs MT-FFT (Python LLM Judge) **********************")
res_3b = list(Wilcoxon.p = c())

print("CA - QLoRA vs FFT (3B)")
res_3b$Wilcoxon.p = append(res_3b$Wilcoxon.p, wilcox_test(data_3b_qlora$CA, data_3b_fft$CA))
cliff.delta(data_3b_qlora$CA, data_3b_fft$CA)

print("Conciseness - QLoRA vs FFT (3B)")
res_3b$Wilcoxon.p = append(res_3b$Wilcoxon.p, wilcox_test(data_3b_qlora$Conciseness, data_3b_fft$Conciseness))
cliff.delta(data_3b_qlora$Conciseness, data_3b_fft$Conciseness)

print("Fluency - QLoRA vs FFT (3B)")
res_3b$Wilcoxon.p = append(res_3b$Wilcoxon.p, wilcox_test(data_3b_qlora$Fluency, data_3b_fft$Fluency))
cliff.delta(data_3b_qlora$Fluency, data_3b_fft$Fluency)

# Apply holm correction
print("***************************************************************************")
res_df_0_5b = data.frame(Wilcoxon.p = res_0_5b$Wilcoxon.p)
res_df_0_5b$Wilcoxon.p = p.adjust(res_df_0_5b$Wilcoxon.p, method="holm")
print("Qwen 0.5B Results (holm Adjusted P-values):"); print(res_df_0_5b)

res_df_1_5b = data.frame(Wilcoxon.p = res_1_5b$Wilcoxon.p)
res_df_1_5b$Wilcoxon.p = p.adjust(res_df_1_5b$Wilcoxon.p, method="holm")
print("Qwen 1.5B Results (holm Adjusted P-values):"); print(res_df_1_5b)

res_df_3b = data.frame(Wilcoxon.p = res_3b$Wilcoxon.p)
res_df_3b$Wilcoxon.p = p.adjust(res_df_3b$Wilcoxon.p, method="holm")
print("Qwen 3B Results (holm Adjusted P-values):"); print(res_df_3b)

sink()

# ============== CLEAN OUTPUT ==============
sink(paste0(output_path, "/rq2_cs_python_llm_judge_mt_qlora_vs_fft_wilcoxon_CLEAN.txt"))

cat("================================================================================\n")
cat("RQ2: Code Summarization - Python LLM Judge (MT-QLoRA vs MT-FFT) - Wilcoxon Test\n")
cat("================================================================================\n\n")

metric_labels <- c("CA", "Conciseness", "Fluency")

delta_0_5b <- c(
  get_delta(data_0_5b_qlora$CA, data_0_5b_fft$CA),
  get_delta(data_0_5b_qlora$Conciseness, data_0_5b_fft$Conciseness),
  get_delta(data_0_5b_qlora$Fluency, data_0_5b_fft$Fluency)
)

delta_1_5b <- c(
  get_delta(data_1_5b_qlora$CA, data_1_5b_fft$CA),
  get_delta(data_1_5b_qlora$Conciseness, data_1_5b_fft$Conciseness),
  get_delta(data_1_5b_qlora$Fluency, data_1_5b_fft$Fluency)
)

delta_3b <- c(
  get_delta(data_3b_qlora$CA, data_3b_fft$CA),
  get_delta(data_3b_qlora$Conciseness, data_3b_fft$Conciseness),
  get_delta(data_3b_qlora$Fluency, data_3b_fft$Fluency)
)

# QWEN 0.5B TABLE
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
  cat(sprintf("%-15s %12s %12.4f %12s\n", metric_labels[i], p_display, d_val, sig))
}
cat("\n")

# QWEN 1.5B TABLE
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
  cat(sprintf("%-15s %12s %12.4f %12s\n", metric_labels[i], p_display, d_val, sig))
}
cat("\n")

# QWEN 3B TABLE
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
  cat(sprintf("%-15s %12s %12.4f %12s\n", metric_labels[i], p_display, d_val, sig))
}
cat("\n")

cat("================================================================================\n")
cat("Note: p-values are Benjamini-Hochberg (FDR) corrected\n")
cat("Cliff's Delta: |d| < 0.147 (N), < 0.33 (S), < 0.474 (M), >= 0.474 (L)\n")
cat("================================================================================\n")

sink()

print("Results saved to:")
print(paste0(output_path, "/rq2_cs_python_llm_judge_mt_qlora_vs_fft_wilcoxon_RAW.txt"))
print(paste0(output_path, "/rq2_cs_python_llm_judge_mt_qlora_vs_fft_wilcoxon_CLEAN.txt"))