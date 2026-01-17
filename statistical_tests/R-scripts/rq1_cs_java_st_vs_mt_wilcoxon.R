rm(list=ls())

if (!require("effsize")) install.packages("effsize")
library(effsize)

# Base path
base_path <- "/scratch/oldhome/mhaque/quantized-model-code-quality/statistical-tests/per-instance-value"
output_path <- "/scratch/oldhome/mhaque/quantized-model-code-quality/statistical-tests/R-scripts-v2"

# ============== LOAD DATA ==============
# Qwen 0.5B
data_0_5b_st <- read.csv(paste0(base_path, "/bleu_meteor_rouge_chrf_bertscore_side/summarization/java/cs_qwen0_5_st_qlora.csv"), header=TRUE)
data_0_5b_mt <- read.csv(paste0(base_path, "/bleu_meteor_rouge_chrf_bertscore_side/summarization/java/cs_qwen0_5_mt_qlora.csv"), header=TRUE)

# Qwen 1.5B
data_1_5b_st <- read.csv(paste0(base_path, "/bleu_meteor_rouge_chrf_bertscore_side/summarization/java/cs_qwen1_5_st_qlora.csv"), header=TRUE)
data_1_5b_mt <- read.csv(paste0(base_path, "/bleu_meteor_rouge_chrf_bertscore_side/summarization/java/cs_qwen1_5_mt_qlora.csv"), header=TRUE)

# Qwen 3B
data_3b_st <- read.csv(paste0(base_path, "/bleu_meteor_rouge_chrf_bertscore_side/summarization/java/cs_qwen3_st_qlora.csv"), header=TRUE)
data_3b_mt <- read.csv(paste0(base_path, "/bleu_meteor_rouge_chrf_bertscore_side/summarization/java/cs_qwen3_mt_qlora.csv"), header=TRUE)

# ============== ENSURE SAME LENGTH ==============
min_len <- min(nrow(data_0_5b_st), nrow(data_0_5b_mt))
data_0_5b_st <- data_0_5b_st[1:min_len, ]; data_0_5b_mt <- data_0_5b_mt[1:min_len, ]

min_len <- min(nrow(data_1_5b_st), nrow(data_1_5b_mt))
data_1_5b_st <- data_1_5b_st[1:min_len, ]; data_1_5b_mt <- data_1_5b_mt[1:min_len, ]

min_len <- min(nrow(data_3b_st), nrow(data_3b_mt))
data_3b_st <- data_3b_st[1:min_len, ]; data_3b_mt <- data_3b_mt[1:min_len, ]

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
sink(paste0(output_path, "/rq1_cs_java_st_vs_mt_wilcoxon_RAW.txt"))

# ============== QWEN 0.5B ==============
print("********************** QWEN 0.5B: ST-QLoRA vs MT-QLoRA (Java Summarization) *********************************")

res_0_5b = list(Wilcoxon.p = c())

print("BLEU - ST vs MT (0.5B)")
res_0_5b$Wilcoxon.p = append(res_0_5b$Wilcoxon.p, wilcox_test(data_0_5b_st$BLEU, data_0_5b_mt$BLEU))
cliff.delta(data_0_5b_st$BLEU, data_0_5b_mt$BLEU)

print("METEOR - ST vs MT (0.5B)")
res_0_5b$Wilcoxon.p = append(res_0_5b$Wilcoxon.p, wilcox_test(data_0_5b_st$METEOR, data_0_5b_mt$METEOR))
cliff.delta(data_0_5b_st$METEOR, data_0_5b_mt$METEOR)

print("Rouge-L - ST vs MT (0.5B)")
res_0_5b$Wilcoxon.p = append(res_0_5b$Wilcoxon.p, wilcox_test(data_0_5b_st$Rouge.L, data_0_5b_mt$Rouge.L))
cliff.delta(data_0_5b_st$Rouge.L, data_0_5b_mt$Rouge.L)

print("chrF - ST vs MT (0.5B)")
res_0_5b$Wilcoxon.p = append(res_0_5b$Wilcoxon.p, wilcox_test(data_0_5b_st$chrF, data_0_5b_mt$chrF))
cliff.delta(data_0_5b_st$chrF, data_0_5b_mt$chrF)

print("BERTScore - ST vs MT (0.5B)")
res_0_5b$Wilcoxon.p = append(res_0_5b$Wilcoxon.p, wilcox_test(data_0_5b_st$BERTScore, data_0_5b_mt$BERTScore))
cliff.delta(data_0_5b_st$BERTScore, data_0_5b_mt$BERTScore)

print("SIDE - ST vs MT (0.5B)")
res_0_5b$Wilcoxon.p = append(res_0_5b$Wilcoxon.p, wilcox_test(data_0_5b_st$SIDE, data_0_5b_mt$SIDE))
cliff.delta(data_0_5b_st$SIDE, data_0_5b_mt$SIDE)

# ============== QWEN 1.5B ==============
print("********************** QWEN 1.5B: ST-QLoRA vs MT-QLoRA (Java Summarization) *********************************")

res_1_5b = list(Wilcoxon.p = c())

print("BLEU - ST vs MT (1.5B)")
res_1_5b$Wilcoxon.p = append(res_1_5b$Wilcoxon.p, wilcox_test(data_1_5b_st$BLEU, data_1_5b_mt$BLEU))
cliff.delta(data_1_5b_st$BLEU, data_1_5b_mt$BLEU)

print("METEOR - ST vs MT (1.5B)")
res_1_5b$Wilcoxon.p = append(res_1_5b$Wilcoxon.p, wilcox_test(data_1_5b_st$METEOR, data_1_5b_mt$METEOR))
cliff.delta(data_1_5b_st$METEOR, data_1_5b_mt$METEOR)

print("Rouge-L - ST vs MT (1.5B)")
res_1_5b$Wilcoxon.p = append(res_1_5b$Wilcoxon.p, wilcox_test(data_1_5b_st$Rouge.L, data_1_5b_mt$Rouge.L))
cliff.delta(data_1_5b_st$Rouge.L, data_1_5b_mt$Rouge.L)

print("chrF - ST vs MT (1.5B)")
res_1_5b$Wilcoxon.p = append(res_1_5b$Wilcoxon.p, wilcox_test(data_1_5b_st$chrF, data_1_5b_mt$chrF))
cliff.delta(data_1_5b_st$chrF, data_1_5b_mt$chrF)

print("BERTScore - ST vs MT (1.5B)")
res_1_5b$Wilcoxon.p = append(res_1_5b$Wilcoxon.p, wilcox_test(data_1_5b_st$BERTScore, data_1_5b_mt$BERTScore))
cliff.delta(data_1_5b_st$BERTScore, data_1_5b_mt$BERTScore)

print("SIDE - ST vs MT (1.5B)")
res_1_5b$Wilcoxon.p = append(res_1_5b$Wilcoxon.p, wilcox_test(data_1_5b_st$SIDE, data_1_5b_mt$SIDE))
cliff.delta(data_1_5b_st$SIDE, data_1_5b_mt$SIDE)

# ============== QWEN 3B ==============
print("********************** QWEN 3B: ST-QLoRA vs MT-QLoRA (Java Summarization) *********************************")

res_3b = list(Wilcoxon.p = c())

print("BLEU - ST vs MT (3B)")
res_3b$Wilcoxon.p = append(res_3b$Wilcoxon.p, wilcox_test(data_3b_st$BLEU, data_3b_mt$BLEU))
cliff.delta(data_3b_st$BLEU, data_3b_mt$BLEU)

print("METEOR - ST vs MT (3B)")
res_3b$Wilcoxon.p = append(res_3b$Wilcoxon.p, wilcox_test(data_3b_st$METEOR, data_3b_mt$METEOR))
cliff.delta(data_3b_st$METEOR, data_3b_mt$METEOR)

print("Rouge-L - ST vs MT (3B)")
res_3b$Wilcoxon.p = append(res_3b$Wilcoxon.p, wilcox_test(data_3b_st$Rouge.L, data_3b_mt$Rouge.L))
cliff.delta(data_3b_st$Rouge.L, data_3b_mt$Rouge.L)

print("chrF - ST vs MT (3B)")
res_3b$Wilcoxon.p = append(res_3b$Wilcoxon.p, wilcox_test(data_3b_st$chrF, data_3b_mt$chrF))
cliff.delta(data_3b_st$chrF, data_3b_mt$chrF)

print("BERTScore - ST vs MT (3B)")
res_3b$Wilcoxon.p = append(res_3b$Wilcoxon.p, wilcox_test(data_3b_st$BERTScore, data_3b_mt$BERTScore))
cliff.delta(data_3b_st$BERTScore, data_3b_mt$BERTScore)

print("SIDE - ST vs MT (3B)")
res_3b$Wilcoxon.p = append(res_3b$Wilcoxon.p, wilcox_test(data_3b_st$SIDE, data_3b_mt$SIDE))
cliff.delta(data_3b_st$SIDE, data_3b_mt$SIDE)

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
sink(paste0(output_path, "/rq1_cs_java_st_vs_mt_wilcoxon_CLEAN.txt"))

cat("================================================================================\n")
cat("RQ1: Code Summarization - Java (ST-QLoRA vs MT-QLoRA) - Wilcoxon Signed-Rank Test\n")
cat("================================================================================\n\n")

# Metric labels
metric_labels <- c("BLEU", "METEOR", "Rouge-L", "chrF", "BERTScore", "SIDE")

# Get delta values for 0.5B
delta_0_5b <- c(
  get_delta(data_0_5b_st$BLEU, data_0_5b_mt$BLEU),
  get_delta(data_0_5b_st$METEOR, data_0_5b_mt$METEOR),
  get_delta(data_0_5b_st$Rouge.L, data_0_5b_mt$Rouge.L),
  get_delta(data_0_5b_st$chrF, data_0_5b_mt$chrF),
  get_delta(data_0_5b_st$BERTScore, data_0_5b_mt$BERTScore),
  get_delta(data_0_5b_st$SIDE, data_0_5b_mt$SIDE)
)

# Get delta values for 1.5B
delta_1_5b <- c(
  get_delta(data_1_5b_st$BLEU, data_1_5b_mt$BLEU),
  get_delta(data_1_5b_st$METEOR, data_1_5b_mt$METEOR),
  get_delta(data_1_5b_st$Rouge.L, data_1_5b_mt$Rouge.L),
  get_delta(data_1_5b_st$chrF, data_1_5b_mt$chrF),
  get_delta(data_1_5b_st$BERTScore, data_1_5b_mt$BERTScore),
  get_delta(data_1_5b_st$SIDE, data_1_5b_mt$SIDE)
)

# Get delta values for 3B
delta_3b <- c(
  get_delta(data_3b_st$BLEU, data_3b_mt$BLEU),
  get_delta(data_3b_st$METEOR, data_3b_mt$METEOR),
  get_delta(data_3b_st$Rouge.L, data_3b_mt$Rouge.L),
  get_delta(data_3b_st$chrF, data_3b_mt$chrF),
  get_delta(data_3b_st$BERTScore, data_3b_mt$BERTScore),
  get_delta(data_3b_st$SIDE, data_3b_mt$SIDE)
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
print(paste0(output_path, "/rq1_cs_java_st_vs_mt_wilcoxon_RAW.txt"))
print(paste0(output_path, "/rq1_cs_java_st_vs_mt_wilcoxon_CLEAN.txt"))