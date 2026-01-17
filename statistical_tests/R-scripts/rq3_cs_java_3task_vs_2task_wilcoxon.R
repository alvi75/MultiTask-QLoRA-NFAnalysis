rm(list=ls())

if (!require("effsize")) install.packages("effsize")
library(effsize)

# Base path
base_path <- "/scratch/oldhome/mhaque/quantized-model-code-quality/statistical-tests/per-instance-value"
output_path <- "/scratch/oldhome/mhaque/quantized-model-code-quality/statistical-tests/R-scripts-v2"

# ============== LOAD DATA ==============
# 3-task (CG+CS+CT)
data_3b_3task <- read.csv(paste0(base_path, "/bleu_meteor_rouge_chrf_bertscore_side/summarization/java/cs_qwen3_mt_qlora.csv"), header=TRUE)
# 2-task combinations
data_3b_cg_cs <- read.csv(paste0(base_path, "/bleu_meteor_rouge_chrf_bertscore_side/summarization/java/cs_qwen3_cg_cs_mt_qlora.csv"), header=TRUE)
data_3b_cs_ct <- read.csv(paste0(base_path, "/bleu_meteor_rouge_chrf_bertscore_side/summarization/java/cs_qwen3_cs_ct_mt_qlora.csv"), header=TRUE)

# ============== ENSURE SAME LENGTH ==============
# 3-task vs CG+CS
min_len <- min(nrow(data_3b_3task), nrow(data_3b_cg_cs))
data_3b_3task_cgcs <- data_3b_3task[1:min_len, ]; data_3b_cg_cs <- data_3b_cg_cs[1:min_len, ]

# 3-task vs CS+CT
min_len <- min(nrow(data_3b_3task), nrow(data_3b_cs_ct))
data_3b_3task_csct <- data_3b_3task[1:min_len, ]; data_3b_cs_ct <- data_3b_cs_ct[1:min_len, ]

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
sink(paste0(output_path, "/rq3_cs_java_3task_vs_2task_wilcoxon_RAW.txt"))

# ============== COMPARISON 1: 3-task vs CG+CS ==============
print("********************** QWEN 3B: MT-QLoRA (3-task) vs MT-QLoRA (CG+CS) (Java Summarization) *********************************")

res_3task_vs_cgcs = list(Wilcoxon.p = c())

print("BLEU - 3-task vs CG+CS (3B)")
res_3task_vs_cgcs$Wilcoxon.p = append(res_3task_vs_cgcs$Wilcoxon.p, wilcox_test(data_3b_3task_cgcs$BLEU, data_3b_cg_cs$BLEU))
cliff.delta(data_3b_3task_cgcs$BLEU, data_3b_cg_cs$BLEU)

print("METEOR - 3-task vs CG+CS (3B)")
res_3task_vs_cgcs$Wilcoxon.p = append(res_3task_vs_cgcs$Wilcoxon.p, wilcox_test(data_3b_3task_cgcs$METEOR, data_3b_cg_cs$METEOR))
cliff.delta(data_3b_3task_cgcs$METEOR, data_3b_cg_cs$METEOR)

print("Rouge-L - 3-task vs CG+CS (3B)")
res_3task_vs_cgcs$Wilcoxon.p = append(res_3task_vs_cgcs$Wilcoxon.p, wilcox_test(data_3b_3task_cgcs$Rouge.L, data_3b_cg_cs$Rouge.L))
cliff.delta(data_3b_3task_cgcs$Rouge.L, data_3b_cg_cs$Rouge.L)

print("chrF - 3-task vs CG+CS (3B)")
res_3task_vs_cgcs$Wilcoxon.p = append(res_3task_vs_cgcs$Wilcoxon.p, wilcox_test(data_3b_3task_cgcs$chrF, data_3b_cg_cs$chrF))
cliff.delta(data_3b_3task_cgcs$chrF, data_3b_cg_cs$chrF)

print("BERTScore - 3-task vs CG+CS (3B)")
res_3task_vs_cgcs$Wilcoxon.p = append(res_3task_vs_cgcs$Wilcoxon.p, wilcox_test(data_3b_3task_cgcs$BERTScore, data_3b_cg_cs$BERTScore))
cliff.delta(data_3b_3task_cgcs$BERTScore, data_3b_cg_cs$BERTScore)

print("SIDE - 3-task vs CG+CS (3B)")
res_3task_vs_cgcs$Wilcoxon.p = append(res_3task_vs_cgcs$Wilcoxon.p, wilcox_test(data_3b_3task_cgcs$SIDE, data_3b_cg_cs$SIDE))
cliff.delta(data_3b_3task_cgcs$SIDE, data_3b_cg_cs$SIDE)

# ============== COMPARISON 2: 3-task vs CS+CT ==============
print("********************** QWEN 3B: MT-QLoRA (3-task) vs MT-QLoRA (CS+CT) (Java Summarization) *********************************")

res_3task_vs_csct = list(Wilcoxon.p = c())

print("BLEU - 3-task vs CS+CT (3B)")
res_3task_vs_csct$Wilcoxon.p = append(res_3task_vs_csct$Wilcoxon.p, wilcox_test(data_3b_3task_csct$BLEU, data_3b_cs_ct$BLEU))
cliff.delta(data_3b_3task_csct$BLEU, data_3b_cs_ct$BLEU)

print("METEOR - 3-task vs CS+CT (3B)")
res_3task_vs_csct$Wilcoxon.p = append(res_3task_vs_csct$Wilcoxon.p, wilcox_test(data_3b_3task_csct$METEOR, data_3b_cs_ct$METEOR))
cliff.delta(data_3b_3task_csct$METEOR, data_3b_cs_ct$METEOR)

print("Rouge-L - 3-task vs CS+CT (3B)")
res_3task_vs_csct$Wilcoxon.p = append(res_3task_vs_csct$Wilcoxon.p, wilcox_test(data_3b_3task_csct$Rouge.L, data_3b_cs_ct$Rouge.L))
cliff.delta(data_3b_3task_csct$Rouge.L, data_3b_cs_ct$Rouge.L)

print("chrF - 3-task vs CS+CT (3B)")
res_3task_vs_csct$Wilcoxon.p = append(res_3task_vs_csct$Wilcoxon.p, wilcox_test(data_3b_3task_csct$chrF, data_3b_cs_ct$chrF))
cliff.delta(data_3b_3task_csct$chrF, data_3b_cs_ct$chrF)

print("BERTScore - 3-task vs CS+CT (3B)")
res_3task_vs_csct$Wilcoxon.p = append(res_3task_vs_csct$Wilcoxon.p, wilcox_test(data_3b_3task_csct$BERTScore, data_3b_cs_ct$BERTScore))
cliff.delta(data_3b_3task_csct$BERTScore, data_3b_cs_ct$BERTScore)

print("SIDE - 3-task vs CS+CT (3B)")
res_3task_vs_csct$Wilcoxon.p = append(res_3task_vs_csct$Wilcoxon.p, wilcox_test(data_3b_3task_csct$SIDE, data_3b_cs_ct$SIDE))
cliff.delta(data_3b_3task_csct$SIDE, data_3b_cs_ct$SIDE)

# ============== PRINT RAW RESULTS ==============
print("***************************************************************************")

res_df_3task_vs_cgcs = data.frame(Wilcoxon.p = res_3task_vs_cgcs$Wilcoxon.p)
res_df_3task_vs_cgcs$Wilcoxon.p = p.adjust(res_df_3task_vs_cgcs$Wilcoxon.p, method="holm")
print("3-task vs CG+CS Results (Adjusted P-values):")
print(res_df_3task_vs_cgcs)

res_df_3task_vs_csct = data.frame(Wilcoxon.p = res_3task_vs_csct$Wilcoxon.p)
res_df_3task_vs_csct$Wilcoxon.p = p.adjust(res_df_3task_vs_csct$Wilcoxon.p, method="holm")
print("3-task vs CS+CT Results (Adjusted P-values):")
print(res_df_3task_vs_csct)

sink()

# ============================================================================
# FILE 2: CLEAN SORTED OUTPUT (table format)
# ============================================================================
sink(paste0(output_path, "/rq3_cs_java_3task_vs_2task_wilcoxon_CLEAN.txt"))

cat("================================================================================\n")
cat("RQ3: Code Summarization - Java (3-task vs 2-task) - Wilcoxon Signed-Rank Test\n")
cat("================================================================================\n\n")

# Metric labels
metric_labels <- c("BLEU", "METEOR", "Rouge-L", "chrF", "BERTScore", "SIDE")

# Get delta values for 3-task vs CG+CS
delta_3task_vs_cgcs <- c(
  get_delta(data_3b_3task_cgcs$BLEU, data_3b_cg_cs$BLEU),
  get_delta(data_3b_3task_cgcs$METEOR, data_3b_cg_cs$METEOR),
  get_delta(data_3b_3task_cgcs$Rouge.L, data_3b_cg_cs$Rouge.L),
  get_delta(data_3b_3task_cgcs$chrF, data_3b_cg_cs$chrF),
  get_delta(data_3b_3task_cgcs$BERTScore, data_3b_cg_cs$BERTScore),
  get_delta(data_3b_3task_cgcs$SIDE, data_3b_cg_cs$SIDE)
)

# Get delta values for 3-task vs CS+CT
delta_3task_vs_csct <- c(
  get_delta(data_3b_3task_csct$BLEU, data_3b_cs_ct$BLEU),
  get_delta(data_3b_3task_csct$METEOR, data_3b_cs_ct$METEOR),
  get_delta(data_3b_3task_csct$Rouge.L, data_3b_cs_ct$Rouge.L),
  get_delta(data_3b_3task_csct$chrF, data_3b_cs_ct$chrF),
  get_delta(data_3b_3task_csct$BERTScore, data_3b_cs_ct$BERTScore),
  get_delta(data_3b_3task_csct$SIDE, data_3b_cs_ct$SIDE)
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
print(paste0(output_path, "/rq3_cs_java_3task_vs_2task_wilcoxon_RAW.txt"))
print(paste0(output_path, "/rq3_cs_java_3task_vs_2task_wilcoxon_CLEAN.txt"))