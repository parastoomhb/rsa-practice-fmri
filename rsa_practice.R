# =====================================================================
# Representational Similarity Analysis (RSA) - practice script
# Data: rsa_practice_fmri_patterns.csv
#   20 subjects x 2 runs x 8 conditions (Self/Other x Pos/Neg/Neutral/Ambiguous)
#   100 voxels per pattern (think: one ROI)
#
# Pipeline
#   1. Load data & build condition x voxel pattern matrices
#   2. Neural RDMs (1 - Pearson r between condition patterns), per subject
#   3. Group-average RDM -> heatmap, MDS, dendrogram
#   4. Model RDMs (Target, Valence, Both)
#   5. Compare neural RDMs to models (Spearman, Fisher z, t-tests) + noise ceiling
#   6. Extra: cross-run reliability, within/between-condition distances
# =====================================================================

# ---- 0. Setup -------------------------------------------------------
# install.packages(c("tidyverse"))   # run once if needed
library(tidyverse)

setwd("C:/Users/ASUS/Desktop/PhD-NRU/Synthetic data")   # forward slashes work in R on Windows
dir.create("rsa_output", showWarnings = FALSE)           # plots are saved here

# ---- 1. Load data ---------------------------------------------------
dat <- read_csv("rsa_practice_fmri_patterns.csv", show_col_types = FALSE)

voxel_cols <- grep("^voxel_", names(dat), value = TRUE)

cond_order <- c("Self_Pos",  "Self_Neg",  "Self_Neutral",  "Self_Ambiguous",
                "Other_Pos", "Other_Neg", "Other_Neutral", "Other_Ambiguous")
dat$condition <- factor(dat$condition, levels = cond_order)

subjects <- sort(unique(dat$subject))

glimpse(dat[, 1:8])   # quick look at the data

# Helper: average rows of one subject (optionally one run) into an
# 8 (conditions) x 100 (voxels) matrix
get_patterns <- function(d) {
  d %>%
    group_by(condition) %>%
    summarise(across(all_of(voxel_cols), mean), .groups = "drop") %>%
    arrange(condition) -> out
  m <- as.matrix(out[, voxel_cols])
  rownames(m) <- as.character(out$condition)
  m
}

# Optional: remove the mean pattern across conditions (a common step so
# that RDMs reflect differences between conditions, not the shared response)
remove_mean_pattern <- FALSE

patterns <- lapply(subjects, function(s) {
  m <- get_patterns(filter(dat, subject == s))
  if (remove_mean_pattern) m <- sweep(m, 2, colMeans(m))
  m
})
names(patterns) <- subjects

# ---- 2. Neural RDMs (one per subject) --------------------------------
# Correlation distance = 1 - r. cor() works on columns, so transpose
# so that each condition becomes a column.
rdms <- lapply(patterns, function(m) 1 - cor(t(m)))

# Group-average RDM
group_rdm <- Reduce(`+`, rdms) / length(rdms)
round(group_rdm, 2)

# Helper to get the unique off-diagonal values (lower triangle) of an RDM
lt <- function(m) m[lower.tri(m)]

# Helper to plot any RDM as a heatmap
plot_rdm <- function(m, title = "", limits = NULL, legend_name = "Dissimilarity") {
  df <- as.data.frame(as.table(m))
  names(df) <- c("row", "col", "value")
  df$row <- factor(df$row, levels = rev(rownames(m)))   # first condition at top
  df$col <- factor(df$col, levels = colnames(m))
  ggplot(df, aes(col, row, fill = value)) +
    geom_tile(colour = "white") +
    scale_fill_viridis_c(name = legend_name, limits = limits) +
    coord_fixed() +
    labs(title = title, x = NULL, y = NULL) +
    theme_minimal(base_size = 12) +
    theme(axis.text.x = element_text(angle = 45, hjust = 1),
          panel.grid = element_blank())
}

# ---- 3. Visualise the neural RDM ------------------------------------
# 3a. Group-average RDM
p_group <- plot_rdm(group_rdm, "Group-average neural RDM (1 - r)")
print(p_group)
ggsave("rsa_output/01_group_rdm.png", p_group, width = 6.5, height = 5.5, dpi = 300)

# 3b. Every subject's RDM (see how much subjects differ)
subj_long <- map_dfr(subjects, function(s) {
  df <- as.data.frame(as.table(rdms[[s]]))
  names(df) <- c("row", "col", "value")
  df$subject <- s
  df
})
subj_long$row <- factor(subj_long$row, levels = rev(cond_order))
subj_long$col <- factor(subj_long$col, levels = cond_order)

p_subj <- ggplot(subj_long, aes(col, row, fill = value)) +
  geom_tile() +
  scale_fill_viridis_c(name = "1 - r") +
  facet_wrap(~ subject, ncol = 5) +
  coord_fixed() +
  labs(title = "Single-subject neural RDMs", x = NULL, y = NULL) +
  theme_minimal(base_size = 8) +
  theme(axis.text = element_blank(), panel.grid = element_blank())
print(p_subj)
ggsave("rsa_output/02_subject_rdms.png", p_subj, width = 10, height = 8, dpi = 300)

# 3c. Multidimensional scaling: 2D map where distance ~ dissimilarity
mds <- cmdscale(as.dist(group_rdm), k = 2)
mds_df <- tibble(condition = rownames(mds), Dim1 = mds[, 1], Dim2 = mds[, 2]) %>%
  separate(condition, into = c("Target", "Valence"), sep = "_", remove = FALSE)

p_mds <- ggplot(mds_df, aes(Dim1, Dim2, colour = Target, shape = Valence, label = condition)) +
  geom_point(size = 5) +
  geom_text(vjust = -1.2, show.legend = FALSE, size = 3.5) +
  labs(title = "MDS of the group-average RDM",
       subtitle = "Closer points = more similar neural patterns") +
  theme_minimal(base_size = 12) +
  coord_equal()
print(p_mds)
ggsave("rsa_output/03_mds.png", p_mds, width = 7, height = 6, dpi = 300)

# 3d. Hierarchical clustering
png("rsa_output/04_dendrogram.png", width = 1800, height = 1400, res = 250)
plot(hclust(as.dist(group_rdm), method = "average"),
     main = "Hierarchical clustering of conditions", xlab = "", sub = "")
dev.off()
plot(hclust(as.dist(group_rdm), method = "average"),
     main = "Hierarchical clustering of conditions", xlab = "", sub = "")

# ---- 4. Model RDMs ---------------------------------------------------
# A model RDM is your hypothesis: 0 = predicted similar, 1 = predicted different
cond_info <- tibble(condition = cond_order) %>%
  separate(condition, into = c("target", "valence"), sep = "_", remove = FALSE)

model_target  <- 1 * outer(cond_info$target,  cond_info$target,  "!=")  # Self vs Other
model_valence <- 1 * outer(cond_info$valence, cond_info$valence, "!=")  # Pos/Neg/Neutral/Ambig
model_both    <- (model_target + model_valence) / 2                     # both factors matter

models <- list(Target = model_target, Valence = model_valence, Both = model_both)
for (nm in names(models)) dimnames(models[[nm]]) <- list(cond_order, cond_order)

# Plot the three model RDMs side by side
models_long <- map_dfr(names(models), function(nm) {
  df <- as.data.frame(as.table(models[[nm]]))
  names(df) <- c("row", "col", "value")
  df$model <- nm
  df
})
models_long$row <- factor(models_long$row, levels = rev(cond_order))
models_long$col <- factor(models_long$col, levels = cond_order)
models_long$model <- factor(models_long$model, levels = names(models))

p_models <- ggplot(models_long, aes(col, row, fill = value)) +
  geom_tile(colour = "white") +
  scale_fill_viridis_c(name = "Predicted\ndissimilarity") +
  facet_wrap(~ model) +
  coord_fixed() +
  labs(title = "Model RDMs", x = NULL, y = NULL) +
  theme_minimal(base_size = 10) +
  theme(axis.text.x = element_text(angle = 60, hjust = 1), panel.grid = element_blank())
print(p_models)
ggsave("rsa_output/05_model_rdms.png", p_models, width = 10, height = 4, dpi = 300)

# ---- 5. Compare neural RDMs with model RDMs ---------------------------
# Spearman correlation between the lower triangles (rank-based, so we do
# not assume a linear relationship between distance and model).
fit <- map_dfr(subjects, function(s) {
  tibble(subject = s,
         model   = names(models),
         rho     = map_dbl(models, ~ cor(lt(rdms[[s]]), lt(.x), method = "spearman")))
})
fit$z <- atanh(fit$rho)                       # Fisher z before running t-tests
fit$model <- factor(fit$model, levels = names(models))

# 5a. Is each model's fit > 0 across subjects? (one-sample t-test on z)
stats_one <- fit %>%
  group_by(model) %>%
  summarise(mean_rho = mean(rho),
            t        = t.test(z, mu = 0)$statistic,
            df       = t.test(z, mu = 0)$parameter,
            p        = t.test(z, mu = 0)$p.value,
            .groups  = "drop") %>%
  mutate(p_holm = p.adjust(p, method = "holm"))
print(stats_one)

# 5b. Do models differ from each other? (paired t-tests)
fit_wide <- fit %>% select(subject, model, z) %>% pivot_wider(names_from = model, values_from = z)
cat("\nTarget vs Valence:\n"); print(t.test(fit_wide$Target, fit_wide$Valence, paired = TRUE))
cat("\nBoth vs Target:\n");    print(t.test(fit_wide$Both,   fit_wide$Target,  paired = TRUE))
cat("\nBoth vs Valence:\n");   print(t.test(fit_wide$Both,   fit_wide$Valence, paired = TRUE))

# 5c. Noise ceiling: how well could ANY model do, given between-subject noise?
#   upper = each subject vs. group mean INCLUDING that subject
#   lower = each subject vs. group mean of the OTHER subjects (leave-one-out)
nc <- map_dfr(subjects, function(s) {
  others <- Reduce(`+`, rdms[setdiff(subjects, s)]) / (length(subjects) - 1)
  tibble(upper = cor(lt(rdms[[s]]), lt(group_rdm), method = "spearman"),
         lower = cor(lt(rdms[[s]]), lt(others),    method = "spearman"))
})
nc_lower <- mean(nc$lower); nc_upper <- mean(nc$upper)
cat(sprintf("\nNoise ceiling: lower = %.3f, upper = %.3f\n", nc_lower, nc_upper))

# 5d. Plot model fit with noise ceiling
p_fit <- ggplot(fit, aes(model, rho)) +
  annotate("rect", xmin = -Inf, xmax = Inf, ymin = nc_lower, ymax = nc_upper,
           fill = "grey80", alpha = 0.6) +
  geom_hline(yintercept = 0, linetype = "dashed") +
  geom_boxplot(aes(fill = model), width = 0.5, alpha = 0.5, outlier.shape = NA, show.legend = FALSE) +
  geom_jitter(width = 0.1, size = 2, alpha = 0.7) +
  stat_summary(fun = mean, geom = "point", shape = 23, size = 4, fill = "white") +
  labs(title = "Model fit to neural RDMs",
       subtitle = "Grey band = noise ceiling; diamond = group mean; dots = subjects",
       x = NULL, y = "Spearman rho (neural RDM vs. model RDM)") +
  theme_minimal(base_size = 12)
print(p_fit)
ggsave("rsa_output/06_model_fit.png", p_fit, width = 6.5, height = 5, dpi = 300)

# ---- 6. Extra analyses ----------------------------------------------

# 6a. Within- vs between-category distances (intuition for what RSA "sees")
idx <- which(lower.tri(group_rdm), arr.ind = TRUE)
pair_info <- tibble(
  i = idx[, 1], j = idx[, 2],
  same_target  = cond_info$target[idx[, 1]]  == cond_info$target[idx[, 2]],
  same_valence = cond_info$valence[idx[, 1]] == cond_info$valence[idx[, 2]]
) %>%
  mutate(pair_type = case_when(
    same_target  & !same_valence ~ "Same target,\ndifferent valence",
    !same_target &  same_valence ~ "Different target,\nsame valence",
    TRUE                         ~ "Different target,\ndifferent valence"))

pair_dist <- map_dfr(subjects, function(s) {
  pair_info %>% mutate(subject = s, dist = rdms[[s]][cbind(i, j)])
}) %>%
  group_by(subject, pair_type) %>%
  summarise(dist = mean(dist), .groups = "drop")

p_pairs <- ggplot(pair_dist, aes(pair_type, dist)) +
  geom_line(aes(group = subject), colour = "grey70") +
  geom_point(aes(colour = pair_type), size = 2.5, show.legend = FALSE) +
  stat_summary(fun = mean, geom = "point", shape = 23, size = 4, fill = "white") +
  labs(title = "Mean neural distance by type of condition pair",
       subtitle = "Each line = one subject; diamond = group mean",
       x = NULL, y = "Mean dissimilarity (1 - r)") +
  theme_minimal(base_size = 12)
print(p_pairs)
ggsave("rsa_output/07_pair_types.png", p_pairs, width = 7, height = 5, dpi = 300)

# 6b. Cross-run reliability: correlate run-1 patterns with run-2 patterns.
#     If a condition's pattern is reliable, its diagonal r (same condition,
#     different run) should exceed its off-diagonal r (different condition).
cross_run <- lapply(subjects, function(s) {
  p1 <- get_patterns(filter(dat, subject == s, run == 1))
  p2 <- get_patterns(filter(dat, subject == s, run == 2))
  cor(t(p1), t(p2))                      # rows = run 1 conditions, cols = run 2 conditions
})
group_cross <- Reduce(`+`, cross_run) / length(cross_run)
p_cross <- plot_rdm(group_cross, "Cross-run pattern correlation (run 1 x run 2)",
                    legend_name = "Pearson r") +
  labs(x = "Run 2", y = "Run 1")
print(p_cross)
ggsave("rsa_output/08_cross_run.png", p_cross, width = 6.5, height = 5.5, dpi = 300)

# Diagonal (same condition) vs off-diagonal (different condition)
reliab <- map_dfr(seq_along(subjects), function(k) {
  m <- cross_run[[k]]
  tibble(subject = subjects[k],
         same_condition = mean(diag(m)),
         diff_condition = mean(m[row(m) != col(m)]))
})
cat("\nCross-run reliability (paired t-test, same vs different condition):\n")
print(t.test(reliab$same_condition, reliab$diff_condition, paired = TRUE))

# 6c. Optional: cross-validated RDM (uses different runs for each side of
#     the comparison, so noise does not inflate similarity)
cv_rdms <- lapply(cross_run, function(m) 1 - (m + t(m)) / 2)
cv_fit <- map_dfr(seq_along(subjects), function(k) {
  tibble(subject = subjects[k], model = names(models),
         rho = map_dbl(models, ~ cor(lt(cv_rdms[[k]]), lt(.x), method = "spearman")))
})
cat("\nMean model fit using cross-validated (cross-run) RDMs:\n")
print(cv_fit %>% group_by(model) %>% summarise(mean_rho = mean(rho), .groups = "drop"))

cat("\nDone. Plots were saved in:", file.path(getwd(), "rsa_output"), "\n")