# ============================================
# Metacritic: subset-normalized critic weights + critic bias
# + diagnostics: weight uncertainty vs review count
# 2026 | Author: Kimmo Kiiveri
# Refactored "production-style" script
# ============================================

suppressPackageStartupMessages({
  library(tidyverse)
  library(cmdstanr)
  library(posterior)
  library(loo)
  library(bayesplot)
  library(ggplot2)
})

# -------------------------------
# 0) Config
# -------------------------------
CONFIG <- list(
  year = 2015,
  base_dir = "/Users/kimmokiiveri/Desktop/metacritic_fetcher",
  out_dir = "/Users/kimmokiiveri/Desktop/metacritic_weight_analysis/"
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 1500,
  iter_sampling = 2000,
  adapt_delta = 0.99,
  max_treedepth = 12,
  init = 0,
  step_size = 0.1,
  refresh = 200,
  max_weight_draws = 2000,   # speed knob for weight CI computations
  topN = 30,
  topB = 30
)

paths <- list(
  input_file = file.path(CONFIG$base_dir, "data/processed",
                         paste0("metacritic_movies_", CONFIG$year, ".csv")),
  model_dir = file.path(CONFIG$base_dir, "models"),
  plot_dir  = file.path(CONFIG$out_dir, "plots")
)

dir.create(paths$model_dir, showWarnings = FALSE, recursive = TRUE)
dir.create(paths$plot_dir,  showWarnings = FALSE, recursive = TRUE)

# -------------------------------
# 1) Utilities (prod-style helpers)
# -------------------------------
assert_that <- function(cond, msg) {
  if (!isTRUE(cond)) stop(msg, call. = FALSE)
}

save_plot <- function(p, filename, width = 12, height = 8, dpi = 150) {
  ggsave(filename = filename, plot = p, width = width, height = height, dpi = dpi)
}

softmax_subset <- function(u_sub) {
  # u_sub: matrix (S x m)
  # stable row-wise softmax
  u_max <- apply(u_sub, 1, max)
  a <- exp(u_sub - u_max)
  a / rowSums(a)
}

# -------------------------------
# 2) Load + clean data
# -------------------------------
df <- read.csv(paths$input_file) %>%
  as_tibble() %>%
  select(movie_title, critic_author, critic_score, metascore) %>%
  filter(
    !is.na(movie_title),
    !is.na(critic_author),
    !is.na(critic_score),
    !is.na(metascore)
  ) %>%
  mutate(
    movie_title   = droplevels(as.factor(movie_title)),
    critic_author = droplevels(as.factor(critic_author)),
    critic_score  = as.numeric(critic_score),
    metascore     = as.numeric(metascore)
  )

cat("Loaded rows:", nrow(df), "\n")
cat("Movies:", nlevels(df$movie_title), " Critics:", nlevels(df$critic_author), "\n")

assert_that(nrow(df) > 0, "No data rows after filtering NAs.")
assert_that(all(is.finite(df$critic_score)), "Non-finite critic_score values found.")
assert_that(all(is.finite(df$metascore)), "Non-finite metascore values found.")

# metascore constant per movie? if not, warn and average
meta_check <- df %>%
  group_by(movie_title) %>%
  summarise(n_meta = n_distinct(metascore), .groups = "drop") %>%
  summarise(max_n_meta = max(n_meta), .groups = "drop") %>%
  pull(max_n_meta)

if (meta_check > 1) {
  warning("metascore is not constant within some movies; using per-movie mean metascore.")
}

# Per-movie metascore in the exact factor level order
df_movie <- df %>%
  mutate(movie_title = factor(movie_title, levels = levels(df$movie_title))) %>%
  group_by(movie_title) %>%
  summarise(metascore = mean(metascore), .groups = "drop") %>%
  arrange(as.integer(movie_title))

# Flattened review rows (sorted by movie, so bounds are contiguous)
df_flat <- df %>%
  arrange(movie_title) %>%
  mutate(
    row_id    = row_number(),
    movie_id  = as.integer(movie_title),
    critic_id = as.integer(critic_author)
  )

J <- nlevels(df_flat$movie_title)
K <- nlevels(df_flat$critic_author)
N <- nrow(df_flat)

assert_that(nrow(df_movie) == J, "df_movie rows != J (movie level mismatch).")

# Movie row bounds
movie_bounds <- df_flat %>%
  group_by(movie_id) %>%
  summarise(start = min(row_id), end = max(row_id), .groups = "drop") %>%
  arrange(movie_id)

assert_that(nrow(movie_bounds) == J, "movie_bounds rows != J.")

stan_data <- list(
  J = J,
  K = K,
  N = N,
  start  = movie_bounds$start,
  end    = movie_bounds$end,
  critic = df_flat$critic_id,
  score  = as.vector(df_flat$critic_score),
  meta   = as.vector(df_movie$metascore)
)

# -------------------------------
# 3) Critical sanity checks (avoid NaN/Inf)
# -------------------------------
counts <- df_flat %>% count(movie_id, name = "n_reviews")
assert_that(nrow(counts) == J, "Not all movies appear in df_flat.")
assert_that(all(counts$n_reviews >= 1), "Some movie has zero reviews (unexpected).")

assert_that(!any(is.na(stan_data$start)), "NA in start bounds.")
assert_that(!any(is.na(stan_data$end)), "NA in end bounds.")
assert_that(all(stan_data$start >= 1), "Some start < 1.")
assert_that(all(stan_data$end <= N), "Some end > N.")
assert_that(all(stan_data$start <= stan_data$end), "Some start > end.")
assert_that(min(stan_data$start) == 1, "Bounds do not start at 1 (data not contiguous by movie).")
assert_that(max(stan_data$end) == N, "Bounds do not end at N (data not contiguous by movie).")

assert_that(all(is.finite(stan_data$score)), "Non-finite score in stan_data.")
assert_that(all(is.finite(stan_data$meta)), "Non-finite meta in stan_data.")
assert_that(all(stan_data$score >= 0 & stan_data$score <= 100), "score out of [0,100].")
assert_that(all(stan_data$meta  >= 0 & stan_data$meta  <= 100), "meta out of [0,100].")

cat("Sanity checks passed.\n")

# -------------------------------
# 4) Stan model
# -------------------------------
stan_code <- "
data {
  int<lower=1> J;
  int<lower=1> K;
  int<lower=1> N;
  array[J] int<lower=1> start;
  array[J] int<lower=1> end;
  array[N] int<lower=1, upper=K> critic;
  vector[N] score;
  vector[J] meta;
}
parameters {
  vector[K] u_raw;
  vector[K] b_raw;

  real<lower=1e-6> tau_u;
  real<lower=1e-6> tau_b;

  real<lower=1e-3> sigma;
  real<lower=2> nu;
}
transformed parameters {
  vector[K] u;
  vector[K] b;
  u = u_raw - mean(u_raw);
  b = b_raw - mean(b_raw);
}
model {
  tau_u ~ exponential(1);
  tau_b ~ exponential(0.2);
  sigma ~ exponential(0.1);
  nu ~ gamma(2, 0.1);

  u_raw ~ normal(0, tau_u);
  b_raw ~ normal(0, tau_b);

  for (j in 1:J) {
    real denom = 0;
    real numer = 0;
    for (n in start[j]:end[j]) {
      int k = critic[n];
      real a = exp(u[k]);
      denom += a;
      numer += a * (score[n] + b[k]);
    }
    if (!(denom > 0)) {
      reject(\"denom <= 0 for movie j=\", j);
    }
    meta[j] ~ student_t(nu, numer / denom, sigma);
  }
}
generated quantities {
  vector[J] log_lik;
  vector[J] mu_hat;

  for (j in 1:J) {
    real denom = 0;
    real numer = 0;
    for (n in start[j]:end[j]) {
      int k = critic[n];
      real a = exp(u[k]);
      denom += a;
      numer += a * (score[n] + b[k]);
    }
    mu_hat[j] = numer / denom;
    log_lik[j] = student_t_lpdf(meta[j] | nu, mu_hat[j], sigma);
  }
}
"

stan_file <- file.path(paths$model_dir, paste0("metacritic_softmax_weights_bias_", CONFIG$year, ".stan"))
writeLines(stan_code, con = stan_file)
mod <- cmdstan_model(stan_file)

# -------------------------------
# 5) Fit
# -------------------------------
fit <- mod$sample(
  data = stan_data,
  chains = CONFIG$chains,
  parallel_chains = CONFIG$parallel_chains,
  iter_warmup = CONFIG$iter_warmup,
  iter_sampling = CONFIG$iter_sampling,
  seed = 123,
  adapt_delta = CONFIG$adapt_delta,
  max_treedepth = CONFIG$max_treedepth,
  init = CONFIG$init,
  step_size = CONFIG$step_size,
  refresh = CONFIG$refresh
)

fit_rds <- file.path(paths$model_dir, paste0("metascore_softmax_bias_fit_", CONFIG$year, ".rds"))
saveRDS(fit, fit_rds)
cat("Saved fit to:", fit_rds, "\n")

# -------------------------------
# 6) Diagnostics + alerts
# -------------------------------
summ <- fit$summary()
rhat_max <- max(summ$rhat, na.rm = TRUE)
ess_min  <- min(summ$ess_bulk, na.rm = TRUE)

cat("Rhat max:", rhat_max, "\n")
cat("ESS bulk min:", ess_min, "\n")

if (is.finite(rhat_max) && rhat_max > 1.01) {
  warning("Rhat > 1.01 detected → possible non-convergence. Consider more iterations / stronger priors / reparameterization.")
}
if (is.finite(ess_min) && ess_min < 200) {
  warning("Low ESS detected (<200) → posterior may be poorly explored. Consider more iterations or model regularization.")
}

fit$cmdstan_diagnose()

log_lik_mat <- posterior::as_draws_matrix(fit$draws("log_lik"))
loo_res <- loo::loo(log_lik_mat)
print(loo_res)

# -------------------------------
# 7) Maps + core posterior matrices
# -------------------------------
critic_map <- tibble(
  critic_id = 1:K,
  critic_author = levels(df$critic_author)
)

u_mat <- posterior::as_draws_matrix(fit$draws("u"))  # draws x K
b_mat <- posterior::as_draws_matrix(fit$draws("b"))  # draws x K

# Save critic means table (alpha mean and bias mean)
critic_table <- tibble(
  critic_id  = 1:K,
  alpha_mean = colMeans(exp(u_mat)),
  bias_mean  = colMeans(b_mat)
) %>%
  left_join(critic_map, by = "critic_id") %>%
  select(critic_author, alpha_mean, bias_mean) %>%
  arrange(desc(alpha_mean))

write.csv(
  critic_table,
  file.path(paths$plot_dir, paste0("critic_alpha_bias_", CONFIG$year, ".csv")),
  row.names = FALSE
)

# -------------------------------
# 8) Plots: trace, bias, PPC, residuals
# -------------------------------
core_draws <- posterior::as_draws_df(fit$draws(c("tau_u", "tau_b", "sigma", "nu")))
p_trace <- bayesplot::mcmc_trace(core_draws) +
  ggtitle("Traceplots: tau_u, tau_b, sigma, nu")
save_plot(p_trace, file.path(paths$plot_dir, paste0("trace_core_params_", CONFIG$year, ".png")),
          width = 12, height = 7)

# Bias hist + top |bias|
b_mean <- colMeans(b_mat)
p_bias_hist <- ggplot(tibble(b_mean = b_mean), aes(x = b_mean)) +
  geom_histogram(bins = 50) +
  labs(title = "Distribution of critic bias means (b)",
       x = "bias mean (points, 0–100 scale)", y = "count") +
  theme_minimal()
save_plot(p_bias_hist, file.path(paths$plot_dir, paste0("bias_hist_", CONFIG$year, ".png")),
          width = 10, height = 6)

topB <- CONFIG$topB
topb_idx <- order(abs(b_mean), decreasing = TRUE)[seq_len(min(topB, length(b_mean)))]
b_top <- b_mat[, topb_idx, drop = FALSE]
b_q <- t(apply(b_top, 2, quantile, probs = c(0.1, 0.5, 0.9)))
b_df <- tibble(
  critic_id = topb_idx,
  b_p10 = b_q[, 1],
  b_p50 = b_q[, 2],
  b_p90 = b_q[, 3]
) %>%
  left_join(critic_map, by = "critic_id") %>%
  mutate(critic_author = factor(critic_author, levels = critic_author[order(b_p50)]))

p_bias_top <- ggplot(b_df, aes(x = b_p50, y = critic_author)) +
  geom_errorbarh(aes(xmin = b_p10, xmax = b_p90), height = 0.2) +
  geom_point(size = 2) +
  geom_vline(xintercept = 0, linetype = "dashed") +
  labs(title = paste0("Top ", topB, " critics by |bias| (b)"),
       x = "bias (posterior median and 10–90% interval)", y = NULL) +
  theme_minimal()
save_plot(p_bias_top, file.path(paths$plot_dir, paste0("bias_top_", topB, "_", CONFIG$year, ".png")),
          width = 12, height = 9)

# PPC + residuals
mu_hat_mat <- posterior::as_draws_matrix(fit$draws("mu_hat"))
assert_that(ncol(mu_hat_mat) == J, "mu_hat columns != J (data indexing mismatch).")

pp_df <- tibble(
  metascore = stan_data$meta,
  mu_mean = colMeans(mu_hat_mat)
) %>%
  mutate(resid = metascore - mu_mean)

p_pp <- ggplot(pp_df, aes(x = mu_mean, y = metascore)) +
  geom_point(alpha = 0.4, size = 1.5) +
  geom_smooth(method = "loess", se = FALSE) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
  labs(title = "Posterior predictive check",
       x = "Posterior mean prediction (mu_hat)",
       y = "Observed metascore") +
  theme_minimal()
save_plot(p_pp, file.path(paths$plot_dir, paste0("ppcheck_scatter_", CONFIG$year, ".png")),
          width = 12, height = 7)

r2 <- cor(pp_df$mu_mean, pp_df$metascore)^2
cat("R^2 (corr^2):", r2, "\n")

p_resid <- ggplot(pp_df, aes(x = mu_mean, y = resid)) +
  geom_point(alpha = 0.4) +
  geom_hline(yintercept = 0, linetype = "dashed") +
  labs(title = "Residuals vs mu_hat",
       x = "mu_hat",
       y = "residual (obs - pred)") +
  theme_minimal()
save_plot(p_resid, file.path(paths$plot_dir, paste0("residuals_", CONFIG$year, ".png")),
          width = 12, height = 7)

# -------------------------------
# 9) Critic weights with uncertainty (mean normalized weight per movie)
# -------------------------------
# Subsample draws for speed (optional)
set.seed(123)
u_mat_all <- posterior::as_draws_matrix(fit$draws("u"))
if (nrow(u_mat_all) > CONFIG$max_weight_draws) {
  keep <- sample.int(nrow(u_mat_all), CONFIG$max_weight_draws)
  u_mat_w <- u_mat_all[keep, , drop = FALSE]
} else {
  u_mat_w <- u_mat_all
}

S <- nrow(u_mat_w)

# Precompute which movies each critic appears in
critic_in_movie <- vector("list", K)
for (j in seq_len(J)) {
  idx <- stan_data$start[j]:stan_data$end[j]
  crits <- unique(stan_data$critic[idx])
  for (k in crits) critic_in_movie[[k]] <- c(critic_in_movie[[k]], j)
}
n_movies_present <- vapply(critic_in_movie, length, integer(1))

sum_w <- matrix(0, nrow = S, ncol = K)

for (j in seq_len(J)) {
  idx <- stan_data$start[j]:stan_data$end[j]
  crits <- unique(stan_data$critic[idx])
  u_sub <- u_mat_w[, crits, drop = FALSE]
  w_sub <- softmax_subset(u_sub)   # S x m
  
  for (t in seq_along(crits)) {
    k <- crits[t]
    sum_w[, k] <- sum_w[, k] + w_sub[, t]
  }
}

mean_w <- sum_w
for (k in seq_len(K)) {
  if (n_movies_present[k] > 0) {
    mean_w[, k] <- mean_w[, k] / n_movies_present[k]
  } else {
    mean_w[, k] <- NA_real_
  }
}

w_q <- apply(mean_w, 2, quantile, probs = c(0.1, 0.5, 0.9), na.rm = TRUE)
w_df <- tibble(
  critic_id = seq_len(K),
  w_p10 = w_q[1, ],
  w_p50 = w_q[2, ],
  w_p90 = w_q[3, ],
  n_movies = n_movies_present
) %>%
  left_join(critic_map, by = "critic_id") %>%
  filter(!is.na(w_p50)) %>%
  arrange(desc(w_p50))

write.csv(
  w_df,
  file.path(paths$plot_dir, paste0("critic_mean_weights_", CONFIG$year, ".csv")),
  row.names = FALSE
)

N_eff <- min(CONFIG$topN, nrow(w_df))
w_top <- w_df %>%
  slice_head(n = N_eff) %>%
  mutate(critic_author = factor(critic_author, levels = rev(critic_author)))

p_w <- ggplot(w_top, aes(x = w_p50, y = critic_author)) +
  geom_errorbarh(aes(xmin = w_p10, xmax = w_p90), height = 0.2) +
  geom_point(size = 2) +
  labs(
    title = paste0("Top ", CONFIG$topN, " critics by mean normalized weight (w̄)"),
    subtitle = "w̄_k = average of movie-level normalized weights across movies where critic appears (10–90% interval)",
    x = "mean normalized weight per movie (posterior median and 10–90% interval)",
    y = NULL
  ) +
  theme_minimal()

save_plot(p_w, file.path(paths$plot_dir, paste0("weights_top_", CONFIG$topN, "_", CONFIG$year, ".png")),
          width = 12, height = 9)

# -------------------------------
# 10) NEW: Weight uncertainty vs review count
# -------------------------------
review_counts <- df %>%
  count(critic_author, name = "n_reviews")

weight_uncertainty <- w_df %>%
  mutate(ci_width = w_p90 - w_p10) %>%
  left_join(review_counts, by = "critic_author")

assert_that(!any(is.na(weight_uncertainty$n_reviews)),
            "Missing n_reviews after join. Check critic_author mapping consistency.")

p_uncert <- ggplot(weight_uncertainty, aes(x = n_reviews, y = ci_width)) +
  geom_point(alpha = 0.6) +
  geom_smooth(method = "lm", se = FALSE, linetype = "dashed", color = "black") +
  scale_x_log10() +
  labs(
    title = "Posterior weight uncertainty vs number of reviews",
    subtitle = "CI width = w_90% − w_10%, x-axis is log scale",
    x = "Number of reviews (log scale)",
    y = "Weight uncertainty (10–90% interval width)"
  ) +
  theme_minimal()

save_plot(p_uncert, file.path(paths$plot_dir, paste0("weight_uncertainty_vs_reviews_", CONFIG$year, ".png")),
          width = 12, height = 7)

corr_val <- cor(log(weight_uncertainty$n_reviews), weight_uncertainty$ci_width, use = "complete.obs")
cat("Correlation: log(n_reviews) vs CI width =", corr_val, "\n")

cat("Done.\n")
