# Bayesian Estimation of Critic Influence in Metacritic Scores

## Overview

This project implements a **Bayesian hierarchical model** to estimate **latent critic influence weights** and **systematic critic bias** behind Metacritic movie scores.

Metacritic does not publicly disclose how individual critic reviews are aggregated into a final metascore.  
The goal of this project is **not to reverse-engineer Metacritic exactly**, but to infer **plausible relative influence weights** of critics based solely on observed critic scores and final metascores.

The core question is:

> *Given a set of critic reviews for each movie and the resulting metascore, what can we infer about how much each critic tends to influence the final score?*

---

## Modeling Philosophy

The model assumes that:

- Each critic has a **global influence parameter** that applies consistently across all movies they review.
- Only critics who reviewed a given movie contribute to its metascore.
- Critic influence is **relative within each movie**, i.e. weights are normalized over the subset of critics present for that movie.
- Critics may exhibit **systematic bias** (tendency to score higher or lower than average).

This results in a **subset-normalized weighted average model** with explicit uncertainty quantification.

---

## Mathematical Model

For movie \( j \) with critics \( k \in \mathcal{C}_j \):

### Latent critic parameters

- Log-weight:  
  \[
  u_k \in \mathbb{R}
  \]
- Bias (in score units):  
  \[
  b_k \in \mathbb{R}
  \]

To ensure identifiability:
- \( u_k \) are centered: \( u_k - \bar{u} \)
- \( b_k \) are centered: \( b_k - \bar{b} \)

### Movie-level normalized weights

For a given movie:
\[
w_{jk} = \frac{\exp(u_k)}{\sum_{k' \in \mathcal{C}_j} \exp(u_{k'})}
\]

This guarantees:
- Positivity
- Automatic rescaling when critics are missing
- Weights sum to 1 **within each movie**

### Expected metascore

\[
\mu_j = \sum_{k \in \mathcal{C}_j} w_{jk} \cdot (s_{jk} + b_k)
\]

where:
- \( s_{jk} \) is the observed critic score

### Likelihood

Observed metascore:
\[
\text{metascore}_j \sim \text{Student-t}(\nu, \mu_j, \sigma)
\]

A Student-t likelihood is used for robustness against outliers and aggregation noise.

---

## Prior Choices and Rationale

All priors are **weakly informative and regularizing**, chosen to improve identifiability and sampling stability:

| Parameter | Prior | Motivation |
|---------|------|------------|
| \( u_k \) | \( \mathcal{N}(0, \tau_u) \) | Allows flexible relative influence |
| \( b_k \) | \( \mathcal{N}(0, \tau_b) \) | Captures critic bias in score units |
| \( \tau_u \) | Exponential(1) | Regularizes influence variability |
| \( \tau_b \) | Exponential(0.2) | Bias typically within a few points |
| \( \sigma \) | Exponential(0.1) | Metascore noise (~10 points mean) |
| \( \nu \) | Gamma(2, 0.1) | Moderately heavy tails |

Strict lower bounds are applied to scale parameters to avoid pathological zero-scale behavior in Stan.

---

## Inference and Computation

- Implemented in **Stan** via **cmdstanr**
- Sampler: **NUTS (No-U-Turn Sampler)** with adaptive step size
- Multiple chains with convergence diagnostics:
  - R-hat
  - Effective Sample Size
  - E-BFMI
  - Divergence checks
- Defensive runtime checks prevent silent NaN / Inf propagation

Posterior summaries include:
- Critic influence weights with uncertainty
- Critic bias distributions
- Posterior predictive checks
- Residual diagnostics
- Influence uncertainty vs. number of reviews

---

## Key Outputs

- **Relative critic influence** (mean normalized weight per movie)
- **Critic bias estimates** (positive or negative scoring tendencies)
- **Uncertainty diagnostics** (credible intervals, review count effects)
- **Posterior predictive validation plots**

These results allow ranking critics by:
- Expected influence
- Bias magnitude
- Certainty of estimation

---

## What This Project Is *Not*

This project is **explicitly not** intended to:

- Reproduce Metacritic’s true proprietary algorithm
- Predict future metascores accurately
- Perform recommendation or sentiment analysis
- Serve as a production scoring system
- Claim causal influence of critics on audience perception

The model is **inferential**, not predictive, and should be interpreted accordingly.

---

## Reproducibility Notes

- Results depend on available critic coverage and review frequency
- Critics with few reviews naturally exhibit higher uncertainty
- All assumptions are explicit and inspectable in the Stan model

---

## License

This repository is intended for **educational and analytical purposes**.  
No affiliation with or endorsement by Metacritic.

