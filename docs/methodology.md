# Methodology Notes

This document gives a compact technical overview of the four tasks implemented in this repository.

## Task 1 - Natural gas price forecasting

The raw dataset contains monthly natural-gas prices from October 2020 to September 2024.

The observed structure contains two effects:

1. a positive long-run trend;
2. recurring monthly seasonality, with systematically different price levels across the year.

A first linear model based only on time cannot capture the seasonal structure. The final specification therefore combines a linear time trend with month indicator variables.

Conceptually:

`Predicted Price = Intercept + Trend Coefficient x Time + Monthly Seasonal Effects`

January is used as the reference month, while February through December are represented by dummy variables.

The model is then used to extrapolate the next 12 month-end prices.

## Task 2 - Natural gas storage contract

The storage contract is valued from projected natural-gas prices and logistics costs.

For each injection date, gas is purchased and placed into storage.

For each withdrawal date, stored gas is released and sold.

The contract value is computed as:

`Contract Value = Sale Revenue - Purchase Cost - Injection Cost - Withdrawal Cost - Transportation Cost - Storage Cost`

The cleaned implementation also verifies that the inventory path:

- never becomes negative;
- never exceeds maximum storage capacity.

The simulation assumptions are kept simple: zero interest rates, no transport delay, and no explicit weekend / holiday handling.

## Task 3 - Credit default modeling

The credit-risk dataset contains borrower information including:

- credit lines outstanding;
- current loan amount;
- total debt;
- income;
- years employed;
- FICO score;
- default indicator.

The repository benchmarks six classifiers:

- Logistic Regression;
- Decision Tree;
- Random Forest;
- XGBoost;
- Support Vector Machine;
- Multi-Layer Perceptron.

Scale-sensitive models are standardized before training.

The main evaluation metrics are:

- accuracy;
- precision on defaults;
- recall on defaults;
- specificity;
- ROC AUC.

For a borrower with estimated default probability `PD`, exposure at default `EAD`, and recovery rate `RR`:

`LGD = 1 - RR`

and:

`Expected Loss = PD x EAD x LGD`

## Task 4 - FICO score bucketing

The goal is to transform continuous FICO scores into a small number of discrete rating buckets.

Two approaches are implemented.

### K-means

K-means minimizes within-cluster squared error.

In one dimension, each FICO score is assigned to the nearest cluster centroid. Bucket thresholds are placed midway between consecutive sorted centroids.

### Maximum log-likelihood dynamic programming

The second method chooses contiguous FICO buckets to maximize Bernoulli log-likelihood.

For one bucket containing `n` borrowers and `k` defaults:

`p = k / n`

and the bucket log-likelihood is:

`LL = k x log(p) + (n-k) x log(1-p)`

For dynamic programming, define:

`DP[i, k] = best log-likelihood obtained by splitting the first i ordered score levels into k buckets`

The recurrence is:

`DP[i, k] = max_j ( DP[j, k-1] + LL(j+1, i) )`

Backtracking through the maximizing split points recovers the optimal FICO thresholds.

The cleaned implementation aggregates borrowers by unique FICO score before dynamic programming. This preserves exact counts while making the optimization practical on the full dataset.
