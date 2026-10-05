# JPMorgan Quantitative Research Job Simulation

Independent Python implementation of the **JPMorgan Chase & Co. Quantitative Research Job Simulation completed through Forage**.

The simulation was completed by **Vincent Haïk Karakoseian** in June 2025 and covered four practical quantitative tasks:

1. natural-gas price analysis and forecasting;
2. commodity storage contract valuation;
3. credit-risk modeling and expected-loss estimation;
4. FICO-score bucketing / quantization.

[View the completion certificate](docs/completion_certificate.pdf)

> This repository documents work completed in a **Forage virtual job simulation**. It does not represent employment, an internship, or professional experience at JPMorgan Chase & Co.

## Repository Structure

```text
JPMorgan-Quantitative-Research-Job-Simulation/
├── data/
│   ├── Nat_Gas.csv
│   ├── Loan_Data.csv
│   └── README.md
├── docs/
│   ├── completion_certificate.pdf
│   └── methodology.md
├── src/
│   ├── __init__.py
│   ├── task1_gas_forecasting.py
│   ├── task2_storage_contract.py
│   ├── task3_credit_risk.py
│   └── task4_fico_bucketing.py
├── .gitignore
├── README.md
└── requirements.txt
```

Generated plots and CSV outputs are written to `outputs/`, which is excluded from version control.

## Task 1 - Natural Gas Price Forecasting

The first task analyzes monthly natural-gas prices and builds a model capable of extrapolating future prices.

The initial analysis identifies:

- a long-term upward trend;
- recurring monthly seasonality.

A simple trend-only regression is insufficient because it ignores the seasonal pattern. The cleaned implementation therefore combines:

- a linear time trend;
- monthly dummy variables.

Conceptually:

`Predicted Price = Intercept + Trend Effect + Monthly Seasonal Effect`

The script produces a 12-month month-end forecast and a historical-versus-forecast plot.

Run:

```bash
python -m src.task1_gas_forecasting
```

## Task 2 - Commodity Storage Contract Pricing

The second task uses the natural-gas forecasting model to value a storage contract.

The contract valuation accounts for:

- gas purchased on injection dates;
- gas sold on withdrawal dates;
- injection costs;
- withdrawal costs;
- transportation costs;
- monthly storage costs;
- maximum storage capacity.

The cleaned implementation also validates the full inventory path so that stored volume can never become negative or exceed capacity.

The main valuation identity is:

`Contract Value = Sale Revenue - Purchase Cost - Logistics Costs`

The original simulation assumptions are retained:

- zero interest rates;
- no transport delay;
- no explicit weekend or holiday adjustment.

Run:

```bash
python -m src.task2_storage_contract
```

## Task 3 - Credit Risk and Expected Loss

The third task estimates borrower default risk from six explanatory variables:

- credit lines outstanding;
- loan amount outstanding;
- total debt outstanding;
- income;
- years employed;
- FICO score.

The repository benchmarks:

- Logistic Regression;
- Decision Tree;
- Random Forest;
- XGBoost;
- Support Vector Machine;
- Multi-Layer Perceptron.

Models are compared using:

- accuracy;
- precision on the default class;
- recall on the default class;
- specificity;
- ROC AUC.

For each borrower, the model can produce a default probability.

Expected loss is then calculated as:

`Expected Loss = Probability of Default x Exposure at Default x Loss Given Default`

with:

`Loss Given Default = 1 - Recovery Rate`

Run:

```bash
python -m src.task3_credit_risk
```

## Task 4 - FICO Score Bucketing

The fourth task transforms continuous FICO scores into discrete credit-risk buckets.

Two methodologies are implemented.

### K-Means Quantization

K-means groups FICO scores by minimizing within-cluster squared error.

The repository includes:

- an elbow-curve analysis;
- centroid estimation;
- automatic bucket-threshold construction;
- bucket-level default-rate summaries.

### Maximum Log-Likelihood Dynamic Programming

A second method searches for contiguous FICO buckets that maximize Bernoulli log-likelihood.

The dynamic-programming state is:

`DP[i, k] = best log-likelihood obtained by splitting the first i ordered score levels into k buckets`

The recurrence is:

`DP[i, k] = max over j of (DP[j, k-1] + LogLikelihood(j+1, i))`

Backtracking recovers the optimal score thresholds.

The cleaned implementation aggregates borrowers by unique FICO score before optimization. This preserves the exact borrower counts and default counts while making the dynamic-programming problem computationally practical.

Run:

```bash
python -m src.task4_fico_bucketing
```

## Installation

Create a virtual environment if desired, then install:

```bash
pip install -r requirements.txt
```

## Python Stack

- Python
- NumPy
- pandas
- Matplotlib
- scikit-learn
- XGBoost

## Improvements Made for the Repository Version

The original task scripts were exploratory learning files. The repository version has been cleaned for reproducibility and review.

Changes include:

- replacing machine-specific absolute Windows paths with relative project paths;
- fixing the invalid Windows path that prevented the original Task 1 script from parsing;
- fixing the Task 3 decision-tree prediction call so it uses the fitted decision-tree model;
- using a consistent 80/20 train/test split with stratification;
- standardizing features for scale-sensitive classifiers;
- computing evaluation metrics directly rather than relying on manually written values;
- returning a numerical storage-contract valuation instead of only printing intermediate calculations;
- validating the storage inventory path;
- replacing the memory-heavy borrower-level dynamic-programming table with an exact unique-FICO-level implementation;
- standardizing documentation and comments in English.

## Technical Notes

A more detailed explanation of the modeling choices is available in:

[Methodology Notes](docs/methodology.md)

## Data

The repository currently includes the two datasets used by the scripts so that the project is reproducible.

If you publish this repository publicly, ensure that redistribution of the simulation datasets is consistent with the terms under which you received them.

## Author

**Vincent Haïk Karakoseian**

## Certificate

The Forage certificate confirms completion of the four practical tasks represented in this repository:

- investigate and analyze price data;
- price a commodity storage contract;
- credit risk analysis;
- bucket FICO scores.

[Open the certificate](docs/completion_certificate.pdf)

## Disclaimer

This repository is provided for educational and portfolio purposes only.

It is not investment advice, credit advice, or a representation of employment by JPMorgan Chase & Co.
