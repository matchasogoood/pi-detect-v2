# Preliminary: Choice of the Decision Model (inputs=new)


## Choice under the one-standard-error rule

**logistic_regression** (highest TPR = logistic_regression 0.991, corrected SE = 0.002)


## Table V

| model | TPR | corr_SE | achieved_FPR | AUC | latency_us | n_distinct_scores | folds_target_reached | target_reached_on_mean | p_ttest | p_mcnemar | p_ttest_vs_rule | selected_by_1SE |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| rule | 0.986 | 0.005 | 0.203 | 0.990 | 0.592 | 226.000 | 0.000 | 0 | 0.143 | 0.018 | - | 0 |
| logistic_regression | 0.991 | 0.002 | 0.174 | 0.991 | 412.010 | 297.400 | 0.000 | 0 | - | - | - | 1 |
| decision_tree | 0.988 | 0.006 | 0.191 | 0.988 | 150.578 | 6.200 | 0.200 | 0 | 0.513 | 0.029 | - | 0 |
| random_forest | 0.991 | 0.002 | 0.209 | 0.991 | 20409.628 | 41.600 | 0.000 | 0 | 1.000 | 0.000 | - | 0 |
| gradient_boosting | 0.988 | 0.003 | 0.200 | 0.989 | 398.638 | 27.600 | 0.000 | 0 | 0.141 | 0.009 | - | 0 |
| lr_without_counts | 0.985 | 0.005 | 0.129 | 0.990 | - | 226.000 | 0.200 | 0 | - | - | 0.398 | 0 |


## Why some models show TPR 0.000

A threshold set for a 5% FPR can only be placed on a score with enough distinct values. n_distinct_scores is the mean number of distinct scores a model produces on a held-out fold (about 407 prompts) and folds_target_reached the share of folds whose achieved FPR landed within 25% of the target. Where that share is low the model's TPR is not a TPR at that FPR: a shallow tree puts almost every prompt in the same leaf, the threshold lands at or above its highest benign score, and nothing is flagged, which shows as TPR 0.000 with FPR 0.000. For those models the AUC, which needs no threshold, is the comparable number. target_reached_on_mean applies the same 25% test to the FPR averaged over all folds; folds_target_reached is lower for every model simply because one fold holds only about 407 prompts, so its FPR moves in steps of a few percent. Affected here: rule, logistic_regression, decision_tree, random_forest, gradient_boosting.


## LR weights

| feature | mean_standardized_coef |
|---|---|
| s | 2.719 |
| n_imp | 1.315 |
| n_ai | 0.700 |


## LaTeX (drop-in replacement for the rows of Table VI)

```
rule & 0.986 & 0.005 & 0.203 & 0.990 & 0.143 / 0.018 & 1 \\
logistic_regression & 0.991 & 0.002 & 0.174 & 0.991 & -- & 412 \\
decision_tree & 0.988 & 0.006 & 0.191 & 0.988 & 0.513 / 0.029 & 151 \\
random_forest & 0.991 & 0.002 & 0.209 & 0.991 & 1.000 / 0.000 & 20,410 \\
gradient_boosting & 0.988 & 0.003 & 0.200 & 0.989 & 0.141 / 0.009 & 399 \\
lr_without_counts & 0.985 & 0.005 & 0.129 & 0.990 & -- & -- \\
```
