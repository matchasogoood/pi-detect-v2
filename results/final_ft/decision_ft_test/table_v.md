### Table VI (rerun, inputs=new, bank rebuilt per fold)

| model | TPR | corr_SE | achieved_FPR | AUC | latency_us | n_distinct_scores | folds_target_reached | target_reached_on_mean | p_ttest | p_mcnemar | p_ttest_vs_rule | selected_by_1SE |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| rule | 0.986 | 0.005 | 0.203 | 0.990 | 0.592 | 226.000 | 0.000 | 0 | 0.143 | 0.018 | - | 0 |
| logistic_regression | 0.991 | 0.002 | 0.174 | 0.991 | 412.010 | 297.400 | 0.000 | 0 | - | - | - | 1 |
| decision_tree | 0.988 | 0.006 | 0.191 | 0.988 | 150.578 | 6.200 | 0.200 | 0 | 0.513 | 0.029 | - | 0 |
| random_forest | 0.991 | 0.002 | 0.209 | 0.991 | 20409.628 | 41.600 | 0.000 | 0 | 1.000 | 0.000 | - | 0 |
| gradient_boosting | 0.988 | 0.003 | 0.200 | 0.989 | 398.638 | 27.600 | 0.000 | 0 | 0.141 | 0.009 | - | 0 |
| lr_without_counts | 0.985 | 0.005 | 0.129 | 0.990 | - | 226.000 | 0.200 | 0 | - | - | 0.398 | 0 |
