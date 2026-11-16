# H1: Behavior-Based versus Memoryless Detection


## Verdict against the paper's criterion (Table IV)

**supported** -- difference = +0.145, 95% CI [+0.127, +0.163]


## Pooled results (mean across seeds [95% CI])

| method | session_detection | session_detection_lo | session_detection_hi | session_fpr | session_fpr_lo | session_fpr_hi | target_reached | target_reached_lo | target_reached_hi | session_precision | session_precision_lo | session_precision_hi | session_F1 | session_F1_lo | session_F1_hi | delay_median | delay_median_lo | delay_median_hi | delay_censored_mean | delay_censored_mean_lo | delay_censored_mean_hi | TPR | TPR_lo | TPR_hi | FPR | FPR_lo | FPR_hi | Precision | Precision_lo | Precision_hi | F1 | F1_lo | F1_hi | AUC | AUC_lo | AUC_hi | ASR | ASR_lo | ASR_hi | untrusted_share_attacks | untrusted_share_attacks_lo | untrusted_share_attacks_hi | n_seeds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| full | 0.359 | 0.329 | 0.393 | 0.070 | 0.043 | 0.094 | 0.000 | 0.000 | 0.000 | 0.841 | 0.796 | 0.898 | 0.501 | 0.471 | 0.531 | 0.000 | 0.000 | 0.000 | 1.772 | 1.682 | 1.862 | 0.321 | 0.287 | 0.361 | 0.012 | 0.009 | 0.014 | 0.796 | 0.770 | 0.834 | 0.456 | 0.420 | 0.493 | 0.910 | 0.898 | 0.923 | 0.679 | 0.639 | 0.713 | 0.435 | 0.394 | 0.472 | 5 |
| ref_only | 0.260 | 0.216 | 0.310 | 0.062 | 0.032 | 0.091 | 0.200 | 0.000 | 0.600 | 0.819 | 0.767 | 0.880 | 0.389 | 0.345 | 0.440 | 0.000 | 0.000 | 0.000 | 1.938 | 1.866 | 2.014 | 0.189 | 0.155 | 0.230 | 0.007 | 0.003 | 0.011 | 0.807 | 0.745 | 0.876 | 0.302 | 0.262 | 0.348 | 0.899 | 0.893 | 0.906 | 0.811 | 0.770 | 0.845 | 0.452 | 0.371 | 0.534 | 5 |
| memoryless | 0.213 | 0.171 | 0.272 | 0.105 | 0.038 | 0.208 | 0.200 | 0.000 | 0.600 | 0.727 | 0.611 | 0.823 | 0.318 | 0.282 | 0.361 | 0.000 | 0.000 | 0.000 | 2.066 | 1.995 | 2.129 | 0.152 | 0.122 | 0.193 | 0.014 | 0.004 | 0.030 | 0.698 | 0.542 | 0.823 | 0.239 | 0.212 | 0.271 | 0.846 | 0.835 | 0.861 | 0.848 | 0.807 | 0.878 | 0.305 | 0.201 | 0.424 | 5 |
| simplified | 0.383 | 0.344 | 0.427 | 0.069 | 0.057 | 0.079 | 0.200 | 0.000 | 0.600 | 0.847 | 0.822 | 0.869 | 0.526 | 0.488 | 0.569 | 0.000 | 0.000 | 0.000 | 1.754 | 1.696 | 1.850 | 0.316 | 0.286 | 0.351 | 0.010 | 0.008 | 0.011 | 0.819 | 0.797 | 0.838 | 0.455 | 0.421 | 0.492 | 0.916 | 0.906 | 0.925 | 0.684 | 0.649 | 0.714 | 0.435 | 0.399 | 0.468 | 5 |
| full(per-channel) | 0.379 | 0.302 | 0.443 | 0.082 | 0.064 | 0.107 | 0.200 | 0.000 | 0.600 | 0.818 | 0.771 | 0.863 | 0.513 | 0.429 | 0.583 | 0.000 | 0.000 | 0.000 | 1.773 | 1.649 | 1.912 | 0.332 | 0.254 | 0.390 | 0.013 | 0.010 | 0.016 | 0.777 | 0.745 | 0.802 | 0.460 | 0.372 | 0.524 | 0.907 | 0.893 | 0.921 | 0.668 | 0.610 | 0.746 | 0.429 | 0.380 | 0.493 | 5 |
| full+B5(ext) | 0.650 | 0.597 | 0.703 | 0.057 | 0.030 | 0.084 | 0.400 | 0.000 | 0.800 | 0.923 | 0.893 | 0.957 | 0.760 | 0.729 | 0.792 | 0.000 | 0.000 | 0.000 | 1.465 | 1.421 | 1.513 | 0.575 | 0.527 | 0.615 | 0.011 | 0.005 | 0.015 | 0.889 | 0.849 | 0.935 | 0.695 | 0.667 | 0.720 | 0.943 | 0.935 | 0.950 | 0.425 | 0.385 | 0.473 | 0.238 | 0.192 | 0.296 | 5 |
| full+B5(ext)-memoryless | 0.528 | 0.480 | 0.573 | 0.053 | 0.015 | 0.091 | 0.200 | 0.000 | 0.600 | 0.918 | 0.867 | 0.973 | 0.666 | 0.639 | 0.689 | 0.000 | 0.000 | 0.000 | 1.720 | 1.666 | 1.774 | 0.409 | 0.375 | 0.441 | 0.007 | 0.002 | 0.011 | 0.906 | 0.850 | 0.963 | 0.560 | 0.539 | 0.580 | 0.888 | 0.876 | 0.901 | 0.591 | 0.559 | 0.625 | 0.241 | 0.186 | 0.306 | 5 |
| full+B5(ext)-ref_only | 0.585 | 0.506 | 0.661 | 0.039 | 0.007 | 0.071 | 0.000 | 0.000 | 0.000 | 0.945 | 0.902 | 0.987 | 0.716 | 0.658 | 0.776 | 0.000 | 0.000 | 0.000 | 1.593 | 1.495 | 1.739 | 0.455 | 0.391 | 0.515 | 0.005 | 0.001 | 0.010 | 0.930 | 0.882 | 0.978 | 0.606 | 0.551 | 0.661 | 0.927 | 0.915 | 0.938 | 0.545 | 0.485 | 0.609 | 0.280 | 0.222 | 0.330 | 5 |
| B4-OursFT2 | 0.480 | 0.460 | 0.502 | 0.056 | 0.045 | 0.067 | 0.600 | 0.200 | 1.000 | 0.896 | 0.878 | 0.915 | 0.624 | 0.607 | 0.645 | 0.000 | 0.000 | 0.000 | 1.899 | 1.852 | 1.946 | 0.388 | 0.366 | 0.411 | 0.013 | 0.010 | 0.016 | 0.813 | 0.777 | 0.844 | 0.524 | 0.502 | 0.547 | 0.842 | 0.832 | 0.854 | 0.612 | 0.589 | 0.634 | 0.000 | 0.000 | 0.000 | 5 |


## Ranking at a common achieved false-positive rate

| method | session_AUC | session_AUC_lo | session_AUC_hi | det_at_fpr_0.05 | det_at_fpr_0.05_lo | det_at_fpr_0.05_hi | det_at_fpr_0.075 | det_at_fpr_0.075_lo | det_at_fpr_0.075_hi | det_at_fpr_0.1 | det_at_fpr_0.1_lo | det_at_fpr_0.1_hi | n_seeds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| full | 0.823 | 0.806 | 0.843 | 0.321 | 0.274 | 0.368 | 0.387 | 0.348 | 0.428 | 0.430 | 0.375 | 0.485 | 5 |
| ref_only | 0.777 | 0.764 | 0.791 | 0.222 | 0.213 | 0.232 | 0.262 | 0.239 | 0.290 | 0.279 | 0.245 | 0.312 | 5 |
| memoryless | 0.732 | 0.711 | 0.754 | 0.209 | 0.187 | 0.232 | 0.230 | 0.190 | 0.276 | 0.239 | 0.203 | 0.281 | 5 |
| simplified | 0.832 | 0.819 | 0.848 | 0.348 | 0.310 | 0.387 | 0.414 | 0.371 | 0.458 | 0.478 | 0.411 | 0.554 | 5 |
| full(per-channel) | 0.821 | 0.800 | 0.845 | 0.312 | 0.250 | 0.374 | 0.384 | 0.296 | 0.465 | 0.430 | 0.330 | 0.537 | 5 |
| full+B5(ext) | 0.907 | 0.894 | 0.922 | 0.671 | 0.636 | 0.712 | 0.710 | 0.661 | 0.758 | 0.740 | 0.694 | 0.786 | 5 |
| full+B5(ext)-memoryless | 0.869 | 0.865 | 0.873 | 0.519 | 0.488 | 0.555 | 0.522 | 0.490 | 0.560 | 0.571 | 0.521 | 0.631 | 5 |
| full+B5(ext)-ref_only | 0.899 | 0.878 | 0.924 | 0.617 | 0.558 | 0.700 | 0.650 | 0.584 | 0.729 | 0.673 | 0.600 | 0.762 | 5 |
| B4-OursFT2 | 0.854 | 0.847 | 0.858 | 0.532 | 0.504 | 0.554 | 0.570 | 0.541 | 0.598 | 0.603 | 0.581 | 0.624 | 5 |


## How the baselines are configured

B1 is a keyword/regex filter and has no threshold to set: a pattern match is a block, so its false-positive rate is whatever the patterns give and is usually above the target.
B2 scores the maximum of a verbatim banned-phrase match (1.0) and the embedding similarity to a banned phrase. Benign text does contain such phrases (a security article quoting an attack), so more than the allowed share of benign sessions sit at the top of its score scale and no threshold reaches the target; target_reached is then 0 and the detection rate is not a rate at the common operating point.
B3a (Prompt-Guard) is scored as its model card prescribes, which differs by channel: the jailbreak probability for messages written by the user, and the injection-or-jailbreak probability for third-party content. Scoring user messages with the stricter third-party rule flags almost every benign message and would understate the model.
B3a and B3c read at most 512 tokens at a time, so longer texts are split into overlapping segments and scored segment by segment (the model cards ask for this); a text counts as high as its worst segment.


## Differences (paired bootstrap)

| comparison | diff | lo | hi | p | excludes_zero | n |
|---|---|---|---|---|---|---|
| full - memoryless | 0.145 | 0.127 | 0.163 | 0.000 | True | 2100 |
| full - memoryless [direct_single] | 0.290 | 0.237 | 0.347 | 0.000 | True | 300 |
| full - memoryless [direct_spread] | 0.417 | 0.357 | 0.473 | 0.000 | True | 300 |
| full - memoryless [drift] | -0.013 | -0.050 | 0.023 | 0.525 | False | 300 |
| full - memoryless [single_chunk] | 0.087 | 0.066 | 0.108 | 0.000 | True | 900 |
| full - memoryless [split_chunks] | 0.063 | 0.023 | 0.107 | 0.002 | True | 300 |
| full - ref_only | 0.099 | 0.083 | 0.116 | 0.000 | True | 2100 |
| full - ref_only [direct_single] | 0.307 | 0.257 | 0.363 | 0.000 | True | 300 |
| full - ref_only [direct_spread] | 0.430 | 0.373 | 0.487 | 0.000 | True | 300 |
| full - ref_only [drift] | -0.033 | -0.060 | -0.007 | 0.017 | True | 300 |
| full - ref_only [single_chunk] | 0.014 | -0.003 | 0.032 | 0.114 | False | 900 |
| full - ref_only [split_chunks] | -0.053 | -0.093 | -0.013 | 0.008 | True | 300 |


## By attack kind

| method | se_attack_kind | session_detection | delay_median | n |
|---|---|---|---|---|
| B4-OursFT2 | direct_single | 0.907 | 0.000 | 300 |
| B4-OursFT2 | direct_spread | 0.913 | 0.000 | 300 |
| B4-OursFT2 | drift | 0.010 | 4.000 | 300 |
| B4-OursFT2 | single_chunk | 0.449 | 0.000 | 900 |
| B4-OursFT2 | split_chunks | 0.180 | 1.000 | 300 |
| full | direct_single | 0.573 | 0.000 | 300 |
| full | direct_spread | 0.630 | 0.000 | 300 |
| full | drift | 0.290 | 7.000 | 300 |
| full | single_chunk | 0.227 | 0.000 | 900 |
| full | split_chunks | 0.337 | 0.000 | 300 |
| full(per-channel) | direct_single | 0.543 | 0.000 | 300 |
| full(per-channel) | direct_spread | 0.587 | 0.000 | 300 |
| full(per-channel) | drift | 0.343 | 7.000 | 300 |
| full(per-channel) | single_chunk | 0.253 | 0.000 | 900 |
| full(per-channel) | split_chunks | 0.417 | 0.000 | 300 |
| full+B5(ext) | direct_single | 0.877 | 0.000 | 300 |
| full+B5(ext) | direct_spread | 0.890 | 0.000 | 300 |
| full+B5(ext) | drift | 0.380 | 7.000 | 300 |
| full+B5(ext) | single_chunk | 0.621 | 0.000 | 900 |
| full+B5(ext) | split_chunks | 0.540 | 0.000 | 300 |
| full+B5(ext)-memoryless | direct_single | 0.880 | 0.000 | 300 |
| full+B5(ext)-memoryless | direct_spread | 0.887 | 0.000 | 300 |
| full+B5(ext)-memoryless | drift | 0.343 | 7.000 | 300 |
| full+B5(ext)-memoryless | single_chunk | 0.434 | 0.000 | 900 |
| full+B5(ext)-memoryless | split_chunks | 0.283 | 0.000 | 300 |
| full+B5(ext)-ref_only | direct_single | 0.877 | 0.000 | 300 |
| full+B5(ext)-ref_only | direct_spread | 0.883 | 0.000 | 300 |
| full+B5(ext)-ref_only | drift | 0.330 | 7.000 | 300 |
| full+B5(ext)-ref_only | single_chunk | 0.524 | 0.000 | 900 |
| full+B5(ext)-ref_only | split_chunks | 0.430 | 0.000 | 300 |
| memoryless | direct_single | 0.283 | 0.000 | 300 |
| memoryless | direct_spread | 0.213 | 0.000 | 300 |
| memoryless | drift | 0.303 | 7.000 | 300 |
| memoryless | single_chunk | 0.140 | 0.000 | 900 |
| memoryless | split_chunks | 0.273 | 0.000 | 300 |
| ref_only | direct_single | 0.267 | 0.000 | 300 |
| ref_only | direct_spread | 0.200 | 0.000 | 300 |
| ref_only | drift | 0.323 | 7.000 | 300 |
| ref_only | single_chunk | 0.212 | 0.000 | 900 |
| ref_only | split_chunks | 0.390 | 0.000 | 300 |
| simplified | direct_single | 0.630 | 0.000 | 300 |
| simplified | direct_spread | 0.687 | 0.000 | 300 |
| simplified | drift | 0.293 | 7.000 | 300 |
| simplified | single_chunk | 0.240 | 0.000 | 900 |
| simplified | split_chunks | 0.350 | 0.000 | 300 |


## By attack source

| method | se_subset | session_detection | n |
|---|---|---|---|
| B4-OursFT2 | bipia | 0.423 | 300 |
| B4-OursFT2 | boiling_frog | 0.010 | 300 |
| B4-OursFT2 | direct_mt | 0.910 | 600 |
| B4-OursFT2 | kb | 0.098 | 600 |
| B4-OursFT2 | tasktracker | 0.907 | 300 |
| full | bipia | 0.123 | 300 |
| full | boiling_frog | 0.290 | 300 |
| full | direct_mt | 0.602 | 600 |
| full | kb | 0.365 | 600 |
| full | tasktracker | 0.163 | 300 |
| full(per-channel) | bipia | 0.180 | 300 |
| full(per-channel) | boiling_frog | 0.343 | 300 |
| full(per-channel) | direct_mt | 0.565 | 600 |
| full(per-channel) | kb | 0.407 | 600 |
| full(per-channel) | tasktracker | 0.183 | 300 |
| full+B5(ext) | bipia | 0.447 | 300 |
| full+B5(ext) | boiling_frog | 0.380 | 300 |
| full+B5(ext) | direct_mt | 0.883 | 600 |
| full+B5(ext) | kb | 0.527 | 600 |
| full+B5(ext) | tasktracker | 0.903 | 300 |
| full+B5(ext)-memoryless | bipia | 0.223 | 300 |
| full+B5(ext)-memoryless | boiling_frog | 0.343 | 300 |
| full+B5(ext)-memoryless | direct_mt | 0.883 | 600 |
| full+B5(ext)-memoryless | kb | 0.287 | 600 |
| full+B5(ext)-memoryless | tasktracker | 0.790 | 300 |
| full+B5(ext)-ref_only | bipia | 0.337 | 300 |
| full+B5(ext)-ref_only | boiling_frog | 0.330 | 300 |
| full+B5(ext)-ref_only | direct_mt | 0.880 | 600 |
| full+B5(ext)-ref_only | kb | 0.415 | 600 |
| full+B5(ext)-ref_only | tasktracker | 0.837 | 300 |
| memoryless | bipia | 0.067 | 300 |
| memoryless | boiling_frog | 0.303 | 300 |
| memoryless | direct_mt | 0.248 | 600 |
| memoryless | kb | 0.282 | 600 |
| memoryless | tasktracker | 0.063 | 300 |
| ref_only | bipia | 0.120 | 300 |
| ref_only | boiling_frog | 0.323 | 300 |
| ref_only | direct_mt | 0.233 | 600 |
| ref_only | kb | 0.388 | 600 |
| ref_only | tasktracker | 0.130 | 300 |
| simplified | bipia | 0.150 | 300 |
| simplified | boiling_frog | 0.293 | 300 |
| simplified | direct_mt | 0.658 | 600 |
| simplified | kb | 0.373 | 600 |
| simplified | tasktracker | 0.173 | 300 |


## Cost per configuration

| method | latency_mean_ms | latency_p95_ms | embedded_texts_per_event | state_bytes_per_session | python_peak_bytes_one_session |
|---|---|---|---|---|---|
| full | 34.826 | 52.793 | 1.512 | 3269.000 | 63713.800 |
| full(per-channel) | 33.628 | 64.555 | 1.512 | 3269.000 | 63780.000 |
| full+B5(ext) | 46.852 | 68.533 | 1.512 | 3269.000 | 75332.600 |
| full+B5(ext)-memoryless | 46.613 | 72.682 | 1.512 | 3269.000 | 75423.000 |
| full+B5(ext)-ref_only | 48.211 | 72.121 | 1.512 | 3269.000 | 75463.200 |
| memoryless | 32.975 | 63.932 | 1.512 | 3269.000 | 63900.400 |
| ref_only | 33.596 | 65.163 | 1.512 | 3269.000 | 63938.000 |
| simplified | 33.927 | 59.975 | 1.512 | 3269.000 | 63808.000 |
