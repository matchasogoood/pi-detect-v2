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
| full+B3b(ind_win_logit_only) | 0.501 | 0.386 | 0.627 | 0.080 | 0.041 | 0.115 | 0.200 | 0.000 | 0.600 | 0.866 | 0.810 | 0.922 | 0.624 | 0.534 | 0.727 | 0.000 | 0.000 | 0.000 | 1.799 | 1.596 | 2.018 | 0.412 | 0.319 | 0.509 | 0.012 | 0.008 | 0.018 | 0.827 | 0.773 | 0.881 | 0.542 | 0.457 | 0.630 | 0.924 | 0.908 | 0.941 | 0.588 | 0.491 | 0.681 | 0.424 | 0.340 | 0.511 | 5 |
| full+B4(ext) | 0.469 | 0.407 | 0.550 | 0.053 | 0.026 | 0.080 | 0.400 | 0.000 | 0.800 | 0.904 | 0.867 | 0.944 | 0.612 | 0.560 | 0.670 | 0.000 | 0.000 | 0.000 | 1.708 | 1.649 | 1.765 | 0.410 | 0.359 | 0.473 | 0.008 | 0.004 | 0.014 | 0.884 | 0.833 | 0.939 | 0.555 | 0.514 | 0.601 | 0.930 | 0.913 | 0.945 | 0.590 | 0.527 | 0.641 | 0.368 | 0.302 | 0.423 | 5 |
| full+B5(ext) | 0.650 | 0.597 | 0.703 | 0.057 | 0.030 | 0.084 | 0.400 | 0.000 | 0.800 | 0.923 | 0.893 | 0.957 | 0.760 | 0.729 | 0.792 | 0.000 | 0.000 | 0.000 | 1.465 | 1.421 | 1.513 | 0.575 | 0.527 | 0.615 | 0.011 | 0.005 | 0.015 | 0.889 | 0.849 | 0.935 | 0.695 | 0.667 | 0.720 | 0.943 | 0.935 | 0.950 | 0.425 | 0.385 | 0.473 | 0.238 | 0.192 | 0.296 | 5 |
| full+B5(ext_win_logit) | 0.629 | 0.572 | 0.685 | 0.058 | 0.021 | 0.095 | 0.000 | 0.000 | 0.000 | 0.922 | 0.878 | 0.969 | 0.744 | 0.714 | 0.770 | 0.000 | 0.000 | 0.000 | 1.539 | 1.450 | 1.620 | 0.543 | 0.475 | 0.614 | 0.011 | 0.003 | 0.019 | 0.890 | 0.823 | 0.957 | 0.666 | 0.630 | 0.701 | 0.948 | 0.941 | 0.956 | 0.457 | 0.386 | 0.525 | 0.292 | 0.211 | 0.373 | 5 |
| B1-regex | 0.226 | 0.205 | 0.250 | 0.151 | 0.142 | 0.159 | 0.000 | 0.000 | 0.000 | 0.598 | 0.570 | 0.623 | 0.328 | 0.302 | 0.356 | 0.000 | 0.000 | 0.000 | 1.980 | 1.922 | 2.037 | 0.161 | 0.145 | 0.177 | 0.029 | 0.024 | 0.033 | 0.438 | 0.402 | 0.475 | 0.235 | 0.216 | 0.254 | 0.566 | 0.558 | 0.574 | 0.839 | 0.823 | 0.855 | 0.000 | 0.000 | 0.000 | 5 |
| B1t-regex-count | 0.000 | 0.000 | 0.000 | 0.039 | 0.005 | 0.078 | 0.200 | 0.000 | 0.600 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | - | - | - | 2.464 | 2.427 | 2.499 | 0.000 | 0.000 | 0.000 | 0.005 | 0.000 | 0.010 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.565 | 0.558 | 0.573 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 5 |
| B2-banned+emb | 0.043 | 0.033 | 0.053 | 0.061 | 0.038 | 0.085 | 0.400 | 0.000 | 0.800 | 0.438 | 0.303 | 0.572 | 0.079 | 0.061 | 0.097 | 0.000 | 0.000 | 0.000 | 2.383 | 2.329 | 2.438 | 0.031 | 0.024 | 0.038 | 0.009 | 0.005 | 0.013 | 0.380 | 0.242 | 0.518 | 0.057 | 0.044 | 0.070 | 0.755 | 0.721 | 0.783 | 0.969 | 0.962 | 0.976 | 0.000 | 0.000 | 0.000 | 5 |
| B3a-PromptGuard | 0.070 | 0.027 | 0.134 | 0.091 | 0.000 | 0.235 | 0.000 | 0.000 | 0.000 | 0.725 | 0.435 | 1.000 | 0.107 | 0.049 | 0.181 | 0.000 | 0.000 | 0.000 | 2.254 | 2.090 | 2.345 | 0.050 | 0.020 | 0.098 | 0.015 | 0.000 | 0.039 | 0.694 | 0.381 | 1.000 | 0.077 | 0.036 | 0.131 | 0.809 | 0.795 | 0.825 | 0.950 | 0.902 | 0.980 | 0.000 | 0.000 | 0.000 | 5 |
| B3b-PIGuard | 0.446 | 0.411 | 0.481 | 0.071 | 0.030 | 0.117 | 0.200 | 0.000 | 0.600 | 0.873 | 0.811 | 0.934 | 0.587 | 0.564 | 0.611 | 0.000 | 0.000 | 0.000 | 2.066 | 1.975 | 2.143 | 0.324 | 0.301 | 0.348 | 0.010 | 0.005 | 0.017 | 0.831 | 0.761 | 0.897 | 0.463 | 0.445 | 0.482 | 0.883 | 0.870 | 0.898 | 0.676 | 0.652 | 0.699 | 0.000 | 0.000 | 0.000 | 5 |
| B3c-ProtectAIv2 | 0.212 | 0.200 | 0.230 | 0.100 | 0.049 | 0.141 | 0.000 | 0.000 | 0.000 | 0.706 | 0.611 | 0.839 | 0.323 | 0.308 | 0.339 | 0.000 | 0.000 | 0.000 | 2.140 | 2.086 | 2.181 | 0.182 | 0.165 | 0.200 | 0.015 | 0.008 | 0.021 | 0.664 | 0.573 | 0.790 | 0.282 | 0.264 | 0.300 | 0.899 | 0.890 | 0.907 | 0.818 | 0.800 | 0.835 | 0.000 | 0.000 | 0.000 | 5 |
| B3a-PromptGuard+sessnorm | 0.125 | 0.013 | 0.241 | 0.040 | 0.005 | 0.076 | 0.000 | 0.000 | 0.000 | 0.771 | 0.685 | 0.886 | 0.185 | 0.026 | 0.348 | 0.000 | 0.000 | 0.000 | 2.253 | 2.046 | 2.460 | 0.090 | 0.010 | 0.175 | 0.008 | 0.000 | 0.018 | 0.706 | 0.593 | 0.864 | 0.137 | 0.019 | 0.261 | 0.754 | 0.741 | 0.769 | 0.910 | 0.825 | 0.990 | 0.000 | 0.000 | 0.000 | 5 |
| B3b-PIGuard+sessnorm | 0.227 | 0.206 | 0.251 | 0.061 | 0.049 | 0.079 | 0.800 | 0.400 | 1.000 | 0.791 | 0.765 | 0.813 | 0.351 | 0.328 | 0.376 | 0.000 | 0.000 | 0.000 | 2.190 | 2.147 | 2.233 | 0.162 | 0.149 | 0.178 | 0.010 | 0.007 | 0.013 | 0.705 | 0.666 | 0.739 | 0.262 | 0.248 | 0.279 | 0.822 | 0.808 | 0.839 | 0.838 | 0.822 | 0.851 | 0.000 | 0.000 | 0.000 | 5 |
| B3c-ProtectAIv2+sessnorm | 0.241 | 0.203 | 0.290 | 0.085 | 0.049 | 0.134 | 0.400 | 0.000 | 0.800 | 0.757 | 0.701 | 0.810 | 0.360 | 0.322 | 0.403 | 0.000 | 0.000 | 0.000 | 2.076 | 2.036 | 2.121 | 0.197 | 0.169 | 0.233 | 0.010 | 0.007 | 0.015 | 0.749 | 0.701 | 0.790 | 0.308 | 0.277 | 0.346 | 0.855 | 0.844 | 0.870 | 0.803 | 0.767 | 0.831 | 0.000 | 0.000 | 0.000 | 5 |
| B2-banned+emb+sessnorm | 0.061 | 0.057 | 0.067 | 0.054 | 0.029 | 0.079 | 0.000 | 0.000 | 0.000 | 0.567 | 0.438 | 0.688 | 0.110 | 0.101 | 0.121 | 0.000 | 0.000 | 0.000 | 2.366 | 2.320 | 2.413 | 0.044 | 0.041 | 0.047 | 0.008 | 0.004 | 0.012 | 0.478 | 0.347 | 0.597 | 0.080 | 0.073 | 0.087 | 0.636 | 0.613 | 0.655 | 0.956 | 0.953 | 0.959 | 0.000 | 0.000 | 0.000 | 5 |
| B4-OursFT2 | 0.480 | 0.460 | 0.502 | 0.056 | 0.045 | 0.067 | 0.600 | 0.200 | 1.000 | 0.896 | 0.878 | 0.915 | 0.624 | 0.607 | 0.645 | 0.000 | 0.000 | 0.000 | 1.899 | 1.852 | 1.946 | 0.388 | 0.366 | 0.411 | 0.013 | 0.010 | 0.016 | 0.813 | 0.777 | 0.844 | 0.524 | 0.502 | 0.547 | 0.842 | 0.832 | 0.854 | 0.612 | 0.589 | 0.634 | 0.000 | 0.000 | 0.000 | 5 |


## Ranking at a common achieved false-positive rate

| method | session_AUC | session_AUC_lo | session_AUC_hi | det_at_fpr_0.05 | det_at_fpr_0.05_lo | det_at_fpr_0.05_hi | det_at_fpr_0.075 | det_at_fpr_0.075_lo | det_at_fpr_0.075_hi | det_at_fpr_0.1 | det_at_fpr_0.1_lo | det_at_fpr_0.1_hi | n_seeds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| full | 0.823 | 0.806 | 0.843 | 0.321 | 0.274 | 0.368 | 0.387 | 0.348 | 0.428 | 0.430 | 0.375 | 0.485 | 5 |
| ref_only | 0.777 | 0.764 | 0.791 | 0.222 | 0.213 | 0.232 | 0.262 | 0.239 | 0.290 | 0.279 | 0.245 | 0.312 | 5 |
| memoryless | 0.732 | 0.711 | 0.754 | 0.209 | 0.187 | 0.232 | 0.230 | 0.190 | 0.276 | 0.239 | 0.203 | 0.281 | 5 |
| simplified | 0.832 | 0.819 | 0.848 | 0.348 | 0.310 | 0.387 | 0.414 | 0.371 | 0.458 | 0.478 | 0.411 | 0.554 | 5 |
| full(per-channel) | 0.821 | 0.800 | 0.845 | 0.312 | 0.250 | 0.374 | 0.384 | 0.296 | 0.465 | 0.430 | 0.330 | 0.537 | 5 |
| full+B3b(ind_win_logit_only) | 0.864 | 0.837 | 0.896 | 0.460 | 0.326 | 0.593 | 0.495 | 0.351 | 0.659 | 0.576 | 0.452 | 0.706 | 5 |
| full+B4(ext) | 0.877 | 0.860 | 0.895 | 0.499 | 0.419 | 0.582 | 0.564 | 0.476 | 0.666 | 0.597 | 0.504 | 0.701 | 5 |
| full+B5(ext) | 0.907 | 0.894 | 0.922 | 0.671 | 0.636 | 0.712 | 0.710 | 0.661 | 0.758 | 0.740 | 0.694 | 0.786 | 5 |
| full+B5(ext_win_logit) | 0.910 | 0.893 | 0.929 | 0.666 | 0.621 | 0.719 | 0.706 | 0.650 | 0.764 | 0.738 | 0.681 | 0.798 | 5 |
| B1-regex | 0.546 | 0.536 | 0.557 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 5 |
| B1t-regex-count | 0.539 | 0.531 | 0.549 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.001 | 0.000 | 0.000 | 0.001 | 5 |
| B2-banned+emb | 0.611 | 0.594 | 0.629 | 0.055 | 0.043 | 0.068 | 0.072 | 0.057 | 0.084 | 0.080 | 0.073 | 0.090 | 5 |
| B3a-PromptGuard | 0.679 | 0.649 | 0.708 | 0.059 | 0.037 | 0.082 | 0.081 | 0.060 | 0.105 | 0.098 | 0.076 | 0.120 | 5 |
| B3b-PIGuard | 0.828 | 0.813 | 0.842 | 0.448 | 0.404 | 0.490 | 0.470 | 0.430 | 0.502 | 0.490 | 0.459 | 0.520 | 5 |
| B3c-ProtectAIv2 | 0.756 | 0.742 | 0.771 | 0.191 | 0.164 | 0.219 | 0.204 | 0.179 | 0.225 | 0.217 | 0.181 | 0.245 | 5 |
| B3a-PromptGuard+sessnorm | 0.754 | 0.749 | 0.756 | 0.274 | 0.148 | 0.350 | 0.294 | 0.174 | 0.361 | 0.388 | 0.374 | 0.405 | 5 |
| B3b-PIGuard+sessnorm | 0.809 | 0.795 | 0.823 | 0.238 | 0.215 | 0.261 | 0.312 | 0.269 | 0.361 | 0.409 | 0.345 | 0.469 | 5 |
| B3c-ProtectAIv2+sessnorm | 0.762 | 0.748 | 0.775 | 0.217 | 0.202 | 0.233 | 0.226 | 0.212 | 0.241 | 0.237 | 0.224 | 0.249 | 5 |
| B2-banned+emb+sessnorm | 0.575 | 0.559 | 0.591 | 0.084 | 0.070 | 0.093 | 0.098 | 0.087 | 0.109 | 0.120 | 0.103 | 0.137 | 5 |
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
| B1-regex | direct_single | 0.377 | 0.000 | 300 |
| B1-regex | direct_spread | 0.297 | 0.000 | 300 |
| B1-regex | drift | 0.323 | 7.000 | 300 |
| B1-regex | single_chunk | 0.110 | 0.000 | 900 |
| B1-regex | split_chunks | 0.257 | 0.000 | 300 |
| B1t-regex-count | direct_single | 0.000 | - | 300 |
| B1t-regex-count | direct_spread | 0.000 | - | 300 |
| B1t-regex-count | drift | 0.000 | - | 300 |
| B1t-regex-count | single_chunk | 0.000 | - | 900 |
| B1t-regex-count | split_chunks | 0.000 | - | 300 |
| B2-banned+emb | direct_single | 0.113 | 0.000 | 300 |
| B2-banned+emb | direct_spread | 0.190 | 0.000 | 300 |
| B2-banned+emb | drift | 0.000 | - | 300 |
| B2-banned+emb | single_chunk | 0.000 | - | 900 |
| B2-banned+emb | split_chunks | 0.000 | - | 300 |
| B2-banned+emb+sessnorm | direct_single | 0.197 | 0.000 | 300 |
| B2-banned+emb+sessnorm | direct_spread | 0.230 | 0.000 | 300 |
| B2-banned+emb+sessnorm | drift | 0.000 | - | 300 |
| B2-banned+emb+sessnorm | single_chunk | 0.001 | 0.000 | 900 |
| B2-banned+emb+sessnorm | split_chunks | 0.000 | - | 300 |
| B3a-PromptGuard | direct_single | 0.000 | - | 300 |
| B3a-PromptGuard | direct_spread | 0.000 | - | 300 |
| B3a-PromptGuard | drift | 0.103 | 7.000 | 300 |
| B3a-PromptGuard | single_chunk | 0.046 | 0.000 | 900 |
| B3a-PromptGuard | split_chunks | 0.247 | 0.000 | 300 |
| B3a-PromptGuard+sessnorm | direct_single | 0.247 | 0.000 | 300 |
| B3a-PromptGuard+sessnorm | direct_spread | 0.227 | 0.000 | 300 |
| B3a-PromptGuard+sessnorm | drift | 0.000 | - | 300 |
| B3a-PromptGuard+sessnorm | single_chunk | 0.113 | 0.000 | 900 |
| B3a-PromptGuard+sessnorm | split_chunks | 0.060 | 0.000 | 300 |
| B3b-PIGuard | direct_single | 0.633 | 0.000 | 300 |
| B3b-PIGuard | direct_spread | 0.607 | 0.000 | 300 |
| B3b-PIGuard | drift | 0.013 | 5.500 | 300 |
| B3b-PIGuard | single_chunk | 0.590 | 0.000 | 900 |
| B3b-PIGuard | split_chunks | 0.100 | 0.000 | 300 |
| B3b-PIGuard+sessnorm | direct_single | 0.370 | 0.000 | 300 |
| B3b-PIGuard+sessnorm | direct_spread | 0.303 | 0.000 | 300 |
| B3b-PIGuard+sessnorm | drift | 0.047 | 4.000 | 300 |
| B3b-PIGuard+sessnorm | single_chunk | 0.249 | 0.000 | 900 |
| B3b-PIGuard+sessnorm | split_chunks | 0.120 | 0.000 | 300 |
| B3c-ProtectAIv2 | direct_single | 0.770 | 0.000 | 300 |
| B3c-ProtectAIv2 | direct_spread | 0.670 | 0.000 | 300 |
| B3c-ProtectAIv2 | drift | 0.000 | - | 300 |
| B3c-ProtectAIv2 | single_chunk | 0.003 | 0.000 | 900 |
| B3c-ProtectAIv2 | split_chunks | 0.037 | 2.000 | 300 |
| B3c-ProtectAIv2+sessnorm | direct_single | 0.760 | 0.000 | 300 |
| B3c-ProtectAIv2+sessnorm | direct_spread | 0.737 | 0.000 | 300 |
| B3c-ProtectAIv2+sessnorm | drift | 0.033 | 7.000 | 300 |
| B3c-ProtectAIv2+sessnorm | single_chunk | 0.019 | 0.000 | 900 |
| B3c-ProtectAIv2+sessnorm | split_chunks | 0.100 | 1.000 | 300 |
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
| full+B3b(ind_win_logit_only) | direct_single | 0.497 | 0.000 | 300 |
| full+B3b(ind_win_logit_only) | direct_spread | 0.550 | 0.000 | 300 |
| full+B3b(ind_win_logit_only) | drift | 0.260 | 7.000 | 300 |
| full+B3b(ind_win_logit_only) | single_chunk | 0.586 | 0.000 | 900 |
| full+B3b(ind_win_logit_only) | split_chunks | 0.443 | 0.000 | 300 |
| full+B4(ext) | direct_single | 0.337 | 0.000 | 300 |
| full+B4(ext) | direct_spread | 0.440 | 1.000 | 300 |
| full+B4(ext) | drift | 0.403 | 7.000 | 300 |
| full+B4(ext) | single_chunk | 0.519 | 0.000 | 900 |
| full+B4(ext) | split_chunks | 0.543 | 0.000 | 300 |
| full+B5(ext) | direct_single | 0.877 | 0.000 | 300 |
| full+B5(ext) | direct_spread | 0.890 | 0.000 | 300 |
| full+B5(ext) | drift | 0.380 | 7.000 | 300 |
| full+B5(ext) | single_chunk | 0.621 | 0.000 | 900 |
| full+B5(ext) | split_chunks | 0.540 | 0.000 | 300 |
| full+B5(ext_win_logit) | direct_single | 0.873 | 0.000 | 300 |
| full+B5(ext_win_logit) | direct_spread | 0.887 | 0.000 | 300 |
| full+B5(ext_win_logit) | drift | 0.377 | 7.000 | 300 |
| full+B5(ext_win_logit) | single_chunk | 0.589 | 0.000 | 900 |
| full+B5(ext_win_logit) | split_chunks | 0.500 | 0.000 | 300 |
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
| B1-regex | bipia | 0.000 | 300 |
| B1-regex | boiling_frog | 0.323 | 300 |
| B1-regex | direct_mt | 0.337 | 600 |
| B1-regex | kb | 0.292 | 600 |
| B1-regex | tasktracker | 0.003 | 300 |
| B1t-regex-count | bipia | 0.000 | 300 |
| B1t-regex-count | boiling_frog | 0.000 | 300 |
| B1t-regex-count | direct_mt | 0.000 | 600 |
| B1t-regex-count | kb | 0.000 | 600 |
| B1t-regex-count | tasktracker | 0.000 | 300 |
| B2-banned+emb | bipia | 0.000 | 300 |
| B2-banned+emb | boiling_frog | 0.000 | 300 |
| B2-banned+emb | direct_mt | 0.152 | 600 |
| B2-banned+emb | kb | 0.000 | 600 |
| B2-banned+emb | tasktracker | 0.000 | 300 |
| B2-banned+emb+sessnorm | bipia | 0.000 | 300 |
| B2-banned+emb+sessnorm | boiling_frog | 0.000 | 300 |
| B2-banned+emb+sessnorm | direct_mt | 0.213 | 600 |
| B2-banned+emb+sessnorm | kb | 0.000 | 600 |
| B2-banned+emb+sessnorm | tasktracker | 0.003 | 300 |
| B3a-PromptGuard | bipia | 0.040 | 300 |
| B3a-PromptGuard | boiling_frog | 0.103 | 300 |
| B3a-PromptGuard | direct_mt | 0.000 | 600 |
| B3a-PromptGuard | kb | 0.167 | 600 |
| B3a-PromptGuard | tasktracker | 0.010 | 300 |
| B3a-PromptGuard+sessnorm | bipia | 0.170 | 300 |
| B3a-PromptGuard+sessnorm | boiling_frog | 0.000 | 300 |
| B3a-PromptGuard+sessnorm | direct_mt | 0.237 | 600 |
| B3a-PromptGuard+sessnorm | kb | 0.077 | 600 |
| B3a-PromptGuard+sessnorm | tasktracker | 0.077 | 300 |
| B3b-PIGuard | bipia | 0.740 | 300 |
| B3b-PIGuard | boiling_frog | 0.013 | 300 |
| B3b-PIGuard | direct_mt | 0.620 | 600 |
| B3b-PIGuard | kb | 0.068 | 600 |
| B3b-PIGuard | tasktracker | 0.993 | 300 |
| B3b-PIGuard+sessnorm | bipia | 0.210 | 300 |
| B3b-PIGuard+sessnorm | boiling_frog | 0.047 | 300 |
| B3b-PIGuard+sessnorm | direct_mt | 0.337 | 600 |
| B3b-PIGuard+sessnorm | kb | 0.090 | 600 |
| B3b-PIGuard+sessnorm | tasktracker | 0.477 | 300 |
| B3c-ProtectAIv2 | bipia | 0.010 | 300 |
| B3c-ProtectAIv2 | boiling_frog | 0.000 | 300 |
| B3c-ProtectAIv2 | direct_mt | 0.720 | 600 |
| B3c-ProtectAIv2 | kb | 0.018 | 600 |
| B3c-ProtectAIv2 | tasktracker | 0.000 | 300 |
| B3c-ProtectAIv2+sessnorm | bipia | 0.017 | 300 |
| B3c-ProtectAIv2+sessnorm | boiling_frog | 0.033 | 300 |
| B3c-ProtectAIv2+sessnorm | direct_mt | 0.748 | 600 |
| B3c-ProtectAIv2+sessnorm | kb | 0.062 | 600 |
| B3c-ProtectAIv2+sessnorm | tasktracker | 0.017 | 300 |
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
| full+B3b(ind_win_logit_only) | bipia | 0.650 | 300 |
| full+B3b(ind_win_logit_only) | boiling_frog | 0.260 | 300 |
| full+B3b(ind_win_logit_only) | direct_mt | 0.523 | 600 |
| full+B3b(ind_win_logit_only) | kb | 0.353 | 600 |
| full+B3b(ind_win_logit_only) | tasktracker | 0.843 | 300 |
| full+B4(ext) | bipia | 0.377 | 300 |
| full+B4(ext) | boiling_frog | 0.403 | 300 |
| full+B4(ext) | direct_mt | 0.388 | 600 |
| full+B4(ext) | kb | 0.542 | 600 |
| full+B4(ext) | tasktracker | 0.640 | 300 |
| full+B5(ext) | bipia | 0.447 | 300 |
| full+B5(ext) | boiling_frog | 0.380 | 300 |
| full+B5(ext) | direct_mt | 0.883 | 600 |
| full+B5(ext) | kb | 0.527 | 600 |
| full+B5(ext) | tasktracker | 0.903 | 300 |
| full+B5(ext_win_logit) | bipia | 0.447 | 300 |
| full+B5(ext_win_logit) | boiling_frog | 0.377 | 300 |
| full+B5(ext_win_logit) | direct_mt | 0.880 | 600 |
| full+B5(ext_win_logit) | kb | 0.468 | 600 |
| full+B5(ext_win_logit) | tasktracker | 0.883 | 300 |
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
| full | 29.815 | 44.947 | 1.512 | 3269.000 | 63816.600 |
| full(per-channel) | 30.786 | 61.413 | 1.512 | 3269.000 | 63827.200 |
| full+B3b(ind_win_logit_only) | 29.955 | 43.639 | 1.512 | 3269.000 | 64126.000 |
| full+B4(ext) | 43.093 | 58.531 | 1.512 | 3269.000 | 75583.600 |
| full+B5(ext) | 43.102 | 68.517 | 1.512 | 3269.000 | 75304.800 |
| full+B5(ext_win_logit) | 44.011 | 60.902 | 1.512 | 3269.000 | 94128.600 |
| memoryless | 31.322 | 63.628 | 1.512 | 3269.000 | 63815.400 |
| ref_only | 31.214 | 63.837 | 1.512 | 3269.000 | 63781.000 |
| simplified | 30.750 | 64.159 | 1.512 | 3269.000 | 63863.600 |
