### H1: session-level ranking. session_AUC scores a session by its highest event score; det_at_fpr_x is the detection rate when every method is placed at the same *achieved* false-positive rate x on benign sessions, so that the comparison does not also compare how well each threshold transferred from validation to test.

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
