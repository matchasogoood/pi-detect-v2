### H1: cost per configuration (latency per event, memory per session)

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
