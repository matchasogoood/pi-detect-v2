### H1: cost per configuration (latency per event, memory per session)

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
