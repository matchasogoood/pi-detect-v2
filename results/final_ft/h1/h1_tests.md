### H1: paired bootstrap of the session detection rate difference

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
