| path | input | output | prefill ms | decode ms | decode tok/s | e2e tok/s | peak MiB |
|---|---|---|---|---|---|---|---|
| hf_cached | 128 | 32 | 178.0 | 1318.6 | 23.5 | 21.3 | 2998 |
| manual_cached | 128 | 32 | 158.7 | 1218.8 | 25.4 | 23.2 | 2967 |
| hf_uncached | 128 | 32 | 189.8 | 7976.4 | 3.9 | 3.9 | 3046 |
| manual_uncached | 128 | 32 | 168.0 | 6800.0 | 4.6 | 4.6 | 2965 |
| hf_cached | 128 | 96 | 176.7 | 3987.5 | 23.8 | 23.1 | 2998 |
| manual_cached | 128 | 96 | 158.3 | 4201.1 | 22.6 | 22.0 | 2969 |
| hf_uncached | 128 | 96 | 191.7 | 27473.7 | 3.5 | 3.5 | 3083 |
| manual_uncached | 128 | 96 | 166.3 | 23523.4 | 4.0 | 4.1 | 2971 |
| hf_cached | 512 | 32 | 705.9 | 1290.7 | 24.0 | 16.0 | 3128 |
| manual_cached | 512 | 32 | 594.2 | 1231.6 | 25.2 | 17.5 | 3020 |
| hf_uncached | 512 | 32 | 709.0 | 24729.0 | 1.3 | 1.3 | 3271 |
| manual_uncached | 512 | 32 | 601.5 | 21017.0 | 1.5 | 1.5 | 3010 |
| hf_cached | 512 | 96 | 697.2 | 4268.3 | 22.3 | 19.3 | 3128 |
| manual_cached | 512 | 96 | 595.6 | 3980.7 | 23.9 | 21.0 | 3021 |
| hf_uncached | 512 | 96 | 700.7 | 78236.5 | 1.2 | 1.2 | 3307 |
| manual_uncached | 512 | 96 | 603.6 | 66543.7 | 1.4 | 1.4 | 3020 |
| hf_cached | 1024 | 32 | 1482.4 | 1307.6 | 23.7 | 11.5 | 3302 |
| manual_cached | 1024 | 32 | 1269.2 | 1237.0 | 25.1 | 12.8 | 3139 |
| hf_uncached | 1024 | 32 | 1488.3 | 49220.9 | 0.6 | 0.6 | 3569 |
| manual_uncached | 1024 | 32 | 1273.5 | 42062.7 | 0.7 | 0.7 | 3117 |
| hf_cached | 1024 | 96 | 1488.4 | 4771.8 | 19.9 | 15.3 | 3302 |
| manual_cached | 1024 | 96 | 1268.3 | 4052.8 | 23.4 | 18.0 | 3141 |
| hf_uncached | 1024 | 96 | 1489.8 | 153600.6 | 0.6 | 0.6 | 3605 |
| manual_uncached | 1024 | 96 | 1273.8 | 131358.4 | 0.7 | 0.7 | 3138 |

Correctness (token equality vs hf_uncached):

- real_prompt_eos (L=370, N=13): hf_uncached=OK, hf_cached=OK, manual_uncached=OK, manual_cached=OK
- L128_N32_forced (L=128, N=32): hf_uncached=OK, hf_cached=MISMATCH@27, manual_uncached=OK, manual_cached=MISMATCH@27
- L128_N96_forced (L=128, N=96): hf_uncached=OK, hf_cached=MISMATCH@27, manual_uncached=OK, manual_cached=MISMATCH@27
- L512_N32_forced (L=512, N=32): hf_uncached=OK, hf_cached=OK, manual_uncached=OK, manual_cached=OK
- L512_N96_forced (L=512, N=96): hf_uncached=OK, hf_cached=OK, manual_uncached=OK, manual_cached=OK
- L1024_N32_forced (L=1024, N=32): hf_uncached=OK, hf_cached=OK, manual_uncached=OK, manual_cached=OK
- L1024_N96_forced (L=1024, N=96): hf_uncached=OK, hf_cached=OK, manual_uncached=OK, manual_cached=OK