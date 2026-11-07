| path | input | output | prefill ms | decode ms | decode tok/s | e2e tok/s | peak MiB |
|---|---|---|---|---|---|---|---|
| hf_cached | 128 | 32 | 146.3 | 1054.6 | 29.4 | 26.6 | 5980 |
| manual_cached | 128 | 32 | 124.5 | 1014.7 | 30.6 | 28.1 | 5926 |
| hf_uncached | 128 | 32 | 144.7 | 7420.4 | 4.2 | 4.2 | 6083 |
| manual_uncached | 128 | 32 | 125.4 | 6272.6 | 4.9 | 5.0 | 5921 |
| hf_cached | 512 | 32 | 488.1 | 1325.4 | 23.4 | 17.6 | 6229 |
| manual_cached | 512 | 32 | 417.4 | 1041.5 | 29.8 | 21.9 | 6004 |
| hf_uncached | 512 | 32 | 481.3 | 19808.6 | 1.6 | 1.6 | 6530 |
| manual_uncached | 512 | 32 | 432.2 | 17464.3 | 1.8 | 1.8 | 5977 |

Correctness (token equality vs hf_uncached):

- real_prompt_eos (L=370, N=13): hf_uncached=OK, hf_cached=OK, manual_uncached=OK, manual_cached=OK
- L128_N32_forced (L=128, N=32): hf_uncached=OK, hf_cached=OK, manual_uncached=OK, manual_cached=OK
- L512_N32_forced (L=512, N=32): hf_uncached=OK, hf_cached=OK, manual_uncached=OK, manual_cached=OK