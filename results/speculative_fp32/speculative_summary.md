| setting | method | draft | k | tok/s | speedup | accept rate | acc/round | tok/target fwd | peak MiB |
|---|---|---|---|---|---|---|---|---|---|
| forced_128 | greedy_target | - | - | 26.6 | 1.00 | - | - | - | 7866 |
| forced_128 | speculative | draft_base | 2 | 21.9 | 0.82 | 0.79 | 1.58 | 2.56 | 7878 |
| forced_128 | speculative | draft_base | 4 | 19.8 | 0.75 | 0.66 | 2.64 | 3.62 | 7878 |
| forced_128 | speculative | draft_base | 6 | 17.4 | 0.66 | 0.55 | 3.27 | 4.24 | 7879 |
| natural_eos | greedy_target | - | - | 21.2 | 1.00 | - | - | - | 7865 |
| natural_eos | hf_assisted_reference | draft_base | - | 15.1 | 0.71 | - | - | - | 0 |
| natural_eos | speculative | draft_base | 2 | 15.4 | 0.73 | 0.75 | 1.50 | 2.44 | 7876 |
| natural_eos | speculative | draft_base | 4 | 14.9 | 0.70 | 0.74 | 2.85 | 3.70 | 7876 |
| natural_eos | speculative | draft_base | 6 | 14.8 | 0.70 | 0.63 | 3.41 | 4.22 | 7876 |

Token-equality rate vs greedy target:
```
{
  "draft_base_hf_assisted_equal": 1.0,
  "draft_base_k2": 1.0,
  "draft_base_k4": 1.0,
  "draft_base_k6": 1.0,
  "greedy_manual_vs_hf_generate": 1.0
}
```

Target top-2 logit margin at each mismatch: {"n_mismatches": 0, "max_margin": null, "margins": []}