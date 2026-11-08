| setting | method | draft | k | tok/s | speedup | accept rate | acc/round | tok/target fwd | peak MiB |
|---|---|---|---|---|---|---|---|---|---|
| natural_eos | greedy_target | - | - | 18.3 | 1.00 | - | - | - | 9750 |
| natural_eos | hf_assisted_reference | draft_base | - | 13.9 | 0.87 | - | - | - | 0 |
| natural_eos | hf_assisted_reference | draft_finetuned | - | 14.8 | 0.92 | - | - | - | 0 |
| natural_eos | speculative | draft_base | 2 | 13.6 | 0.75 | 0.74 | 1.48 | 2.40 | 9761 |
| natural_eos | speculative | draft_base | 4 | 13.4 | 0.73 | 0.70 | 2.45 | 3.23 | 9761 |
| natural_eos | speculative | draft_base | 6 | 12.4 | 0.68 | 0.63 | 3.13 | 3.89 | 9761 |
| natural_eos | speculative | draft_finetuned | 2 | 14.1 | 0.77 | 0.97 | 1.88 | 2.75 | 9761 |
| natural_eos | speculative | draft_finetuned | 4 | 14.8 | 0.81 | 0.93 | 3.60 | 4.37 | 9761 |
| natural_eos | speculative | draft_finetuned | 6 | 14.7 | 0.81 | 0.91 | 4.96 | 5.70 | 9761 |

Token-equality rate vs greedy target:
```
{
  "draft_base_hf_assisted_equal": 0.6666666666666666,
  "draft_base_k2": 1.0,
  "draft_base_k4": 1.0,
  "draft_base_k6": 1.0,
  "draft_finetuned_hf_assisted_equal": 0.6666666666666666,
  "draft_finetuned_k2": 1.0,
  "draft_finetuned_k4": 1.0,
  "draft_finetuned_k6": 1.0,
  "greedy_manual_vs_hf_generate": 0.6666666666666666
}
```

Target top-2 logit margin at each mismatch: {"n_mismatches": 0, "max_margin": null, "margins": []}