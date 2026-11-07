| setting | method | draft | k | tok/s | speedup | accept rate | acc/round | tok/target fwd | peak MiB |
|---|---|---|---|---|---|---|---|---|---|
| forced_128 | greedy_target | - | - | 21.4 | 1.00 | - | - | - | 4909 |
| forced_128 | speculative | draft_base | 2 | 14.0 | 0.66 | 0.79 | 1.58 | 2.56 | 4915 |
| forced_128 | speculative | draft_base | 4 | 14.3 | 0.67 | 0.73 | 2.89 | 3.87 | 4915 |
| forced_128 | speculative | draft_base | 6 | 13.0 | 0.61 | 0.62 | 3.64 | 4.61 | 4915 |
| forced_128 | speculative | draft_finetuned | 2 | 14.3 | 0.67 | 0.83 | 1.65 | 2.63 | 4915 |
| forced_128 | speculative | draft_finetuned | 4 | 14.7 | 0.69 | 0.75 | 2.97 | 3.94 | 4915 |
| forced_128 | speculative | draft_finetuned | 6 | 12.4 | 0.58 | 0.57 | 3.39 | 4.36 | 4915 |
| natural_eos | greedy_target | - | - | 14.1 | 1.00 | - | - | - | 4908 |
| natural_eos | hf_assisted_reference | draft_base | - | 9.8 | 0.81 | - | - | - | 0 |
| natural_eos | hf_assisted_reference | draft_finetuned | - | 11.4 | 0.94 | - | - | - | 0 |
| natural_eos | speculative | draft_base | 2 | 9.2 | 0.65 | 0.75 | 1.48 | 2.41 | 4914 |
| natural_eos | speculative | draft_base | 4 | 9.3 | 0.66 | 0.71 | 2.48 | 3.27 | 4914 |
| natural_eos | speculative | draft_base | 6 | 9.0 | 0.64 | 0.64 | 3.17 | 3.93 | 4914 |
| natural_eos | speculative | draft_finetuned | 2 | 10.1 | 0.72 | 0.97 | 1.88 | 2.75 | 4914 |
| natural_eos | speculative | draft_finetuned | 4 | 10.7 | 0.76 | 0.94 | 3.60 | 4.36 | 4914 |
| natural_eos | speculative | draft_finetuned | 6 | 10.9 | 0.77 | 0.93 | 5.12 | 5.85 | 4914 |

Token-equality rate vs greedy target:
```
{
  "draft_base_hf_assisted_equal": 0.6666666666666666,
  "draft_base_k2": 0.9166666666666666,
  "draft_base_k4": 0.7916666666666666,
  "draft_base_k6": 0.8333333333333334,
  "draft_finetuned_hf_assisted_equal": 0.6666666666666666,
  "draft_finetuned_k2": 0.8333333333333334,
  "draft_finetuned_k4": 0.8333333333333334,
  "draft_finetuned_k6": 0.75,
  "greedy_manual_vs_hf_generate": 0.6666666666666666
}
```