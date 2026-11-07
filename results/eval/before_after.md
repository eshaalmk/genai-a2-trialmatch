| model | split | n | accuracy | macro_f1 | incl_acc | incl_macro_f1 | excl_acc | excl_macro_f1 | evid_micro_p | evid_micro_r | evid_micro_f1 | evid_macro_f1 | invalid_rate | json_valid_rate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base | val | 206 | 0.296 | 0.191 | 0.561 | 0.289 | 0.171 | 0.122 | 0.111 | 0.157 | 0.130 | 0.110 | 0.000 | 1.000 |
| finetuned | val | 206 | 0.723 | 0.353 | 0.727 | 0.374 | 0.721 | 0.210 | 0.703 | 0.391 | 0.503 | 0.415 | 0.000 | 1.000 |
| base | test | 206 | 0.383 | 0.207 | 0.707 | 0.377 | 0.198 | 0.090 | 0.148 | 0.266 | 0.190 | 0.172 | 0.000 | 1.000 |
| finetuned | test | 206 | 0.728 | 0.465 | 0.787 | 0.426 | 0.695 | 0.273 | 0.854 | 0.372 | 0.519 | 0.407 | 0.000 | 1.000 |

Per-example label changes (same annotation IDs):
```
{
  "val": {
    "n_common": 206,
    "fixed_by_finetuning": 115,
    "broken_by_finetuning": 27
  },
  "test": {
    "n_common": 206,
    "fixed_by_finetuning": 97,
    "broken_by_finetuning": 26
  }
}
```