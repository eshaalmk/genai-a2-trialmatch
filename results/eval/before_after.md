| model | split | n | accuracy | macro_f1 | incl_acc | incl_macro_f1 | excl_acc | excl_macro_f1 | evid_micro_p | evid_micro_r | evid_micro_f1 | evid_macro_f1 | invalid_rate | json_valid_rate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base | val | 206 | 0.461 | 0.263 | 0.545 | 0.318 | 0.421 | 0.177 | 0.072 | 0.417 | 0.122 | 0.211 | 0.000 | 1.000 |
| finetuned | val | 206 | 0.718 | 0.349 | 0.712 | 0.366 | 0.721 | 0.210 | 0.500 | 0.574 | 0.534 | 0.577 | 0.000 | 1.000 |
| base | test | 206 | 0.267 | 0.173 | 0.360 | 0.151 | 0.214 | 0.122 | 0.077 | 0.564 | 0.136 | 0.317 | 0.000 | 1.000 |
| finetuned | test | 206 | 0.733 | 0.468 | 0.800 | 0.431 | 0.695 | 0.273 | 0.586 | 0.617 | 0.601 | 0.569 | 0.000 | 1.000 |

Per-example label changes (same annotation IDs):
```
{
  "val": {
    "n_common": 206,
    "fixed_by_finetuning": 64,
    "broken_by_finetuning": 11
  },
  "test": {
    "n_common": 206,
    "fixed_by_finetuning": 100,
    "broken_by_finetuning": 4
  }
}
```