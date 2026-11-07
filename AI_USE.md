# AI-use acknowledgement

- **Claude (Anthropic)** was used as a coding assistant to scaffold the repository structure, the
  training/evaluation scripts, the manual KV-cached decoder, the tiled attention, the speculative
  decoder, the Gradio app and the Colab notebook.
- All reused or generated code was run and tested by me: unit tests in `tests/`, token-equality
  tests against Hugging Face reference decoding (`scripts/bench_kvcache.py`,
  `scripts/bench_speculative.py`) and numerical tests of the attention kernels
  (`scripts/bench_attention.py`). Results in `results/` come from my own runs.
- Libraries used: PyTorch, Hugging Face Transformers / PEFT / Datasets, scikit-learn, Gradio,
  matplotlib. Hugging Face `generate()` (including assisted generation) is used only as a
  reference for correctness checks; the graded mechanisms are implemented in `trialmatch/`.
- _Edit this file to describe exactly how you used AI tools before submitting._
