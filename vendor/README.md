# Upstream model code and datasets

The [upstream](upstream) Git submodule supplies the original model, preprocessing
scripts, and raw and processed data for Electricity, PM2.5 and Exchange.
Use `git clone --recurse-submodules` for the group repository. From an existing
checkout, `git submodule update --init` fetches the recorded revision.

Run `python scripts/setup_upstream.py` at the repository root to create the
patched working copy in `vendor/DeLELSTM/`. Preparation verifies the six core
source files and all nine CSVs against the [manifest](../patches/upstream.json).
The server training script performs this step automatically. [Patch details](../patches/README.md) · [Third-party notices](../THIRD_PARTY_NOTICES.md)
