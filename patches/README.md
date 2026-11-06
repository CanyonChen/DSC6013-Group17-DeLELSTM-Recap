# Reproduction patch

`reproduction.patch` is one consolidated diff from upstream commit
`b9f29dbe48bd2204e5ce7edcdbbfd55f031d30a9` to the code supplied with the course
project. It replaces the old sequential `port.patch` and `changes.patch` workflow.

It changes five files: `Electricity.py`, `PM.py`, `Exchange.py`,
`model_training.py`, and `networks.py`. Changes include relative data paths,
CUDA device selection, updated NumPy/scheduler APIs, model and epoch controls,
fixed/resampled split handling, independent training seeds, seed-aware output
directories, and RETAIN dropout semantics. The model architecture, loss, and
default epoch counts are retained.

Run `python scripts/setup_upstream.py` from the repository root. The script
downloads that exact upstream revision, applies the patch with Git, and checks
the six core Python files against `upstream.json` (SHA-256 after LF
normalization). `config.py` is unchanged but included in the verification.

To use the locally retained original archive without downloading:

```bash
python scripts/setup_upstream.py --archive local_archive/original/DeLELSTM-main.zip
```

The archive's Git commit identifier is checked before applying the patch.
An existing prepared directory is verified rather than overwritten. A modified
local source tree should be kept separately before preparing another copy.

The upstream source context in the diff remains third-party material; see
[THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).
