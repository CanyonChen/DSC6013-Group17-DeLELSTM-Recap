# Third-party materials

The root [MIT License](LICENSE) covers original contributions by the DS6013
DeLELSTM project contributors. It does not relicense upstream code, datasets,
papers, or unchanged upstream context included in patches.

## DeLELSTM

- Authors: Chaoqun Wang, Yijun Li, Xiangqian Sun, Qi Wu, Dongdong Wang, and Zhixiang Huang.
- Repository: <https://github.com/wangcq01/DeLELSTM>
- Pinned commit: [`b9f29dbe48bd2204e5ce7edcdbbfd55f031d30a9`](https://github.com/wangcq01/DeLELSTM/commit/b9f29dbe48bd2204e5ce7edcdbbfd55f031d30a9).
- Paper: *DeLELSTM: Decomposition-based Linear Explainable LSTM to Capture Instantaneous and Long-term Effects in Time Series*, IJCAI 2023, pp. 4299–4307, [DOI](https://doi.org/10.24963/ijcai.2023/478).

The supplied upstream archive at this commit contains no explicit license.
The group repository pins the authors' GitHub source and data through the
[`vendor/upstream` submodule](vendor/upstream). A recursive clone retrieves
the exact revision. Setup prepares a working copy in `vendor/DeLELSTM/`.
This repository distributes the project's changes
as [a patch](patches/reproduction.patch), with attribution retained here;
no MIT grant is asserted over the upstream portions of that patch.

The local reference paper is retained in `local_archive/references/` and is not
part of the Git publication. Use the DOI above to obtain the paper.

## Data sources

The submodule includes three raw CSVs and six processed CSVs at the recorded
revision. The chronological PM2.5 experiment reconstructs its inputs and
observed targets from the raw hourly records. The historical protocols use
the released processed files. The original data sources are:

- [Beijing PM2.5, UCI](https://archive.ics.uci.edu/ml/datasets/Beijing+PM2.5+Data)
- [Electricity consumption, Kaggle](https://www.kaggle.com/datasets/unajtheb/homesteadus-electricity-consumption)
- [Exchange rates](https://github.com/laiguokun/multivariate-time-series-data)

These datasets retain their respective terms; the project MIT license does
not replace them. Generated experimental metrics and explanations are stored
in `results/`, separately from the input datasets.
