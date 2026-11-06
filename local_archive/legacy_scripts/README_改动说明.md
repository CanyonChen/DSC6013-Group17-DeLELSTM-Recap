# DeLELSTM 改进版 · 改动说明

在原来的复现代码上改了 **6 个文件、共 9 处**（另加 1 个新增脚本）。只做两件事：

1. **让「5 次重复实验」真正可做** —— 加一个划分随机化开关，并修掉 3 个会让结果作废的坑；
2. **让汇总直接输出 mean ± std** —— 原来只会输出 75 行明细。

模型结构、损失函数、优化器、超参数、75/15/10 的划分比例**一律没有改动**。
不加 `--resample_split` 时，行为与作者原代码等价。

---

## 文件清单

| 文件 | 状态 | 说明 |
|---|---|---|
| `DeLELSTM-main/Electricity.py` | 改 | 加开关、抽 `prepare()`、划分移进循环 |
| `DeLELSTM-main/PM.py` | 改 | 同上 + 删死代码 + 修空目录 |
| `DeLELSTM-main/Exchange.py` | 改 | 同上 + 删死代码 |
| `DeLELSTM-main/model_training.py` | 改 | 3 处结果目录名加 seed |
| `DeLELSTM-main/networks.py` | 改 | 1 处 Dropout 语义（不改变已有结果） |
| `summarize.py` | 改 | 聚合 mean±std、多目录、明细分离 |
| `DeLELSTM-main/config.py` | 未改 | — |
| `DeLELSTM-main/model_training.py` 以外 | 未改 | 训练循环、指标计算均原样 |
| `DeLELSTM-main/DATA/` | 未改 | 原始与处理后的数据 |
| `changes.patch` | 新增 | 完整 unified diff（456 行），可用 `git apply` 复现 |
| `port.patch` | 原样保留 | 原作者代码的兼容性补丁（路径 / cuda / 失效 API） |
| `run_all.ps1` | 新增 | 一键跑方案 A / B |
| `README_改动说明.md` | 新增 | 本文件 |

---

## 逐条改动

### 改动 1 ｜新增 `--resample_split` 开关

**位置**：`Electricity.py` 第 24–26 行、`PM.py` / `Exchange.py` 第 25–27 行（紧接 `--save_models` 之后）

```python
# [改动 1] 新增开关：每个 exp 是否重新播种并重新划分数据
parser.add_argument('--resample_split', action='store_true',
                    help='每个 exp 重新播种并重新划分；默认 False = 与作者代码一致（固定划分）')
```

原代码没有这个参数，`--num_exp 5` 时 5 次实验**共用同一份划分**（划分在第 48/51 行就算完，而在第 94/97 行才进循环）。

---

### 改动 2 ｜把数据准备抽成 `prepare(seed)`

**位置对照**：

| 脚本 | 原来（直线代码，只跑一次） | 现在 |
|---|---|---|
| `Electricity.py` | 第 36–81 行 | 第 32–89 行 `def prepare(seed)` |
| `PM.py` | 第 39–84 行 | 第 33–91 行 |
| `Exchange.py` | 第 39–84 行 | 第 33–90 行 |

函数体与原文**逐行一致**，只做两件事：整体缩进、把写死的 `np.random.seed(args.seed)` 换成参数 `seed`。

三处细节值得说明：

1. **读 CSV 也一起搬进了函数**。归一化是原地修改 `data`，如果 `data` 留在函数外重复调用 `prepare()` 会二次归一化。
2. **函数结尾新增一行日志**，每次划分直接把样本数打出来，肉眼就能确认划分是否变了：
   ```python
   print(f'[prepare] seed={seed}  train={len(train_idx)}  val={len(val_idx)}  test={len(test_idx)}')
   ```
3. **各数据集的列切片保持原样**：Electricity 用 `columns[2:]`、PM2.5 用 `columns[6:]`、Exchange 用 `columns[1:]`。

---

### 改动 3 ｜播种位置：从模块顶部挪到主流程 ⚠️ 最容易改错的一处

**原代码**（`Electricity.py` 第 29–32 行，在 `if __name__ == '__main__':` 最上方）：

```python
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
```

**现在**（`Electricity.py` 第 94–97 行）：

```python
# [改动 3] 播种从"模块级一次性"移到主流程。
# 固定划分模式下只在进入循环前种一次（等价作者原行为：5 次 exp 的初始化与 batch 顺序仍不同）。
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
```

`np.random.seed` 移进了 `prepare()`（第 40 行），成为参数。

> **⚠️ 为什么不把 `torch.manual_seed` 也搬进循环？**
> 固定划分模式下，如果每个 exp 都重置成同一个种子，5 次 exp 会拿到**完全相同的初始化**，
> 算出来的 std 就是 0。这是"跑了 5 次但结果其实是假的"最典型的坑。

---

### 改动 4 ｜只有重采样模式才在循环内重置 torch 种子

**位置**：`Electricity.py` 第 110–118 行、`PM.py` 第 115–122 行、`Exchange.py` 第 110–117 行

```python
for exp_id in range(args.num_exp):
    seed = args.seed + exp_id if args.resample_split else args.seed
    if args.resample_split:
        # [改动 4] 只有重采样模式才在循环内重置 torch 种子
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # [改动 5] 划分 + 构造 loader 从循环外移进循环内
    train_loader, val_loader, test_loader = prepare(seed)

    for model_name in model_list:
        ...
```

**注意调用位置**：`prepare()` 在 `for model_name` 之外、`for exp_id` 之内。
即**每个 exp 调用一次，不是每个模型一次** —— 同一 exp 内 5 个模型必须面对同一份数据，否则横向比较不成立。

---

### 改动 5 ｜删掉循环外的划分代码

原来第 36–81 行（PM / Exchange 是 39–84 行）那一整段已经在 `prepare()` 里，`if __name__` 中不再重复。
`save_path` 的创建保留，只是简化成：

```python
save_path = os.path.join(args.save_dirs, 'Electricity')
os.makedirs(save_path, exist_ok=True)
```

---

### 改动 6 ｜`PM.py`：删死代码 + 修空目录

**删死代码**（原第 91–92 行）：

```python
model_list = ['Delelstm', 'IMV_full', 'IMV_tensor', 'Retain', 'LSTM']
model_list= [   'Retain']      # ← 不注意到就会只跑 Retain，白等 40 分钟
```

这两行都被第 95 行的环境变量开关覆盖，属于无效代码；更糟的是第 92 行容易被误读成"只跑 Retain"。
现在只保留环境变量开关作为唯一来源。

**修空目录**（原第 86 行）：

```python
- save_path = os.path.join(args.save_dirs, 'newrevision_pm')
+ save_path = os.path.join(args.save_dirs, 'newrevision_pmc')
```

原作者代码里 `PM.py` 建的是 `newrevision_pm`，而真正写结果的 `model_training.py` 第 434 行用的是 `newrevision_pmc`。
前者只是个空目录，不影响结果，改过来之后目录结构更清楚。

---

### 改动 7 ｜`Exchange.py`：删死代码

原第 91–92 行同样是被覆盖的两行（`model_list = [...]` 和 `#model_list=['LSTM']`），已删除。

---

### 改动 8 ｜`model_training.py`：结果目录名带 seed（3 处）

| 行号 | 原来 | 现在 |
|---|---|---|
| 20 | `'exp_' + str(exp_id)` | `f'exp_{args.seed}_{exp_id}'` |
| 227 | `'exp_' + str(exp_id)` | `f'exp_{args.seed}_{exp_id}'` |
| 434 | `'exp_' + str(exp_id)` | `f'exp_{args.seed}_{exp_id}'` |

**为什么改**：`exp_id` 在每次独立运行时都从 0 开始。若用「5 个独立进程、不同 `--seed`」的方式做重复实验，
第 2 次运行就会**静默覆盖**第 1 次的 `exp_0`，26 小时白跑。

目录名仍以 `exp_` 开头，`summarize.py` 的扫描逻辑不用改。

---

### 改动 9 ｜`networks.py`：修 Dropout 语义（第 379 行）

```python
- self.dropout = nn.Dropout(self.keep_prob)
+ self.dropout = nn.Dropout(1 - self.keep_prob)
```

原写法把 `keep_prob`（保留率）当成 `drop_prob`（丢弃率）传给 `nn.Dropout`，语义是反的。
当前 `keep_prob=1.0` 且被第 421 行 `if self.keep_prob < 1` 挡住，所以**这次修改不改变任何已有结果**，只是修掉潜在陷阱。
若以后要把 `keep_prob` 调成 0.8，原代码会丢弃 80% 而新代码丢弃 20%。

---

### 改动 10 ｜`summarize.py`：输出 mean ± std

| | 原来 | 现在 |
|---|---|---|
| 聚合 | 无，每个 exp 一行 | 按 数据集×模型 聚合，输出 `mean±std` |
| `--num_exp 5` 后果 | 75 行明细，没有一行是均值 | `summary.csv` 15 行，带 std |
| 明细 | 混在 `summary.csv` | 单独存 `summary_detail.csv` |
| n=1 | 无法表达 | 输出 `1.6739 (n=1)`，不写 `±nan` |
| 多目录 | 不支持 | `python summarize.py results_A results_B` |
| 无结果 | `SystemExit("no results yet")` | 明确提示检查 `--save_dirs` |

用法：

```powershell
python summarize.py                       # 汇总 ./results
python summarize.py results_A results_B    # 汇总多个目录
```

---

## 怎么跑

### ① 冒烟测试（1–2 分钟，先确认改动生效）

```powershell
cd DeLELSTM_改进版\DeLELSTM-main
$env:DELE_MODELS='LSTM'
$env:DELE_EPOCHS='2'
python Electricity.py --num_exp 3 --resample_split --save_dirs ..\smoke
```

**期望看到**：

- 打出三行 `[prepare] seed=555 ...` / `seed=556 ...` / `seed=557 ...`，且 train/val/test 的样本数**每次不同**
- `smoke\Electricity\` 下出现 `exp_555_0`、`exp_556_0`、`exp_557_0` 三个文件夹（不是 `exp_0/1/2`）
- 开头有一行 `[info] dataset=Electricity  mode=RESAMPLE-SPLIT ...`

### ② 方案 A：固定划分（对齐论文口径）

```powershell
.\run_all.ps1 -Mode A -NumExp 5
```

### ③ 方案 B：每次重新划分（推荐作为主实验）

```powershell
.\run_all.ps1 -Mode B -NumExp 5
```

### ④ 汇总

```powershell
python summarize.py results_A
python summarize.py results_B
```

### 单独跑某个数据集

```powershell
cd DeLELSTM-main
python Electricity.py --num_exp 5 --resample_split --save_dirs ..\results_B
python PM.py          --num_exp 5 --resample_split --save_dirs ..\results_B
python Exchange.py    --num_exp 5 --resample_split --save_dirs ..\results_B
```

---

## 我验证了什么

- 6 个改动文件全部通过 `python -m py_compile`（语法无误）
- 静态检查：三个入口脚本都定义了 `prepare`、都在循环里调用、都带 `--resample_split`、主流程里已无 `np.random.seed`
- `summarize.py` 用现有 15 组结果**实跑通过**（n=1 分支），并临时复制 2 个 exp 验证了 `mean±std` 分支
- `changes.patch` 已用一个独立的 hunk 解析器**逐行回放验证**：把它打到原始 6 个文件上，结果与改进版**逐行完全一致**
  - 注：原 `Electricity.py` / `PM.py` / `Exchange.py` 末尾缺换行符，patch 会顺带补上，不影响任何逻辑

## 我没能验证什么（需要你在有 GPU 的环境确认）

**这台机器没有装 torch，所以没有真正跑过训练。** 下面三点只能由你冒烟确认：

1. 三次 `[prepare]` 打印的 train/val/test 样本数确实不同
2. 结果目录是 `exp_555_0` 而不是 `exp_0`
3. 两次 `python Electricity.py --num_exp 1 --seed 601/602 --save_dirs ..\x` 不会互相覆盖

另外建议做一次**一致性抽查**：不加 `--resample_split`、`--seed 555`、`DELE_EPOCHS=1` 跑一遍，
对比改前改后的第一个 epoch 训练 loss 是否一致（应当一致，因为固定划分模式下行为等价）。
如果不一致，说明重构动了不该动的东西，请把两边的 loss 贴给我。

---

## 兼容性

- 不加 `--resample_split` → 固定划分，5 次 exp 的初始化与 batch 顺序不同，与作者原代码等价
- 旧结果（`results/` 里的 `exp_0`）仍能被 `summarize.py` 正确读取
- 新结果目录名多了 seed 前缀，不会再互相覆盖
