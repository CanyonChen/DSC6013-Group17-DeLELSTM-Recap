# -*- coding: utf-8 -*-
"""Minimal port patch: make the released DeLELSTM code run on this machine.

Rules:
  * only touch what is impossible to run otherwise (paths, device index,
    removed torch/numpy APIs, model selection, epoch override);
  * never change model math, data split, loss or hyper-parameters;
  * keep a unified diff (port.patch) next to the patched tree.
"""
import difflib
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "DeLELSTM-main"
PRISTINE = ROOT / "_pristine"

if not PRISTINE.exists():
    shutil.copytree(SRC, PRISTINE)

DEV = 'torch.device("cuda:0" if torch.cuda.is_available() else "cpu")'
MODEL_LINE = (
    "    model_list = os.environ.get('DELE_MODELS', "
    "'Delelstm,IMV_full,IMV_tensor,Retain,LSTM').split(',')"
)

DATASETS = {
    "Electricity.py": ("elec", "Electricity"),
    "PM.py": ("pm", "PM2.5"),
    "Exchange.py": ("exchange", "Exchange"),
}


def patch_entry(name, sub, folder):
    p = SRC / name
    t = p.read_text(encoding="utf-8")
    t = t.replace("np.in1d", "np.isin")
    t = re.sub(r'torch\.device\("cuda:\d" if torch\.cuda\.is_available\(\) else "cpu"\)', DEV, t)
    t = re.sub(
        r"data_path = r'C:\\clean\\Explain_time_series\\Code\\DATA\\Processed\\"
        + sub + "'",
        "data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'DATA', '"
        + folder + "')",
        t,
    )
    t = t.replace(
        "    for exp_id in range(args.num_exp):",
        MODEL_LINE + "\n\n    for exp_id in range(args.num_exp):",
    )
    p.write_text(t, encoding="utf-8")


def patch_training():
    p = SRC / "model_training.py"
    t = p.read_text(encoding="utf-8")
    t = t.replace(", verbose=True,", ",")
    t = re.sub(
        r"^(\s*)epochs\s*=\s*(\d+)\s*$",
        r'\1epochs = int(os.environ.get("DELE_EPOCHS", \2))',
        t,
        flags=re.M,
    )
    p.write_text(t, encoding="utf-8")


for n, (sub, folder) in DATASETS.items():
    patch_entry(n, sub, folder)
patch_training()

# ---- unified diff for auditing ----
chunks = []
for f in sorted(PRISTINE.glob("*.py")):
    a = f.read_text(encoding="utf-8").splitlines(keepends=True)
    b = (SRC / f.name).read_text(encoding="utf-8").splitlines(keepends=True)
    if a != b:
        chunks += list(
            difflib.unified_diff(a, b, fromfile=f"_pristine/{f.name}", tofile=f"DeLELSTM-main/{f.name}")
        )
(ROOT / "port.patch").write_text("".join(chunks), encoding="utf-8")
print("".join(chunks))
