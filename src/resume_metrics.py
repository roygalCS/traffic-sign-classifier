"""
Resume metrics for the traffic-sign classifier.

Four self-contained analyses on the already-trained ``models/base_model.pt``.
Nothing here trains, calibrates against test, or overwrites the checkpoint.

    python -m src.resume_metrics

1. Per-split accuracy (train / val / test) with the deterministic eval
   transform -> shows which split the headline ~99.67% number came from
   (it is the ``phase == "val"`` number in ``src/train.py``).
2. INT8 post-training static quantization (``torch.ao.quantization`` /
   ``torch.quantization``) -> val accuracy before/after + delta, on-disk size.
3. Out-of-distribution detection: max-softmax-probability (MSP) and energy
   scores for in-distribution (val) vs OOD samples, AUROC via
   ``sklearn.metrics.roc_auc_score``.  OOD = held-out GTSRB classes (near-OOD)
   and Gaussian noise (far-OOD).
4. Single-batch CPU latency (``eval`` + ``no_grad``, warmup then average) for
   the fp32 model vs the quantized model.
"""

import copy
import io
import time
from glob import glob
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from sklearn.metrics import roc_auc_score

from src.config import DATA_DIR, RAW_DATA_DIR, MODELS_DIR, CLASS_INDICES, RANDOM_SEED
from src.dataset import TRANSFORMS
from src.model import create_model

try:  # torch >= 1.8; "torch.quantization" is the (still-valid) legacy alias
    from torch.ao.quantization import get_default_qconfig, QConfigMapping
    from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx
except ImportError:  # pragma: no cover
    from torch.quantization import get_default_qconfig, QConfigMapping
    from torch.quantization.quantize_fx import prepare_fx, convert_fx

try:
    from torch.ao.quantization import quantize_dynamic
except ImportError:  # pragma: no cover
    from torch.quantization import quantize_dynamic

CPU = torch.device("cpu")
BATCH_SIZE = 32
CKPT = MODELS_DIR / "base_model.pt"


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def load_fp32_model(num_classes=4):
    model = create_model(num_classes)
    model.load_state_dict(torch.load(CKPT, map_location=CPU))
    return model.to(CPU).eval()


def split_loader(split, transform_key="val"):
    ds = ImageFolder(str(DATA_DIR / split), TRANSFORMS[transform_key])
    return ds, DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)


@torch.no_grad()
def accuracy(model, loader):
    model.eval()
    correct = total = 0
    for x, y in loader:
        correct += (model(x).argmax(1) == y).sum().item()
        total += y.numel()
    return correct / total


def state_dict_size_mb(model):
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.getbuffer().nbytes / 1e6


class PathListDataset(Dataset):
    """Images from an explicit list of file paths; label is a dummy 0."""

    def __init__(self, paths, transform):
        self.paths = list(paths)
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        return self.transform(img), 0


class NoiseDataset(Dataset):
    """Gaussian noise with the same shape as a normalized input tensor."""

    def __init__(self, n, seed=0):
        self.n = n
        self.seed = seed

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        g = torch.Generator().manual_seed(self.seed + i)
        return torch.randn(3, 224, 224, generator=g), 0


def heldout_gtsrb_paths(n, seed=RANDOM_SEED):
    """Sample .ppm paths from GTSRB classes NOT used for training."""
    folders = sorted(
        f for f in glob(str(RAW_DATA_DIR / "GTSRB" / "Final_Training" / "Images" / "*"))
        if Path(f).is_dir()
    )
    held = [f for i, f in enumerate(folders) if i not in set(CLASS_INDICES)]
    paths = []
    for f in held:
        paths.extend(glob(f"{f}/*.ppm"))
    rng = np.random.default_rng(seed)
    rng.shuffle(paths)
    return paths[:n]


@torch.no_grad()
def msp_energy(model, loader):
    """Return (max-softmax-prob, energy) arrays. Energy E(x) = -logsumexp(logits)."""
    model.eval()
    msp, energy = [], []
    for x, _ in loader:
        logits = model(x)
        msp.append(F.softmax(logits, dim=1).max(1).values)
        energy.append(-torch.logsumexp(logits, dim=1))
    return torch.cat(msp).numpy(), torch.cat(energy).numpy()


def auroc_id_positive(id_score, ood_score, higher_is_more_id=True):
    """AUROC with in-distribution as the positive class."""
    y = np.r_[np.ones(len(id_score)), np.zeros(len(ood_score))]
    s = np.r_[id_score, ood_score]
    return roc_auc_score(y, s if higher_is_more_id else -s)


@torch.no_grad()
def time_inference_ms(model, batch, warmup=5, iters=20):
    model.eval()
    for _ in range(warmup):
        model(batch)
    t0 = time.perf_counter()
    for _ in range(iters):
        model(batch)
    return (time.perf_counter() - t0) / iters * 1000.0


def quantize_int8(fp32_model, calib_loader, engine, calib_batches=12):
    """INT8 static post-training quantization via FX graph mode.

    FX mode is used because torchvision's plain ResNet-34 has a functional
    residual add that eager-mode static quant cannot handle without editing
    the model.  Falls back to dynamic quantization if FX tracing fails.
    """
    torch.backends.quantized.engine = engine
    m = copy.deepcopy(fp32_model).to(CPU).eval()
    try:
        qmap = QConfigMapping().set_global(get_default_qconfig(engine))
        example = next(iter(calib_loader))[0].to(CPU)
        prepared = prepare_fx(m, qmap, example)
        with torch.no_grad():
            for i, (x, _) in enumerate(calib_loader):
                prepared(x)
                if i + 1 >= calib_batches:
                    break
        return convert_fx(prepared), "static (FX graph mode, INT8)"
    except Exception as exc:  # pragma: no cover - environment dependent
        print(f"    [warn] FX static quant failed ({exc!r}); using dynamic INT8")
        return quantize_dynamic(m, {torch.nn.Linear}, dtype=torch.qint8), "dynamic (INT8, Linear only)"


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    import sys
    import warnings
    warnings.filterwarnings("ignore")  # quantization APIs are noisy with DeprecationWarnings
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    engines = torch.backends.quantized.supported_engines
    engine = "qnnpack" if "qnnpack" in engines else "fbgemm"

    print("=" * 72)
    print(f"torch {torch.__version__} | CPU threads {torch.get_num_threads()} | "
          f"quant engine '{engine}' | checkpoint {CKPT.name}")
    print("=" * 72)

    fp32 = load_fp32_model()

    # --- 1. which split is the ~99.67% figure? --------------------------------
    print("\n[1] Accuracy per split (deterministic eval transform, argmax)")
    accs = {}
    loaders = {}
    for split in ("train", "val", "test"):
        ds, loader = split_loader(split)
        loaders[split] = loader
        accs[split] = accuracy(fp32, loader)
        print(f"    {split:<5} n={len(ds):>5}  accuracy = {accs[split] * 100:.2f}%")
    match = min(accs, key=lambda s: abs(accs[s] * 100 - 99.67))
    print(f"    -> the ~99.67% headline matches the '{match}' split "
          f"({accs['val'] * 100:.2f}% on val).")
    print(f"       src/train.py only ever assigns best_accuracy inside its")
    print(f"       `phase == \"val\"` branch and prints \"Best val accuracy\", so the")
    print(f"       headline figure is VALIDATION accuracy (not train).")

    # --- 2. INT8 post-training quantization ---------------------------------
    print("\n[2] INT8 post-training quantization (calibrated on val)")
    val_loader = loaders["val"]
    acc_before = accs["val"]
    q_model, q_kind = quantize_int8(fp32, val_loader, engine)
    print(f"    method: {q_kind}")
    acc_after = accuracy(q_model, val_loader)
    size_fp32 = state_dict_size_mb(fp32)
    size_q = state_dict_size_mb(q_model)
    print(f"    val accuracy  before (fp32) = {acc_before * 100:.2f}%")
    print(f"    val accuracy  after  (int8) = {acc_after * 100:.2f}%")
    print(f"    delta                       = {(acc_after - acc_before) * 100:+.2f} pp")
    print(f"    state_dict size  fp32 = {size_fp32:.1f} MB   int8 = {size_q:.1f} MB   "
          f"({size_fp32 / size_q:.1f}x smaller)")

    # --- 3. OOD detection: MSP + energy, AUROC -----------------------------
    print("\n[3] OOD detection  (in-dist = val;  AUROC, in-dist = positive class)")
    n_id = len(val_loader.dataset)
    id_msp, id_energy = msp_energy(fp32, val_loader)

    ood_sets = {}
    heldout = heldout_gtsrb_paths(n_id)
    if heldout:
        ood_sets["held-out GTSRB classes"] = DataLoader(
            PathListDataset(heldout, TRANSFORMS["val"]), batch_size=BATCH_SIZE, num_workers=0
        )
    ood_sets["Gaussian noise"] = DataLoader(
        NoiseDataset(n_id, seed=RANDOM_SEED), batch_size=BATCH_SIZE, num_workers=0
    )

    print(f"    in-dist samples = {n_id}   "
          f"mean MSP = {id_msp.mean():.3f}   mean energy = {id_energy.mean():.2f}")
    for name, loader in ood_sets.items():
        ood_msp, ood_energy = msp_energy(fp32, loader)
        auroc_msp = auroc_id_positive(id_msp, ood_msp, higher_is_more_id=True)
        # higher energy => more OOD, so lower energy is "more in-dist"
        auroc_energy = auroc_id_positive(id_energy, ood_energy, higher_is_more_id=False)
        print(f"    vs {name:<24} n={len(loader.dataset):>4}  "
              f"mean MSP = {ood_msp.mean():.3f}  mean energy = {ood_energy.mean():.2f}")
        print(f"        AUROC (max softmax prob) = {auroc_msp:.4f}")
        print(f"        AUROC (energy score)     = {auroc_energy:.4f}")

    # --- 4. single-batch CPU latency -------------------------------------
    print(f"\n[4] Single-batch CPU latency  (batch={BATCH_SIZE}, eval + no_grad, "
          f"5 warmup + 20 timed)")
    batch = next(iter(val_loader))[0].to(CPU)
    ms_fp32 = time_inference_ms(fp32, batch)
    ms_q = time_inference_ms(q_model, batch)
    print(f"    fp32  = {ms_fp32:7.2f} ms   ({ms_fp32 / BATCH_SIZE:.2f} ms/img)")
    print(f"    int8  = {ms_q:7.2f} ms   ({ms_q / BATCH_SIZE:.2f} ms/img)")
    print(f"    speedup = {ms_fp32 / ms_q:.2f}x")

    print("\n" + "=" * 72)
    print("SUMMARY")
    print(f"  val accuracy (headline)      : {acc_before * 100:.2f}%   [split: val]")
    print(f"  INT8 val accuracy / delta    : {acc_after * 100:.2f}%  "
          f"({(acc_after - acc_before) * 100:+.2f} pp)")
    print(f"  model size  fp32 -> int8     : {size_fp32:.1f} MB -> {size_q:.1f} MB")
    print(f"  CPU latency fp32 -> int8     : {ms_fp32:.1f} ms -> {ms_q:.1f} ms "
          f"(batch {BATCH_SIZE})")
    print("=" * 72)


if __name__ == "__main__":
    main()
