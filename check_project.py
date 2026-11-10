"""
Project check: confirms the old files are archived, the files we keep are present, and the
SEN12MS-CR step can start. Read-only: it changes nothing.

Run from the project root (D:\\cloud_removal_project):
  python check_project.py              # includes a short Hugging Face connection test
  python check_project.py --skip_hf    # skip the network test
"""
import argparse
import importlib
import py_compile
import shutil
from pathlib import Path

ROOT = Path("cloud_removal_project")
SRC = ROOT / "src"
RES = ROOT / "results"
DATA = ROOT / "data"
MODELS = ROOT / "models"
ARCH = Path("archive_old")

results = []


def check(name, ok, detail=""):
    results.append((ok, name, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))


def exists_all(paths):
    missing = [str(p) for p in paths if not Path(p).exists()]
    return missing


# 1. Files that must have been archived (not in their original place)
moved_candidates = [
    SRC / "eval_unet_downstream.py",
    ROOT / "PROJECT_REPORT_v2_DRAFT.md", ROOT / "PROJECT_REPORT_v3_DRAFT.md",
    ROOT / "PROJECT_REPORT_v4_DRAFT.md", ROOT / "PROJECT_REPORT_v5_DRAFT.md",
    RES / "classifier_ablation.csv", RES / "opacity_sweep.csv", RES / "unet_downstream.csv",
]
still_there = [str(p) for p in moved_candidates if p.exists()]
check("superseded report drafts and results moved out",
      not still_there, f"still in place: {still_there}" if still_there else "")

leftover_bak = list(SRC.glob("*.bak")) if SRC.exists() else []
check("no .bak files left in src/", not leftover_bak,
      f"found {[p.name for p in leftover_bak]}" if leftover_bak else "")

old_sen = [p for p in list(DATA.glob("sen12mscr_*.npz")) + list(DATA.glob("sen12mscr_*.csv")) + [DATA / "test_pixels.npz"]
           if p.exists()] if DATA.exists() else []
check("old SEN12MS-CR data moved out of data/ (near-black images)", not old_sen,
      f"still present: {[p.name for p in old_sen]}" if old_sen else "")

check("archive_old folder exists", ARCH.exists(),
      "" if ARCH.exists() else "run the move commands first, or skip if you deleted instead")

# 2. Files that must be kept
keep = [
    RES / "threeway_seed0.csv", RES / "threeway_seed1.csv", RES / "threeway_seed2.csv",
    RES / "metric_validity_summary_seed0.csv", RES / "metric_validity_tile_summary_seed0.csv",
    RES / "within_method_summary_seed0.csv",
    RES / "unet_transfer_seed0.csv", RES / "unet_transfer_seed0_imagenet_lr0.0001.csv",
    ROOT / "PROJECT_REPORT.md", Path("ANALYSIS_PLAN.md"),
    SRC / "threeway_comparison.py", SRC / "ablate_classifier_finetune.py", SRC / "opacity_sweep.py",
    SRC / "dehaze_dcp.py", SRC / "evaluate_all.py", SRC / "audit_failures.py",
    SRC / "metric_validity_full.py", SRC / "summarize_metric_validity_tiles.py",
    SRC / "within_method_validity.py", SRC / "unet_transfer.py",
    MODELS / "unet_inpaint.pth", MODELS / "resnet18_clean.pth", MODELS / "resnet18_mask_aware.pth",
    MODELS / "resnet18_fill_aware.pth",
    DATA / "splits" / "train.csv", DATA / "splits" / "val.csv", DATA / "splits" / "test.csv",
]
missing = exists_all(keep)
check("files needed for the report are present", not missing,
      f"missing: {missing}" if missing else f"{len(keep)} files checked")

# 3. Scripts compile
scripts = sorted(SRC.glob("*.py")) if SRC.exists() else []
bad = []
for p in scripts:
    try:
        py_compile.compile(str(p), doraise=True, cfile=str(p) + "c")
    except py_compile.PyCompileError as e:
        bad.append(p.name)
check("all scripts in src/ compile", not bad, f"errors in {bad}" if bad else f"{len(scripts)} scripts")
for p in SRC.glob("*.pyc"):
    p.unlink(missing_ok=True)
shutil.rmtree(SRC / "__pycache__", ignore_errors=True)

# 4. SEN12MS-CR readiness
builder = SRC / "build_sen12mscr_subset.py"
if builder.exists():
    text = builder.read_text(encoding="utf-8")
    fixed = "SCALE = 0.3" in text
    check("SEN12MS-CR scale bug fixed (SCALE = 0.3)", fixed,
          "" if fixed else "change SCALE = 3000.0 to SCALE = 0.3 before streaming")
    has_cap = "--max_per_scene" in text
    check("scene cap option present in builder", has_cap,
          "" if has_cap else "builder is an old version; re-apply the patch")
else:
    check("SEN12MS-CR builder present", False, "src/build_sen12mscr_subset.py missing")

lc = sorted(Path(DATA / "SEN12MS_lc").glob("*.tar.gz")) if (DATA / "SEN12MS_lc").exists() else []
check("4 land-cover archives present", len(lc) == 4, f"found {len(lc)}")

for mod in ["numpy", "pandas", "torch", "cv2", "datasets", "tifffile", "lpips", "skimage", "scipy"]:
    try:
        importlib.import_module(mod)
        ok, detail = True, ""
    except Exception as e:
        ok, detail = False, f"{type(e).__name__}: {e}"
    check(f"python package: {mod}", ok, detail)

try:
    free_gb = shutil.disk_usage(".").free / 1e9
    check("at least 15 GB free disk space", free_gb >= 15, f"{free_gb:.1f} GB free")
except Exception as e:
    check("disk space readable", False, str(e))

# 5. Hugging Face streaming (one sample, nothing downloaded in full)
parser = argparse.ArgumentParser()
parser.add_argument("--skip_hf", action="store_true")
args, _ = parser.parse_known_args()
if args.skip_hf:
    print("[SKIP] Hugging Face test (--skip_hf)")
else:
    try:
        from datasets import load_dataset
        ds = load_dataset("Hermanni/sen12mscr", split="train", streaming=True)
        first = next(iter(ds))
        check("Hugging Face streaming works", True, f"first row season={first['season']}")
    except Exception as e:
        check("Hugging Face streaming works", False, f"{type(e).__name__}: {str(e)[:120]}")

failed = [name for ok, name, _ in results if not ok]
print("\n" + ("ALL CHECKS PASSED" if not failed else f"{len(failed)} CHECK(S) FAILED:"))
for n in failed:
    print("  -", n)
