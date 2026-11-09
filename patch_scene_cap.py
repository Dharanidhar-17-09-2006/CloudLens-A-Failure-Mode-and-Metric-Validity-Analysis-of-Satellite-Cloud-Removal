"""
One-time patch for build_sen12mscr_subset.py: adds --max_per_scene and --buffer.
Run from the project root:
  python patch_scene_cap.py cloud_removal_project/src/build_sen12mscr_subset.py
A backup is written next to the file as build_sen12mscr_subset.py.bak.
"""
import shutil
import sys
from pathlib import Path

path = Path(sys.argv[1])
src = path.read_text(encoding="utf-8")

if "max_per_scene" in src:
    print("Already patched. Nothing changed.")
    sys.exit(0)

edits = [
    (
        "def stream(per_season, max_stream, thr, out_path):",
        "def stream(per_season, max_stream, thr, out_path, max_per_scene=20, buffer=300):",
    ),
    (
        "    seen = set(keys)\n",
        "    seen = set(keys)\n"
        "\n"
        "    # Count pairs per (season, scene) so no scene dominates the subset.\n"
        "    scene_count = {}\n"
        "    for s_, sc, _ in keys:\n"
        "        scene_count[(s_, sc)] = scene_count.get((s_, sc), 0) + 1\n",
    ),
    (
        '            key = (season, str(sample["scene"]), str(sample["patch"]))\n',
        '            key = (season, str(sample["scene"]), str(sample["patch"]))\n'
        "\n"
        "            if scene_count.get((season, key[1]), 0) >= max_per_scene:\n"
        "                continue\n",
    ),
    (
        "            seen.add(key)\n",
        "            seen.add(key)\n"
        "            scene_count[(season, key[1])] = scene_count.get((season, key[1]), 0) + 1\n",
    ),
    (
        ".shuffle(seed=0, buffer_size=50)",
        ".shuffle(seed=0, buffer_size=buffer)",
    ),
    (
        '    ap.add_argument("--purity", type=float, default=0.7)\n',
        '    ap.add_argument("--purity", type=float, default=0.7)\n'
        '    ap.add_argument("--max_per_scene", type=int, default=20)\n'
        '    ap.add_argument("--buffer", type=int, default=300)\n',
    ),
    (
        "        stream(args.per_season, args.max_stream, args.thr, args.pairs)\n",
        "        stream(args.per_season, args.max_stream, args.thr, args.pairs,\n"
        "               args.max_per_scene, args.buffer)\n",
    ),
]

for old, new in edits:
    count = src.count(old)
    if count != 1:
        sys.exit(f"Patch aborted: expected exactly 1 match, found {count} for:\n{old!r}")
    src = src.replace(old, new)

shutil.copy(path, path.with_suffix(path.suffix + ".bak"))
path.write_text(src, encoding="utf-8")
print("Patched", path, "| backup:", path.name + ".bak")
