import os, sys, random

root, cloudy_dir, out_prefix = sys.argv[1], sys.argv[2], sys.argv[3]
ids = sorted(os.path.splitext(f)[0] for f in os.listdir(os.path.join(root, cloudy_dir)))
random.Random(42).shuffle(ids)
n = len(ids)
a, b = int(0.7 * n), int(0.8 * n)
parts = {"train": ids[:a], "val": ids[a:b], "test": ids[b:]}
for k, v in parts.items():
    with open(f"{out_prefix}_{k}.txt", "w") as f:
        f.write("\n".join(v) + "\n")
    print(k, len(v))
