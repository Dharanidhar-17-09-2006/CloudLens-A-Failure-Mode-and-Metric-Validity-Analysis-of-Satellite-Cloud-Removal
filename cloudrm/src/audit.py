import os, sys, collections
from PIL import Image

root = sys.argv[1]
for d, _, files in os.walk(root):
    if not files:
        continue
    ext = collections.Counter(os.path.splitext(f)[1].lower() for f in files)
    sizes = collections.Counter()
    for f in sorted(files)[:50]:
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff")):
            with Image.open(os.path.join(d, f)) as im:
                sizes[(im.size, im.mode)] += 1
    print(d, len(files), dict(ext), dict(sizes))
