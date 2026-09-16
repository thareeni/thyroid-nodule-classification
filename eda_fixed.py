# ============================================================
# FINAL EDA (AUTO FIXED PATH + NO EMPTY GRAPH ISSUE)
# ============================================================

import os, zipfile, random, warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from collections import defaultdict
import hashlib

# ================= CONFIG =================
zip_filename = "dataset.zip"
extract_path = "dataset"

# ================= EXTRACT =================
if not os.path.exists(extract_path):
    with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

print("✅ Dataset Extracted")

# ================= AUTO FIND CORRECT PATH =================
def find_real_path(path):
    while True:
        items = os.listdir(path)
        if len(items) == 1 and os.path.isdir(os.path.join(path, items[0])):
            path = os.path.join(path, items[0])
        else:
            return path

real_path = find_real_path(extract_path)
print("📂 Using dataset path:", real_path)

# ================= LOAD DATA =================
IMAGE_EXT = {".jpg",".jpeg",".png",".JPG",".PNG"}

def load_data(root):
    data = defaultdict(list)
    for cls in os.listdir(root):
        cls_path = os.path.join(root, cls)
        if os.path.isdir(cls_path):
            for f in os.listdir(cls_path):
                if os.path.splitext(f)[1] in IMAGE_EXT:
                    data[cls].append(os.path.join(cls_path, f))
    return dict(data)

class_paths = load_data(real_path)

# ================= DEBUG =================
print("\n🔍 DEBUG INFO")
print("Classes:", class_paths.keys())
print("Counts:", {k: len(v) for k,v in class_paths.items()})

if len(class_paths) == 0:
    print("❌ No images found — check dataset")
    exit()

# ================= CLASS DISTRIBUTION =================
counts = {k:len(v) for k,v in class_paths.items()}
total = sum(counts.values())

plt.figure()
plt.pie(counts.values(), labels=counts.keys(), autopct='%1.1f%%')
plt.title("Class Distribution")
plt.show()

# ================= SIZE ANALYSIS =================
widths, heights = [], []

for paths in class_paths.values():
    for p in paths[:100]:
        img = Image.open(p)
        w,h = img.size
        widths.append(w)
        heights.append(h)

plt.figure()
sns.histplot(widths, kde=True)
plt.title("Width Distribution")
plt.show()

plt.figure()
sns.histplot(heights, kde=True)
plt.title("Height Distribution")
plt.show()

# ================= CHANNEL ANALYSIS =================
gray, rgb = 0,0

for paths in class_paths.values():
    for p in paths[:100]:
        ch = len(Image.open(p).getbands())
        if ch==1: gray+=1
        elif ch==3: rgb+=1

print(f"\n🎨 Grayscale: {gray}, RGB: {rgb}")

# ================= CORRUPTED CHECK =================
bad=[]
for cls,paths in class_paths.items():
    for p in paths:
        try:
            Image.open(p).verify()
        except:
            bad.append(p)

print(f"⚠️ Corrupted images: {len(bad)}")

# ================= DUPLICATE CHECK =================
hashes=set()
duplicates=0

for paths in class_paths.values():
    for p in paths:
        with open(p,'rb') as f:
            h=hashlib.md5(f.read()).hexdigest()
        if h in hashes:
            duplicates+=1
        hashes.add(h)

print(f"🧬 Duplicate images: {duplicates}")

# ================= FINAL =================
print("\n✅ EDA COMPLETED SUCCESSFULLY")