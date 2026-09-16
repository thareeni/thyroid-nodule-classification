# test_data_loading.py
import sys
sys.path.insert(0, '.')

from utils.data_utils import get_dataset_info

DATASET_ROOT = "dataset/Main_data"

print("Testing data loading...")
print("-" * 50)

try:
    train_total, train_counts = get_dataset_info(DATASET_ROOT, "train")
    print(f"Train samples: {train_total}")
    print(f"  - Benign: {train_counts[0]}")
    print(f"  - Malignant: {train_counts[1]}")
    
    val_total, val_counts = get_dataset_info(DATASET_ROOT, "val")
    print(f"\nVal samples: {val_total}")
    print(f"  - Benign: {val_counts[0]}")
    print(f"  - Malignant: {val_counts[1]}")
    
    test_total, test_counts = get_dataset_info(DATASET_ROOT, "test")
    print(f"\nTest samples: {test_total}")
    print(f"  - Benign: {test_counts[0]}")
    print(f"  - Malignant: {test_counts[1]}")
    
    print("\n✅ Test successful! Data loading works.")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    print("\nCheck:")
    print("1. Is DATASET_ROOT path correct?")
    print("2. Does dataset/Main_data/ImageSets/Main/ exist?")
    print("3. Are there train.txt, val.txt, test.txt files?")
    print("4. Do the XML files exist in Annotations folder?")