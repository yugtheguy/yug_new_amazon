import os
import sys

def find_dataset():
    drives = [f"{d}:\\" for d in "CDEFGHIJKLMNOPQRSTUVWXYZ" if os.path.exists(f"{d}:\\")]
    target = "train_source1.tsv"
    found = []
    for drive in drives:
        for root, dirs, files in os.walk(drive):
            # Prune slow/irrelevant dirs
            lower_root = root.lower()
            if any(x in lower_root for x in ['\\windows', '\\program files', '\\$recycle.bin', '\\appdata']):
                dirs.clear()
                continue
                
            if target in files:
                full_path = os.path.join(root, target)
                print(f"Found: {full_path}")
                found.append(root)
    return found

found_dirs = find_dataset()
if not found_dirs:
    print("Dataset not found!")
    sys.exit(1)

dataset_dir = found_dirs[0]
print(f"\nAnalyzing dataset at: {dataset_dir}")
expected_files = ['train_source1.tsv', 'train_source2.tsv', 'train_source3.tsv', 
                  'train_ground_truth.tsv', 'test_source1.tsv', 'test_source2.tsv', 'test_source3.tsv']

for file in expected_files:
    file_path = os.path.join(dataset_dir, file)
    if os.path.exists(file_path):
        size = os.path.getsize(file_path)
        
        # count rows and get header
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            header = f.readline().strip()
            row_count = 0
            for _ in f:
                row_count += 1
                
        # detect delimiter
        delimiter = "\\t" if "\\t" in repr(header) else ("comma" if "," in header else "unknown")
        
        print(f"File: {file} | Size: {size} bytes | Rows: {row_count} | Delimiter: {delimiter}")
        print(f"Columns: {header}")
    else:
        print(f"File: {file} | MISSING")
