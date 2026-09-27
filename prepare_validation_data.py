import os
import sys

def main():
    val_ids = set()
    with open('experiments/val_s1_ids.txt', 'r', encoding='utf-8') as f:
        for line in f:
            sid = line.strip()
            if sid:
                val_ids.add(sid)
            if len(val_ids) >= 20000:
                break
    
    os.makedirs('temp_val_dir', exist_ok=True)
    
    # Write test_source1.tsv
    with open('data/student_resource/dataset/train/train_source1.tsv', 'r', encoding='utf-8') as fin, \
         open('temp_val_dir/test_source1.tsv', 'w', encoding='utf-8') as fout:
        header = fin.readline()
        fout.write(header)
        for line in fin:
            parts = line.split('\t')
            if parts[0] in val_ids:
                fout.write(line)
                
    print(f"test_source1.tsv created with {len(val_ids)} target validation IDs.")

if __name__ == '__main__':
    main()
