import os

def count_files(folder_path):
    count = 0
    for _, _, files in os.walk(folder_path):
        for f in files:
            if f.endswith('.jpg') or f.endswith('.png'):
                count += 1
    return count

base_path = os.path.join('datasets', 'lung_data')

train_c = count_files(os.path.join(base_path, 'train', 'images'))
val_c = count_files(os.path.join(base_path, 'val', 'images'))
test_c = count_files(os.path.join(base_path, 'test', 'images'))

total = train_c + val_c + test_c

print(f"--- Dataset Audit ---")
print(f"Total Images: {total}")
print(f"Train: {train_c} ({train_c/total*100:.1f}%)")
print(f"Val:   {val_c}   ({val_c/total*100:.1f}%)")
print(f"Test:  {test_c}  ({test_c/total*100:.1f}%)")