import hashlib


def split_dataset(index_files, val_ratio=0.2):
    train_files = []
    val_files = []

    for file in index_files:
        hash_value = int(hashlib.md5(file.encode()).hexdigest(), 16)
        if hash_value % 100 < val_ratio * 100:
            val_files.append(file)
        else:
            train_files.append(file)

    return train_files, val_files
