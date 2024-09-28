import os

dir1 = '/home/mansour/ML3300-24a/omersela3/before_6h'
dir2 = '/home/mansour/ML3300-24a/omersela3/before_6h_no_suffix'

# list all files recursively in dir
def list_files(dir):
    files = []
    for root, _, filenames in os.walk(dir):
        for filename in filenames:
            files.append(os.path.join(root, filename))
    return files

file_pairs = []

for f in list_files(dir1):
    if f.endswith('_before.npy'):
        old_f = f
        new_f = f.removesuffix('_before.npy') + '.npy'
        new_f = new_f.replace(dir1, dir2)
        file_pairs.append((old_f, new_f))

# create link between old and new files if no such link exists
for old_f, new_f in file_pairs:
    if not os.path.exists(new_f):
        os.symlink(old_f, new_f)