import os


dir = '../before_6h'

new_dir = '../before_6h_no_suffix'

# create directories recursively to match directory structure of dir
def create_datadir_tree(dir, new_dir):
    for root, _, _ in os.walk(dir):
        new_root = root.replace(dir, new_dir)
        os.makedirs(new_root, exist_ok=True)

create_datadir_tree(dir, new_dir)



