import os

def symlink_all(dir, params, new_dir, suffix=''):
    for param in params:
        os.symlink(os.path.join(dir, param), os.path.join(new_dir, param) + suffix)

def symlink_all_with_priority(dir1, dir2, params1, params2, new_dir):
    all_params = params1.union(params2)

    for param in all_params:
        if param in params2:
            # create symlink to high-res
            os.symlink(os.path.join(dir2, param), os.path.join(new_dir, param) + '_0.25')
        else:
            # create symlink to low res
            os.symlink(os.path.join(dir1, param), os.path.join(new_dir, param))

dir1 = '/home/mansour/ML3300-24a/omersela3/fixed_tensors-v2/fixed_tensors-v2'
dir2 = '/home/mansour/ML3300-24a/omersela3/before_6h_no_suffix'

new_dir = '/home/mansour/ML3300-24a/omersela3/tensors_0h_6h-v2'
os.makedirs(new_dir, exist_ok=True)

params1 = set(os.listdir(dir1))
params2 = set(os.listdir(dir2))

symlink_all(dir1, params1, new_dir)
symlink_all(dir2, params2, new_dir, '-6h')




