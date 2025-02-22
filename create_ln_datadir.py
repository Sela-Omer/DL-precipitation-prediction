import os

OVERRIDE_PARAM_NAMES = {'day': 'date', 'latitude': 'lat', 'longitude': 'lon', 'z_geo': 'z'}

def symlink_all(dir, params, new_dir, suffix=''):
    for param in params:
        out_param = param
        if param in OVERRIDE_PARAM_NAMES:
            out_param = OVERRIDE_PARAM_NAMES[param]
        os.symlink(os.path.join(dir, param), os.path.join(new_dir, out_param) + suffix)


dir1 = '/home/mansour/ML3300-24a/omersela3/v5/v5_0'
dir2 = '/home/mansour/ML3300-24a/omersela3/v5/v5_2'
dir3 = '/home/mansour/ML3300-24a/omersela3/v5/v5_4'
dir4 = '/home/mansour/ML3300-24a/omersela3/v5/v5_6'
dir5 = '/home/mansour/ML3300-24a/omersela3/v5/v5_8'

new_dir = '/home/mansour/ML3300-24a/omersela3/tensors-v5'
os.makedirs(new_dir, exist_ok=True)

params1 = set(os.listdir(dir1))
params2 = set(os.listdir(dir2))
params3 = set(os.listdir(dir3))
params4 = set(os.listdir(dir4))
params5 = set(os.listdir(dir5))

symlink_all(dir1, params1, new_dir)
symlink_all(dir2, params2, new_dir, '-6h')
symlink_all(dir3, params2, new_dir, '-12h')
symlink_all(dir4, params2, new_dir, '-18h')
symlink_all(dir5, params2, new_dir, '-24h')
