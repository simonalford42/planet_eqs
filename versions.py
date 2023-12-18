import glob

def is_valid(version, seed=None):
    files = glob.glob('results/steps=*/')
    # go from 'steps=500_megno=0_angles=1_power=0_hidden=40_latent=20_nommr=1_nonan=1_noeplusminus=1_v26498_0'/ to 26498
    versions_present = [int(f.split('_')[-2][1:]) for f in files]
    if seed is None:
        return version not in versions_present
    else:
        seeds = [int(f.split('_')[-1][:-1]) for f in files]
        return (version, seed) not in zip(versions_present, seeds)

def next_version():
    files = glob.glob('results/steps=*/')
    # go from 'steps=500_megno=0_angles=1_power=0_hidden=40_latent=20_nommr=1_nonan=1_noeplusminus=1_v26498_0'/ to 26498
    versions_present = [int(f.split('_')[-2][1:]) for f in files]
    for i in range(100000):
        if i not in versions_present:
            return i

    raise Exception("No valid version found")

    print(next_version())
