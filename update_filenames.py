import glob
import subprocess

# for each file in results/, remove the 'steps=300000_megno=0_angles=1_power=0_hidden=40_latent=20_nommr=1_nonan=1_noeplusminus=1_' from the front
# note that sometimes the steps, hidden, or latent will be different

filenames = glob.glob('steps=*')
for f in filenames:
    new_f = f[f.index('_v')+2:]
    # mv f new_f
    subprocess.run(['mv', f, new_f])

