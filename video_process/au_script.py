import os
all_samples=os.listdir()
output_dir=""
for video in all_samples:
    if not os.path.isdir(output_dir+video):
        os.mkdir(output_dir+video)
    cmd='/home/ysy/gln/OpenFace/build/bin/FeatureExtration -f "'+''+video +'" '+output_dir+video+'/'
    os.system(cmd)

