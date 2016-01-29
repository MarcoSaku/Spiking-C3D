import random
orig_file='/home/marcosaviano/C3D-master/examples/c3d_train_yupenn/lst_files/test_yupenn_full_4vid.lst'
dest_file='/home/marcosaviano/C3D-master/examples/c3d_train_yupenn/lst_files/test_yupenn_full_4vid_shuffle.lst'

lines = open(orig_file).readlines()
random.shuffle(lines)
open(dest_file, 'w').writelines(lines)
print 'Done! '+dest_file+' created!'