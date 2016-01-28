import os
import sys

main_path='/opt/Datasets/YUPENN_dynamic_scenes_data_set/frames/'
path_lst_file_train='/home/marcosaviano/C3D-master/examples/c3d_train_yupenn/lst_files/train_yupenn_full_26vid.lst'
path_lst_file_test='/home/marcosaviano/C3D-master/examples/c3d_train_yupenn/lst_files/test_yupenn_full_4vid.lst'

file = open(path_lst_file_train, 'w')
file_test = open(path_lst_file_test, 'w')

main_dir=os.listdir(main_path)
main_dir.sort()
#for i_class in range(0,):
for i_class in range(0,len(main_dir)):
    print "Processing class "+str(i_class)
    class_dir_name=main_dir[i_class]
    class_dir=os.listdir(main_path+class_dir_name)
    class_dir.sort()
    #for i_video in range(0,len(class_dir)):
    for i_video in range(0,26):
        video_path=main_path+class_dir_name+'/'+class_dir[i_video]
        num_frames=len(os.listdir(video_path))
        kk=0
        while ((16*(kk+1))+1)<num_frames:
            file.write(video_path+'/ '+str((16*kk)+1)+' '+str(i_class)+'\n')
            kk +=1
    for i_video in range(26,30):
        video_path=main_path+class_dir_name+'/'+class_dir[i_video]
        num_frames=len(os.listdir(video_path))
        kk=0
        while ((16*(kk+1))+1)<num_frames:
            file_test.write(video_path+'/ '+str((16*kk)+1)+' '+str(i_class)+'\n')
            kk +=1
file.close()
file_test.close()
print 'Done!'

import random
orig_file='/home/marcosaviano/C3D-master/examples/c3d_train_yupenn/lst_files/test_yupenn_full_4vid.lst'
dest_file='/home/marcosaviano/C3D-master/examples/c3d_train_yupenn/lst_files/test_yupenn_full_4vid_shuffle.lst'

lines = open(orig_file).readlines()
random.shuffle(lines)
open(dest_file, 'w').writelines(lines)
print 'Done! '+dest_file+' created!'