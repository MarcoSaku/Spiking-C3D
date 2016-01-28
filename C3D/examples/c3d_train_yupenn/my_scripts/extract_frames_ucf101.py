import os
import sys
#import subprocess

main_path='/opt/Datasets/UCF-101/avi/'
frm_main_path='/opt/Datasets/UCF-101/frames/'

main_dir=os.listdir(main_path)
main_dir.sort()

for i_class in range(0,len(main_dir)):
#for i_class in range(0,2):
    print "Processing class "+str(i_class)
    class_dir_name=main_dir[i_class]
    os.makedirs(frm_main_path+class_dir_name)
    class_dir=os.listdir(main_path+class_dir_name)
    class_dir.sort()
    for i_video in range(0,len(class_dir)):
    #for i_video in range(0,1):
        video_path=main_path+class_dir_name+'/'+class_dir[i_video]
        video_frm_dest=frm_main_path+class_dir_name+'/'+class_dir[i_video][:-4]
        os.makedirs(video_frm_dest)
        #string='ffmpeg -i '+video_path+ ' '+video_frm_dest +' .jpg'
        cmd='ffmpeg -i '+video_path+ ' '+video_frm_dest+'/%06d.jpg'
        #subprocess.call(string)
        os.system(cmd)


        
# from subprocess import check_call
# extension=".jpg"
# check_call(["ffmpeg" ,"-i",video_path + extension])
    

