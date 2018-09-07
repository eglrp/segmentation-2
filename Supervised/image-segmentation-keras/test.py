import os
import glob

Dir = '/media/buiduchanh/STUDY/Javis/Workspace/pytorch-segmentation/result/'
listfile = sorted(glob.glob('{}*'.format(Dir)))
for file in listfile:
    base_name = os.path.splitext(os.path.basename(file))[0]
    typefile = os.path.splitext(os.path.basename(file))[1]
    print(base_name)
    # exit()
    new_name = base_name + "_result" + typefile
    # print(new_name)
    # exit()
    newpath = os.path.join(Dir,new_name)
    os.rename(file,newpath)