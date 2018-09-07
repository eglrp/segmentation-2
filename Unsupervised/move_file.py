import os
import glob
import shutil
# /home/joyvan/work/Hanh
Dir = '/home/buiduchanh/Downloads/画像データ/61'
Des = '/media/buiduchanh/Work/Workspace/Javis/data_javis/image_test'
for f1 in os.listdir(Dir):
    path = os.path.join(Dir,f1)
    for f2 in os.listdir(path):
        path2 = os.path.join(path,f2)
        for f3 in os.listdir(path2):
            path3 = os.path.join(path2,f3)
            listimage = sorted(glob.glob('{}/*.jpg'.format(path3)))
            for image in listimage:
                print(image)

                shutil.copy2(image,Des)
