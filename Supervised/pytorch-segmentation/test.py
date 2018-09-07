import cv2 as cv
import numpy as np
import os
import glob
# path = '/media/buiduchanh/STUDY/Javis/data_javis/data/VOC2012test/JPEGImages/2008_000500.jpg'
Dir  = '/home/buiduchanh/Desktop/data_v3/annotations'
tmp = []
imagelist = sorted(glob.glob('{}/*'.format(Dir)))
for path in imagelist:
    basename = os.path.splitext(os.pathd.basename(path))[0]
    newname = basename + '.png'
    newpath  = os.path.join(Dir,newname)
    os.rename(path,newpath)
    # tmp.append(basename)
# print(tmp)
# imgFile = cv.imread(path)

#
# cv.imshow('dst_rt', imgFile)
# cv.imwrite('/media/buiduchanh/STUDY/Javis/Workspace/pytorch-segmentation/reuslt/{}_result.png'.format(basename))
# cv.waitKey(0)
# cv.destroyAllWindows()
# #
# a = np.arange(30).reshape(3,10)
# print("oorigin",a)
# for i in range (0, a.shape[0]):
#     for j in range (0, a.shape[1]):
#         if a[i][j] > 10 :
#             a[i][j] = 5
# print("after", a)
