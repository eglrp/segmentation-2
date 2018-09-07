import cv2
import numpy as np
import argparse
import time
import find_boundary as bz
import os
import glob
def kmeans_seg(img, K):
    size = img.shape
    height = size[0]
    width = size[1]
    channel = size[2]

    imgPos = np.zeros(shape=(height, width, channel + 2))

    for i in range(height):
        for j in range(width):
            imgPos[i][j] = np.append(img[i][j], [i, j])

    Z = imgPos.reshape((-1, 5))

    # convert to np.float32
    Z = np.float32(Z)

    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 1000, 0.1)
    # criteria = (10, 100, 1.0)

    ret, label, center = cv2.kmeans(Z, K, None, criteria, 20, cv2.KMEANS_RANDOM_CENTERS)

    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten(), 0:3]
    res2 = res.reshape((img.shape))

    mask = bz.find_bound(label, size)

    # mask.dtype = "uint8"

    # cv2.imshow('res2', res2)
    # cv2.imshow('mask',mask)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return res2, mask


def seg(input_path,output_path, output_path2,K):
    start = time.time()
    img = cv2.imread(input_path)
    img_seg, img_binary = kmeans_seg(img, K)
    cv2.imwrite(output_path, img_seg)
    cv2.imwrite(output_path2, img_binary)
    end = time.time()
    print("Spend time: " + str(end - start))

def main():
    # Dir = '/home/tone/workspace/jupyter_lab/lab/Hanh/data/images/image_test'
    Desresult = '/home/buiduchanh/WorkSpace/Javis/Kmeans/result_Kmeans'
    Desresult_bound = '/home/buiduchanh/WorkSpace/Javis/Kmeans/result_bound'
    # imagelist = sorted(glob.glob('{}/*'.format(Dir)))
    listimage = [['/home/buiduchanh/Downloads/画像データ/61/6162/15100040/610000119/610005146.jpg', 'shadow'],
                 ['/home/buiduchanh/Downloads/画像データ/61/6162/10500115/610000690/610032722.jpg', 'small_rust'],
                 ['/home/buiduchanh/Downloads/画像データ/61/6162/10500115/610000690/610032743.jpg', 'pipe'],
                 ['/home/buiduchanh/Downloads/画像データ/61/6162/15100090/610000143/610006427.jpg', 'large_rust'],
                 ['/home/buiduchanh/Downloads/画像データ/61/6162/15100090/610000143/610006454.jpg', 'shadow'],
                 ['/home/buiduchanh/Downloads/画像データ/61/6162/15100040/610000119/610005120.jpg', 'cable'],
                 ['/home/buiduchanh/Downloads/画像データ/61/6162/15100190/610000688/610032632.jpg', 'tree'],
                 ['/home/buiduchanh/Downloads/画像データ/61/6162/15100420/610000299/610014266.jpg', 'large_rust'],
                 ['/home/buiduchanh/Downloads/画像データ/61/6162/10500115/610000004/610000203.jpg', 'high_contrast'],
                 ['/home/buiduchanh/Downloads/画像データ/61/6162/10500180/610000033/610001218.jpg', 'medium_contrast'],
                 ['/home/buiduchanh/Downloads/画像データ/61/6162/15100390/610000263/610012394.jpg', 'low_contrast']]

    for input_path in listimage:
        print(input_path[0])
        basename = os.path.splitext(os.path.basename(input_path[0]))[0]
        # newname = basename + '_result' + '.jpg'
        newname = '{}_{}.jpg'.format(basename,input_path[1])
        output_path = os.path.join(Desresult,newname)
        output_path2 = os.path.join(Desresult_bound,newname)
        numberK = 3
        seg(input_path[0],output_path,output_path2,numberK)

if __name__ == "__main__":
    main()