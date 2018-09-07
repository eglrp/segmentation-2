import glob, os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import scipy.misc
import shutil
# gaussian function
def gau(mean, var, varInv, feature):
    a = np.sqrt(2 * np.pi ** 3)
    b = np.exp(-0.5 * np.dot((feature - mean), np.dot(varInv, (feature - mean).transpose())))
    return b / a


# calculating responsibilities
def res(likelihoods):
    tempList = []
    for comp in likelihoods:
        tempList.append(comp / sum(likelihoods))
    return tempList


# calculating likelihoods
def likeli(mean, var, varInv, weights, feature):
    temp = []
    for x in v:
        temp.append(weights[x] * gau(mean[x], var[x], varInv[x], feature))
    return temp


listimage = [['/home/buiduchanh/WorkSpace/Javis/Machine-Learning-and-Pattern-Recognition/GMM-based-clustering/test_image/610006427.jpg' , 'test']]
# listimage = [['/home/buiduchanh/Downloads/画像データ/61/6162/15100040/610000119/610005146.jpg', 'shadow'],
#              ['/home/buiduchanh/Downloads/画像データ/61/6162/10500115/610000690/610032722.jpg', 'small_rust'],
#             ['/home/buiduchanh/Downloads/画像データ/61/6162/10500115/610000690/610032743.jpg',  'pipe'],
#              ['/home/buiduchanh/Downloads/画像データ/61/6162/15100090/610000143/610006427.jpg', 'large_rust'],
#              ['/home/buiduchanh/Downloads/画像データ/61/6162/15100090/610000143/610006454.jpg', 'shadow'],
#              ['/home/buiduchanh/Downloads/画像データ/61/6162/15100040/610000119/610005120.jpg', 'cable'],
#              ['/home/buiduchanh/Downloads/画像データ/61/6162/15100190/610000688/610032632.jpg', 'tree'],
#              ['/home/buiduchanh/Downloads/画像データ/61/6162/15100420/610000299/610014266.jpg', 'large_rust'],
#             ['/home/buiduchanh/Downloads/画像データ/61/6162/10500115/610000004/610000203.jpg', 'high_contrast'],
#              ['/home/buiduchanh/Downloads/画像データ/61/6162/10500180/610000033/610001218.jpg', 'medium_contrast'],
#             ['/home/buiduchanh/Downloads/画像データ/61/6162/15100390/610000263/610012394.jpg' ,'low_contrast']]

for image in listimage :

    # print(image[0])
    # shutil.copy2(image[0],'/home/buiduchanh/WorkSpace/Javis/Machine-Learning-and-Pattern-Recognition/GMM-based-clustering/test_image/')
    # exit()
    img = Image.open(image[0]).resize((480, 320), Image.ANTIALIAS)
    basenmae = os.path.splitext(os.path.basename(image[0]))[0]
    pixels = np.asarray(((img.getdata())))
    print(len(pixels))
    # total no of pixels
    N = 153600

    # initializing means, variances and weights
    feat = pixels
    v = [0, 1, 2]
    val = 100
    mean = [np.array([120, 120, 120]), np.array([12, 12, 12]), np.array([180, 180, 180])]
    # mean = [np.array([60, 60, 60]), np.array([24, 24, 24]), np.array([240, 240, 240])]
    var = [val * np.identity(3), val * np.identity(3), val * np.identity(3)]

    weights = [float(1 / 3), float(1 / 3), float(1 / 3)]


    varInv = [np.linalg.inv(var[0]), np.linalg.inv(var[1]), np.linalg.inv(var[2])]
    meanPrev = [np.array([0, 0, 0]), np.array([0, 0, 0]), np.array([0, 0, 0])]
    iteration = []
    logLikelihoods = []
    counterr = 0

    # iterating until convergence is reached
    while sum(sum(np.absolute(np.asarray(mean) - np.asarray(meanPrev)))) >= 10:
        resp = []
        likelihoods = []
        for feature in feat:
            # print("feature",feature)
            classLikelihoods = likeli(mean, var, varInv, weights, feature)  # cacultae likelihood -> return list
            # print("lilkehood",classLikelihoods)
            # print("sumlikehood",sum(classLikelihoods))
            rspblts = res(classLikelihoods)  # return list of likelihood
            # print("responsibiliyt",rspblts)
            likelihoods.append(sum(classLikelihoods))
            resp.append(rspblts)
        logLikelihoods.append(sum(np.log(likelihoods)))
        # exit()
        nK = [sum(np.asarray(resp)[:, 0:1]), sum(np.asarray(resp)[:, 1:2]), sum(np.asarray(resp)[:, 2:3])]
        weights = [float(nK[0] / N), float(nK[1] / N), float(nK[2] / N)]
        meanIterator = np.dot(np.asarray(resp).T, feat)
        meanPrev = mean
        mean = [meanIterator[0] / nK[0], meanIterator[1] / nK[1], meanIterator[2] / nK[2]]
        counterr += 1
        iteration.append(counterr)

    resp = []
    for feature in feat:
        classLikelihoods = likeli(mean, var, varInv, weights, feature)
        rspblts = res(classLikelihoods)
        resp.append(rspblts)

    result = []
    counter = 0
    segmentedImage = np.zeros((N, np.shape(img)[2]), np.uint8)

    # assigning values to pixels of different segments
    for response in resp:
        maxResp = max(response)
        respmax = response.index(maxResp)
        result.append(respmax)
        segmentedImage[counter] = mean[respmax]
        counter = counter + 1

    print('shape',type(segmentedImage),segmentedImage.shape)
    blue0 = segmentedImage[:, 0]
    green0 = segmentedImage[:, 1]
    red0 = segmentedImage[:, 2]

    # rgb values of all the pixels segmented according to gaussian models
    blue = np.reshape(blue0.flatten(), (np.shape(img)[0], np.shape(img)[1]))
    green = np.reshape(green0.flatten(), (np.shape(img)[0], np.shape(img)[1]))
    red = np.reshape(red0.flatten(), (np.shape(img)[0], np.shape(img)[1]))

    recns = np.zeros((320, 480, 3))

    for i in range(320):
        for j in range(480):
            recns[i][j] = np.array([blue[i][j], green[i][j], red[i][j]])
    scipy.misc.imsave('test_2208/{}_{}_10_{}.png'.format(basenmae, image[1],counter), recns)

    # plotting segmented image
    plt.imshow(recns)
    plt.show()

    # plotting the graph of likelihood versus number of iterations
    plt.plot(iteration, logLikelihoods)
    plt.title('likelihood convergence')
    plt.ylabel('Likelihood')
    plt.xlabel('Iteration number')

    # Show the figure.
    plt.show()

