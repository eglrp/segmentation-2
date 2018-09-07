import torch
from torch.utils import data
import torch.optim as optim
from torch.autograd import Variable
from transform import Colorize
from torchvision.transforms import ToPILImage, Compose, ToTensor, CenterCrop
from transform import Scale
# from resnet import FCN
from upsample import FCN
# from gcn import FCN
from datasets import VOCTestSet
from PIL import Image
import numpy as np
from tqdm import tqdm
import glob
import os
from collections import OrderedDict
'''
color map
0=background, 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle # 6=bus, 7=car, 8=cat, 9=chair, 10=cow, 11=diningtable,
12=dog, 13=horse, 14=motorbike, 15=person # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
'''
palette = [0, 0, 0, 128, 0, 0, 0, 128, 0, 128, 128, 0, 0, 0, 128, 128, 0, 128, 0, 128, 128,
           128, 128, 128, 64, 0, 0, 192, 0, 0, 64, 128, 0, 192, 128, 0, 64, 0, 128, 192, 0, 128,
           64, 128, 128, 192, 128, 128, 0, 64, 0, 128, 64, 0, 0, 192, 0, 128, 192, 0, 0, 64, 128]

zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)


def colorize_mask(mask):
    # mask: numpy array of the mask
    # print(palette)
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)

    return new_mask

# label_transform = Compose([Scale((256, 256), Image.BILINEAR), ToTensor()])
# batch_size = 1
# dst = VOCTestSet("./data", transform=label_transform)

# testloader = data.DataLoader(dst, batch_size=batch_size,
#                              num_workers=8)


model = torch.nn.DataParallel(FCN(22), device_ids=[0])
model.cuda()
model.load_state_dict(torch.load("./pth/fcn-deconv-35.pth"))
model.eval()

# state_dict = torch.load("./pth/fcn-deconv-40_v2_resnet50.pth")
# new_state_dict = OrderedDict()
# for k, v in state_dict.items():
#     print(type(state_dict.items()))
#     print(k,v)
    # name = k[7:] # remove module.
    # new_state_dict[name] = v

# model.load_state_dict(new_state_dict)

print("load model success")
# 10 13 48 86 101
TestDir = '/media/buiduchanh/Work/Workspace/Javis/Workspace/pytorch-segmentation/test'
# ResultDir = '/media/buiduchanh/STUDY/Javis/Workspace/pytorch-segmentation/result'
ResultDir = '/media/buiduchanh/Work/Workspace/Javis/Workspace/pytorch-segmentation/result_resnet101_v3'
imagelist = sorted(glob.glob('{}/*'.format(TestDir)))

for img in imagelist:
    print(img)
    basename = os.path.splitext(os.path.basename(img))[0]
    img = Image.open(img).convert("RGB")
    original_size = img.size
    # img.save("original.png")
    img = img.resize((256, 256), Image.BILINEAR)
    img = ToTensor()(img)
    img = Variable(img).unsqueeze(0)
    outputs = model(img)
    # 22 256 256

    # for i, output in enumerate(outputs):
    output = outputs[0]

    prediction = output.data.max(1)[1].squeeze_(1).squeeze_(0).cpu().numpy()
    # print(prediction.numpy())
    prediction = colorize_mask(prediction)

    print(prediction.size)
    # prediction.show()

    # output = np.transpose(prediction.numpy(), (1, 2, 0))
    # img = Image.fromarray(output, "RGB")

    # output = np.transpose(img, (1, 2, 0))
    newname = basename + '_result' + '.png'
    DesDir = os.path.join(ResultDir,newname)
    prediction.save(DesDir)


