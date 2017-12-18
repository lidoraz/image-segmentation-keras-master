import argparse
import Models, LoadBatches
from keras.models import load_model
import glob
import cv2
import numpy as np
import random

# from keras.utils import plot_model
# import pydot
#







parser = argparse.ArgumentParser()
parser.add_argument("--save_weights_path", type=str)
parser.add_argument("--epoch_number", type=int, default=5)
parser.add_argument("--test_images", type=str, default="")
parser.add_argument("--output_path", type=str, default="")
parser.add_argument("--input_height", type=int, default=224)
parser.add_argument("--input_width", type=int, default=224)
parser.add_argument("--model_name", type=str, default="")
parser.add_argument("--n_classes", type=int)

args = parser.parse_args()

n_classes = args.n_classes
model_name = args.model_name
images_path = args.test_images
input_width = args.input_width
input_height = args.input_height
epoch_number = args.epoch_number
#
modelFns = {'vgg_segnet': Models.VGGSegnet.VGGSegnet, 'vgg_unet': Models.VGGUnet.VGGUnet,
            'vgg_unet2': Models.VGGUnet.VGGUnet2, 'fcn8': Models.FCN8.FCN8, 'fcn32': Models.FCN32.FCN32}
modelFN = modelFns[model_name]

m = modelFN(n_classes, input_height=input_height, input_width=input_width)
m.load_weights(args.save_weights_path + "." + str(epoch_number))
m.compile(loss='categorical_crossentropy',
          optimizer='adadelta',
          metrics=['accuracy'])

# plot_model(m, to_file='model.png')

output_height = m.outputHeight
output_width = m.outputWidth

images = glob.glob(images_path + "*.jpg") + glob.glob(images_path + "*.png") + glob.glob(images_path + "*.jpeg")
images = [image.replace('\\', '/') for image in images]
images.sort()

colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(n_classes)]

for imgName in images:
    outName = imgName.replace(images_path, args.output_path)
    M, N = 300, 300
    im = cv2.imread(imgName, cv2.IMREAD_COLOR)
    tiles = LoadBatches.splitImg(im)


    ## colorize tiles
    # for tile in tiles:
    #     blue, green, red = random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
    #     ones = np.ones_like(tile[:, :, 1])
    #     tile[:, :, 0] += ones * blue
    #     tile[:, :, 1] += ones * green
    #     tile[:, :, 2] += ones * red

    i=0
    for tile in tiles:
        print( 'tile no:',i,'/',len(tiles))
        #tile = np.rollaxis(tile, 2, 0) #roll axis so channels are first
        X = LoadBatches.getImageArr(tile, args.input_width, args.input_height)
        pr = m.predict(np.array([X]))[0]
        print('done predict')
        pr = pr.reshape((output_height, output_width, n_classes)).argmax(axis=2)
        seg_img = np.zeros((output_height, output_width, 3))
        for c in range(n_classes):
            seg_img[:, :, 0] += ((pr[:, :] == c) * (colors[c][0])).astype('uint8')
            seg_img[:, :, 1] += ((pr[:, :] == c) * (colors[c][1])).astype('uint8')
            seg_img[:, :, 2] += ((pr[:, :] == c) * (colors[c][2])).astype('uint8')
        seg_img = cv2.resize(seg_img, (tile.shape[0], tile.shape[1]))
        tiles[i]=seg_img
        i+=1

    outputImg = LoadBatches.combineImg(tiles, im.shape[0], im.shape[1])

    cv2.imwrite(outName, outputImg)
    break
