import numpy as np
import matplotlib.pyplot as plt
import cv2
import math
import random

class Transformations2D():
    def __init__(self, channels, angle, translation, output_side):
        self.pair = []
        self.dims = dict()
        self.angle = angle
        self.translation = translation
        self.side = output_side
        self.center = (0, 0)

        self.rotated_pair = []
        self.transformed_pair = np.zeros((2, output_side, output_side, channels))
        self.report = dict(transformation="None", angle=0, translation="None")

    def rotation(self, scale=1.0):
        teta = random.randint(-self.angle, self.angle)
        self.report['angle'] = teta

        M = cv2.getRotationMatrix2D(self.center, teta, scale)
        for img_counter, img in enumerate(self.pair):
            self.rotated_pair[img_counter] = cv2.warpAffine(img, M, (self.dims['width'], self.dims['height']))

    def croppingTranslation(self, pair="None", move="None"):
        if type(pair) == str:
            pair = self.pair
        if move == "None":
            move = random.randint(0, self.translation)
        else:
            self.report['translation'] = 0

        which = random.randint(0, 3)
        where = np.zeros(4, dtype='int8')
        where[which] = 1
        if (which == 0 or which == 1) and self.report['translation'] != 0:
            self.report['translation'] = f"x by {move}"
        if (which == 2 or which == 3) and self.report['translation'] != 0:
            self.report['translation'] = f"y by {move}"
        
        x_cropLimit_low = int(self.center[0] - self.side/2 + where[0]*move - where[1]*move)
        x_cropLimit_high = int(self.center[0] + self.side/2 + where[0]*move - where[1]*move)
        y_cropLimit_low = int(self.center[1] - self.side/2 + where[2]*move - where[3]*move)
        y_cropLimit_high = int(self.center[1] + self.side/2 + where[2]*move - where[3]*move)

        for img_counter, img in enumerate(pair):
            self.transformed_pair[img_counter] = img[y_cropLimit_low:y_cropLimit_high, x_cropLimit_low:x_cropLimit_high, 0:self.dims['channels']+1]

    def random_transform(self, pair, dimensions):
        self.pair = np.asarray(pair)
        self.dims = dimensions
        self.center = (int(math.floor(self.dims['width']/2)), int(math.floor(self.dims['height']/2)))
        self.rotated_pair = np.zeros((2, self.dims['height'], self.dims['width'], self.dims['channels']))

        which_transform = random.randint(0, 1)
        if which_transform == 0:
            self.rotation()
            self.croppingTranslation(self.rotated_pair, 0)
            self.report['transformation'] = 'rotation'
        elif which_transform == 1:
            self.croppingTranslation()
            self.report['transformation'] = 'translation'

    def getTransformedPair(self):
        returning = [self.transformed_pair, self.report]
        return returning

dims = {
    'slices': 2,
    'width':960,
    'height': 640,
    'channels':3,
}

#get trying dataset of two houses
dataset = (cv2.imread('../../Data/Tests_data/try/house1.jpg'), cv2.imread('../../Data/Tests_data/try/house2.jpg'))
print(np.shape(dataset[0]))

#look at how opencv puts (y, x)
'''fig, ax = plt.subplots()
ax.imshow(dataset[0])
ax.scatter(int(math.floor(dims['width']/2)), int(math.floor(dims['height']/2)))
fig.show()
plt.pause(10)'''


dataset = list(dataset)
dataset[1] = dataset[1][0:640,0:960,:]
dataset = tuple(dataset)
assert np.shape(dataset[0]) == np.shape(dataset[1])

#make a not transformed dataset
no_transform = Transformations2D(3, 0, 0, 288)
no_transform.random_transform(dataset, dims)
only_cropped = no_transform.getTransformedPair()[0]

#try transformations
transform = Transformations2D(3, 10, 15, 288)
transform.random_transform(dataset, dims)

transformed_data = transform.getTransformedPair()[0]
transformation_report = transform.getTransformedPair()[1]
print(f"The trasformation report: {transformation_report}")

#sanity check
plt.subplot(1, 2, 1)
plt.imshow((only_cropped[0])/255)

plt.subplot(1, 2, 2)
plt.imshow((transformed_data[0])/255)
plt.show()