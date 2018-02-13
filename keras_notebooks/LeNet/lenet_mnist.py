"""
1. Loading the MNIST dataset.
2. Partitioning MNIST into training and testing splits.
3. Loading and compiling the LeNet architecture.
4. Training the network.
5. Optionally saving the serialized network weights to disk so that it can be reused (without
having to re-train the network).
6. Displaying visual examples of the network output to demonstrate that our implementation is
indeed working properly.
"""

# USAGE
# python lenet_mnist.py --save-model 1 --weights output/lenet_weights.hdf5
# python lenet_mnist.py --load-model 1 --weights output/lenet_weights.hdf5

# import the necessary packages
from image_classification.cnn.networks import lenet
from image_classification.cnn.networks import lenet_ver_2
# from pyimagesearch.cnn.networks import LeNet
from sklearn.cross_validation import train_test_split
from sklearn import datasets
from keras.optimizers import SGD
from keras.utils import np_utils
import numpy as np
import argparse

# from keras import backend as K
# K.set_image_dim_ordering('tf')

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

# import cv2

# construct the argument parse and parse the arguments
# Following are the optional command line arguments each of which are detailed below:
# '--save-model': An indicator variable, used to specify whether or not we should save
# our model to disk after training LeNet.
# '--load-model': Another indicator variable, this time specifying whether or not we
# should load a pre-trained model from disk.
# '--weights': In the case that --save-model  is supplied, the --weights-path  should
# point to where we want to save the serialized model. And in the case that --load-model
# is supplied, the --weights  should point to where the pre-existing weights file lives on our system.
ap = argparse.ArgumentParser()
ap.add_argument("-s", "--save-model", type=int, default=-1,
                help="(optional) whether or not model should be saved to disk")
ap.add_argument("-l", "--load-model", type=int, default=-1,
                help="(optional) whether or not pre-trained model should be loaded")
ap.add_argument("-w", "--weights", type=str,
                help="(optional) path to weights file")
args = vars(ap.parse_args())

# grab the MNIST dataset (if this is your first time running this
# script, the download may take a minute -- the 55MB MNIST dataset
# will be downloaded)
print("[INFO] downloading MNIST...")
dataset = datasets.fetch_mldata("MNIST Original")

# reshape the MNIST dataset from a flat list of 784-dim vectors, to
# 28 x 28 pixel images, then scale the data to the range [0, 1.0]
# and construct the training and testing splits
data = dataset.data.reshape((dataset.data.shape[0], 28, 28))
data = data[:, np.newaxis, :, :]  # Change shape from (70000, 28, 28) to (70000, 1, 28, 28). 1 is for # of channels.
# Reduce images from the range [0, 255] to [0, 1.0], a common scaling technique.
(trainData, testData, trainLabels, testLabels) = train_test_split(
    data / 255.0, dataset.target.astype("int"), test_size=0.33)

# transform the training and testing labels into vectors in the
# range [0, classes] -- this generates a vector for each label,
# where the index of the label is set to `1` and all other entries
# to `0`; in the case of MNIST, there are 10 class labels
trainLabels = np_utils.to_categorical(trainLabels, 10)
testLabels = np_utils.to_categorical(testLabels, 10)

# initialize the optimizer and model
print("[INFO] compiling model...")
opt = SGD(lr=0.01)
model = lenet_ver_2.LeNet.build(width=28, height=28, depth=1, classes=10,
                          weightsPath=args["weights"] if args["load_model"] > 0 else None)
model.compile(loss="categorical_crossentropy", optimizer=opt,
              metrics=["accuracy"])

# only train and evaluate the model if we *are not* loading a
# pre-existing model
if args["load_model"] < 0:
    print("[INFO] training...")
    model.fit(trainData, trainLabels, batch_size=128, nb_epoch=5,
              verbose=1)

    # show the accuracy on the testing set
    print("[INFO] evaluating...")
    (loss, accuracy) = model.evaluate(testData, testLabels,
                                      batch_size=128, verbose=1)
    print("[INFO] accuracy: {:.2f}%".format(accuracy * 100))

# check to see if the model should be saved to file
if args["save_model"] > 0:
    print("[INFO] dumping weights to file...")
    model.save_weights(args["weights"], overwrite=True)

# # randomly select a few testing digits
# for i in np.random.choice(np.arange(0, len(testLabels)), size=(10,)):
#     # classify the digit
#     probs = model.predict(testData[np.newaxis, i])
#     prediction = probs.argmax(axis=1)
#
#     # resize the image from a 28 x 28 image to a 96 x 96 image so we
#     # can better see it
#     image = (testData[i][0] * 255).astype("uint8")
#     image = cv2.merge([image] * 3)
#     image = cv2.resize(image, (96, 96), interpolation=cv2.INTER_LINEAR)
#     cv2.putText(image, str(prediction[0]), (5, 20),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
#
#     # show the image and prediction
#     print("[INFO] Predicted: {}, Actual: {}".format(prediction[0],
#                                                     np.argmax(testLabels[i])))
#     cv2.imshow("Digit", image)
#     cv2.waitKey(0)
