# import the necessary packages
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense


class LeNet:
    @staticmethod
    def build(width, height, depth, classes, weightsPath=None):
        """
        Function is responsible for constructing the  network architecture. Width, height and depth refers to the
        image.
        :param width:
        :param height:
        :param depth:
        :param classes: Number of  unique class labels in our data
        :param weightsPath: To be used for loading a pre-trained model
        :return:
        """
        # initialize the model. Rather instantiate a Sequential class which will be used for constructing the network.
        model = Sequential()

        # first set of CONV => RELU => POOL
        model.add(Convolution2D(20, 5, 5, border_mode="same",
                                input_shape=(depth, height, width)))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # second set of CONV => RELU => POOL
        # It's common to see the number of CONV  filters increase in deeper layers of the network.
        model.add(Convolution2D(50, 5, 5, border_mode="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # set of FC => RELU layers
        # Flatten the output of the MaxPooling2D layer obtained in the previous layer. This is needed to apply
        # dense or fully connected layer.
        model.add(Flatten())
        model.add(Dense(500))  # FC layer with 500 units.
        model.add(Activation("relu"))

        # softmax classifier
        # Note than this line defines another Dense layer, but accepts a variable (i.e., not hardcoded) size.
        model.add(Dense(classes))
        # Finally, apply the softmax classifier (multinomial logistic regression) that'll return a list of
        # probabilities, one for each of the 10 class labels. The class label with the largest probability will
        # be chosen as the final classification of the network.
        model.add(Activation("softmax"))

        # if a weights path is supplied (indicating that the model was
        # pre-trained), then load the weights
        if weightsPath is not None:
            model.load_weights(weightsPath)

        # return the constructed network architecture
        return model
