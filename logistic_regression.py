import numpy as np
from numpy import *
import matplotlib.pyplot as plt
# matplotlib.use('TKAgg')


def loadDataSet():
    """
    Opens the text file and reads every line. The first two values in each line are X1 and X2. Third value is the
    class label. There are a total of 100 instance. This function sets the value of X0 to 1 for convenience.
    :return:
    dataMat: List of length 100.
    labelMat: List of length 100.
    """
    dataMat = []
    labelMat = []
    fr = open('./datasets/logistic_regression/test_set.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat,labelMat


def sigmoid(inX):
    return 1.0/(1+exp(-inX))


def gradAscent(dataMatIn, classLabels, alpha=0.001, maxCycles=500):
    """
    :param dataMatIn: 2D Numpy array where columns are features and rows are instances of data. Including the intercept,
    dataset has three features. So, it's a 100*3 matrix.
    :param classLabels:
    :param alpha: Learning rate
    :param maxCycles: Maximum number of iterations
    :return:
    """
    print "type(dataMatIn)", type(dataMatIn)
    print "type(classLabels)", type(classLabels)
    dataMatrix = mat(dataMatIn)  # Convert to NumPy matrix
    labelMat = mat(classLabels).transpose()  # Convert to NumPy matrix and transpose to make it a 100*1 column vector.
    print "dataMatrix shape", dataMatrix.shape
    m,n = shape(dataMatrix)
    weights = ones((n,1))
    print "shape weights:", weights.shape
    # Iterate over the data set and return the weights
    for k in range(maxCycles):              # heavy on matrix operations
        h = sigmoid(dataMatrix*weights)     # matrix mult
        print "shape h:", h.shape
        error = (labelMat - h)              # vector subtraction
        print "type(error):", type(error)
        print "shape error", error.shape
        weights = weights + alpha * dataMatrix.transpose() * error  # matrix mult
        print "shape weights:", weights.shape
    return weights


def plotBestFit(weights, method="sgd"):
    dataMat, labelMat = loadDataSet()
    dataArr = array(dataMat)
    n = shape(dataArr)[0]
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n):
        if int(labelMat[i])== 1:
            xcord1.append(dataArr[i,1]); ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1]); ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = arange(-3.0, 3.0, 0.1)
    y = (-weights[0]-weights[1]*x)/weights[2]
    print "y shape", y.shape
    print "x shape", x.shape
    if method=="gd":
        ax.plot(np.expand_dims(x, axis=0), y)
        plt.xlabel('X1')
        plt.ylabel('X2')
        plt.title("Coefficients with vanilla Gadient Descent")
    elif method == "sgd" or method == "modified_sgd":
        ax.plot(x, y)
        plt.xlabel('X1')
        plt.ylabel('X2')
        plt.title("Coefficients with Stochastic Gadient Descent")
    elif method == "modified_sgd":
        ax.plot(x, y)
        plt.xlabel('X1')
        plt.ylabel('X2')
        plt.title("Coefficients with improved Stochastic Gadient Descent")


def stocGradAscent0(dataMatrix, classLabels, alpha=0.01):
    """
    :param dataMatrix:
    :param classLabels:
    :return:
    """
    print type(dataMatrix)
    m,n = shape(dataMatrix)
    weights = ones(n)   #initialize to all ones
    for i in range(m):
        # The variables h and error below are single values now rather than being vectors in GD.
        h = sigmoid(sum(dataMatrix[i]*weights))
        error = classLabels[i] - h
        weights = weights + alpha * error * dataMatrix[i]
    return weights


def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    """
    Difference from the previous SGD
    1. Alpha is no more a constant now. This is with a view to ensuring that the weights converge without much of osci-
    llation. It decreases as number of iterations increase but never reaches zero. There is a constant term in alpha.
    2. Data instances are randomly selected in updating the weights.
    :param dataMatrix:
    :param classLabels:
    :param numIter:
    :return:
    """
    m,n = shape(dataMatrix)
    weights = ones(n)  # initialize to all ones
    for j in range(numIter):
        dataIndex = range(m)
        for i in range(m):
            alpha = 4/(1.0+j+i)+0.0001  # alpha decreases with iteration, does not
            # Randomly select an
            randIndex = int(random.uniform(0,len(dataIndex)))  # go to 0 because of the constant
            h = sigmoid(sum(dataMatrix[randIndex]*weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del(dataIndex[randIndex])
    return weights


# Following three functions are meant for the classification example.
def classifyVector(inX, weights):
    """
    Given an input vector and weights, calculates the sigmoid. If the value of sigmoid is > 0.5, it returns a 1. Other-
    wise a 0.
    :param inX:
    :param weights:
    :return:
    """
    prob = sigmoid(sum(inX*weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0

def colicTest():
    """
    Opens the training and the test data and formats it. Last column in the training set is assumed to be the class
    labels. Two class labels in this dataset viz. "lived" and "did not live".
    :return:
    """
    frTrain = open('./datasets/logistic_regression/horseColicTraining.txt');
    frTest = open('./datasets/logistic_regression/horseColicTest.txt')
    trainingSet = []; trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr =[]
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    trainWeights = stocGradAscent1(array(trainingSet), trainingLabels, 1000)
    errorCount = 0; numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr =[]
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(array(lineArr), trainWeights))!= int(currLine[21]):
            errorCount += 1
    errorRate = (float(errorCount)/numTestVec)
    print "the error rate of this test is: %f" % errorRate
    return errorRate


def multiTest():
    """
    Runs the function colicTest() 10 times and returns the average error rate after each iteration.
    :return: NULL
    """
    numTests = 10
    errorSum=0.0
    for k in range(numTests):
        errorSum += colicTest()
    print "after %d iterations the average error rate is: %f" % (numTests, errorSum/float(numTests))


if __name__ == "__main__":
    # Vanilla gradient descent
    dataArr, labelArr = loadDataSet()
    weights = gradAscent(dataArr, labelArr)
    # Plot of decision boundary
    plotBestFit(weights, "gd")
    plt.savefig("./output/logistic_regression/classification_gd.png")


    # Stochastic Gradient Descent
    weights_sgd = stocGradAscent0(array(dataArr), labelArr)  # weights_sgd shape: (3,)
    plotBestFit(weights_sgd, "sgd")
    plt.savefig("./output/logistic_regression/classification_sgd.png")

    # Modified SGD
    weights_modified_sgd = stocGradAscent1(array(dataArr), labelArr)  # weights_sgd shape: (3,)
    plotBestFit(weights_modified_sgd, "modified_sgd")
    plt.savefig("./output/logistic_regression/classification_modified_sgd.png")

    # Example of classification using Logistic Regression
    multiTest()