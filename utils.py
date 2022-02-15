from sklearn.datasets import load_svmlight_file

MNIST = "dataset/mnist/mnist.scale"
IJCNN1 = "dataset/ijcnn1/ijcnn1"
SHUTTLE = "dataset/shuttle/shuttle.scale.tr"

if __name__ == '__main__':
    data = load_svmlight_file(MNIST)
    x_train = data[0].toarray()
    print(x_train.shape)
    #print(len(data[1]))

