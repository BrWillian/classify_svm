import sys
import os
import cv2
import numpy as np
from sklearn.svm import SVC
import _pickle as pickle


def read_files(directory):
    print("Reading files...")
    feature_list = list()
    label_list = list()
    num_classes = 0
    for root, dirs, files in os.walk(directory):
        for d in dirs:
            num_classes += 1
            images = os.listdir(root + d)
            for image in images:
                label_list.append(d)
                feature_list.append(extract_feature(root + d + "/" + image))

    print(str(num_classes) + " classes")
    return np.asarray(feature_list), np.asarray(label_list)


def extract_feature(image_file):
    img = cv2.imread(image_file)
    img = cv2.resize(img, (30,30), interpolation=cv2.INTER_CUBIC)
    img = img.flatten()
    img = img / np.mean(img)
    return img


def train(directory):

    # gerar duas arrays com features e classes
    feature_array, label_array = read_files(directory)

    # checar modelo
    if os.path.isfile("svm_model.pkl"):
        print("Using model")
    else:
        print("Train....")

        # Caso nao tenha nenhum modelo cria um novo modelo
        svm = SVC()
        svm.fit(feature_array, label_array)

        print("Saving model...")
        pickle.dump(svm, open("svm_model.pkl", "wb"))


def classify(directory):

    # leio modelo
    svm = pickle.load(open("svm_model.pkl", "rb"))

    # extraio features das imagens a serem classificadas
    feature_array, label_array = read_files(directory)


    # calculo a taxa de acerto
    right = 0
    total = 0
    for x, y in zip(feature_array, label_array):
        x = x.reshape(1, -1)
        prediction = svm.predict(x)[0]

        if y == prediction:
            right += 1
        total += 1

    accuracy = float(right) / float(total) * 100
    print(str(accuracy) + "% acuracia")



if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage python3 svm.py --option [--train img_folder] or [--classify img_folder]")
        exit()

    if sys.argv[1] == '--train':
        directory = sys.argv[2]
        train(directory)

    if sys.argv[1] == '--classify':
        directory = sys.argv[2]
        classify(directory)
