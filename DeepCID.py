import tensorflow as tf
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
import os
import csv

def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])

if __name__ == '__main__':
    path = '.'
    # Load the mixture spectrum, its' labels and components information
    datafile1 = os.path.join(path, 'mixture.npy')
    Xtest = np.load(datafile1)

    datafile2 = os.path.join(path, 'label.npy')
    label = np.load(datafile2)

    n = label.shape[0]

    csv_reader = csv.reader(open(os.path.join(path, 'namedata.csv'), encoding='utf-8'))
    names = [row for row in csv_reader]
    ypred = np.zeros((n * Xtest.shape[0], 2))

    # Set the root directory of models and reload the models one by one
    root = os.path.join(path, 'model')
    list_dirs = os.walk(root)
    i = 0
    for root_dir, dirs, files in list_dirs:
        for d in dirs:
            Y1 = label[i, :].reshape([Xtest.shape[0], 1])
            Y2 = np.ones((Y1.shape)) - Y1
            Ytest = np.concatenate((Y1, Y2), axis=1)

            model_dir = os.path.join(root_dir, d)
            #os.chdir(model_dir)
            datafile = os.path.join(model_dir,'X_scale.npy')
            X_scale = np.load(datafile)
            Xtest_scale = (Xtest - X_scale[0]) / X_scale[1]

            # Load the model
            model = tf.keras.models.load_model('./saved_model')

            # Make predictions
            test_ypred = model.predict(Xtest_scale)
            ypred[i * Xtest.shape[0]:(i + 1) * Xtest.shape[0], :] = test_ypred
            print(f'component {i} finished. The test accuracy {accuracy(ypred[i * Xtest.shape[0]:(i + 1) * Xtest.shape[0], :], Ytest):.1f}%')

            i += 1

    # Print the components' names that are present in the mixtures
    for j in range(Xtest.shape[0]):
        print(f'The {j}th sample contains:')

        # Yreal is designed to calculate the accuracy.
        y_real1 = label[:, j - Xtest.shape[0]].reshape([n, 1])
        y_real2 = np.ones((y_real1.shape)) - y_real1
        y_real = np.concatenate((y_real1, y_real2), axis=1)

        ypre = np.zeros((n, 2))
        for k in range(n):
            ypre[k, :] = ypred[j, :]
            j = j + Xtest.shape[0]

        for h in range(n):
            if ypre[h, 0] >= 0.5:
                print(names[h])
        print('The prediction finished')
