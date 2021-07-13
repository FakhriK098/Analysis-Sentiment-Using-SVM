import pandas as pd
import numpy as np
import TFIDF as tf
import Trainning as tr

def analisisSentiment(data_latih, bias, data_uji):
    term_uji = data_uji['TF-IDF-Vec'][0]
    df = {'alpha': [data_latih['Alpha'][0]],
          'bias': [bias],
          'data uji': [data_latih['TF-IDF-Vec'][0]]}
    fix_data = pd.DataFrame(df)

    print(data_latih['Alpha'][0])
    print(bias)

    kernel = []
    for x in range(len(data_uji['TF-IDF-Vec'])):
        k = []
        for y in range(len(data_uji['TF-IDF-Vec'])):
            a = np.dot(data_uji['TF-IDF-Vec'][x], data_latih['TF-IDF-Vec'][y])
            k.append((a + 1) ** 2)
        kernel.append(k)

    result_h = []
    for x in range(len(data_uji)):
        result = 0
        for y in range(len(data_uji)):
            h = data_latih['Alpha'][y] * data_uji['Label'][y] * kernel[x][y]
            result += h
        result_h.append(result + bias)

    system_result = []
    for x in result_h:
        if x >= 0:
            system_result.append(1)
        else:
            system_result.append(-1)
    data_uji["System Result"] = system_result

    TP, TN, FP, FN = 0, 0, 0, 0
    for xA in range(len(data_uji)):
        if data_uji['Label'][xA] == 1:
            if data_uji['Label'][xA] == data_uji['System Result'][xA]:
                TP += 1
            else:
                FP += 1
        else:
            if data_uji['Label'][xA] == data_uji['System Result'][xA]:
                TN += 1
            else:
                FN += 1

    # perhitngan confusion matrix
    akurasi = (TP + TN) / (TP + FP + TN + FN) * 100
    precision = TP / (TP + FP)
    recall = TP / (TP + FN) if FN != 0 else 0

    return data_uji, akurasi, precision, recall, fix_data


def trainning(data_latih, data_uji, alpha, lmbda, learningRate, slack, iterasi):
    # mengubah label 0 -> 1 or 1 -> -1
    label_new = []
    for label in data_latih['Sentiment']:
        if label == 'Positif':
            label_new.append(1)
        else:
            label_new.append(-1)
    data_latih["Label"] = label_new

    label_new = []
    for label in data_uji['Sentiment']:
        if label == 'Positif':
            label_new.append(1)
        else:
            label_new.append(-1)
    data_uji["Label"] = label_new

    bias, final_data = tr.SequentialSVM(alpha, data_latih, lmbda, learningRate, slack, iterasi)
    final_data_fix, akurasi, precision, recall, fix_data = analisisSentiment(final_data, bias, data_uji)

    return final_data_fix, akurasi, precision, recall, fix_data


def testing(kalimat, sentimen):
    wr = [kalimat]
    label = 0
    if sentimen == 'Positif' or sentimen == 'positif':
        label = 1
    else:
        label = -1
    data = pd.DataFrame(wr, columns=['Tweet'], index=None)
    data = tf.tfidf(data)
    k = np.dot(data['TF-IDF-Vec'][0], data['TF-IDF-Vec'][0])
    a = (np.dot(data['TF-IDF-Vec'][0], data['TF-IDF-Vec'][0]) + 1) ** 2

    h = (0.1 * label * a) + 212.2109409639695
    result = ''
    if h >= 0:
        result = 'Positif'
    else:
        result = 'Negatif'
    return result