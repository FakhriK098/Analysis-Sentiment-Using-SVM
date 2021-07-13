import pandas as pd
import numpy as np

def matrikHessian(data, lmbda):
    matrik_hessian = []
    kernel = []

    # menghitung kernel
    for x in range(len(data['TF-IDF-Vec'])):
        k = []
        for y in range(len(data['TF-IDF-Vec'])):
            a = np.dot(data['TF-IDF-Vec'][x], data['TF-IDF-Vec'][y])
            k.append((a + 1) ** 2)
        kernel.append(k)

    # menghitung matrik hessian
    for x in range(len(data)):
        hessian = []
        d = 0
        for y in range(len(data)):
            d = (data['Label'][x] * data['Label'][y]) * (kernel[x][y] + (lmbda ** 2))
            hessian.append(d)
        matrik_hessian.append(hessian)

    return matrik_hessian, data


def FindEi(alphaN, matrikHessian):
    result_Ei = []
    for x in range(len(matrikHessian)):
        a = pd.Series(matrikHessian[x])
        result_Ei.append(sum(a * alphaN[x]))
    return result_Ei


def CalculateBA(learningRate, resultEi, alphaN, slack):
    result_BA = []
    for x in range(len(resultEi)):
        ba = min(max(learningRate * (1 - resultEi[x]), -alphaN[x]), (slack - alphaN[x]))
        result_BA.append(ba)
    return result_BA


def FindAlphaNew(alphaN, resultBA):
    alpha_new = []
    for x in range(len(resultBA)):
        alpha_new.append(alphaN[x] + resultBA[x])  # menghitung alpha baru
    return alpha_new


def FindBias(data):
    positif = []
    negatif = []

    # mencari aplha terbesar dari masing-masing kelas
    for x in range(len(data)):
        if data['Label'][x] == 1:
            positif.append(data['Alpha'][x])
        else:
            negatif.append(data['Alpha'][x])
    x_positif = max(positif)
    x_negatif = max(negatif)

    # mencari vector tf-idf dari alpha terbesar
    x_p, x_n = 0, 0
    for x in range(len(data)):
        if data['Alpha'][x] == x_positif:
            x_p = data['TF-IDF-Vec'][x]
    for x in range(len(data)):
        if data['Alpha'][x] == x_negatif:
            x_n = data['TF-IDF-Vec'][x]

    vector_positif = np.dot(x_p, x_p)
    vector_negatif = np.dot(x_n, x_n)
    print(vector_positif, vector_negatif)

    # perhitungan kernel dari vector data dengan vector alpha terbesar
    k_positif = []
    k_negatif = []
    for x in data['TF-IDF-Vec']:
        k_positif.append((np.dot(x, x_p) + 1) ** 2)
        k_negatif.append((np.dot(x, x_n) + 1) ** 2)

    b_positif = []
    b_negatif = []
    for x in range(len(data)):
        b_positif.append(data['Alpha'][x] * data['Label'][x] * k_positif[x])
        b_negatif.append(data['Alpha'][x] * data['Label'][x] * k_negatif[x])

    sum_b_positif = sum(b_positif)
    sum_b_negatif = sum(b_negatif)

    # menghitung nilai bias
    b = -(0.5 * (sum_b_positif + sum_b_negatif))
    return b


def SequentialSVM(alpha, data, lmbda, learningRate, slack, iterasi):
    matriks_Hessian, data_new = matrikHessian(data, lmbda)
    alpha_old = []

    # proses pembuatan aplha awal
    for x in range(len(matriks_Hessian)):
        alpha_old.append(alpha)

    jmlhiterasi = iterasi
    alpha_All = []
    iterasi = 0

    while (iterasi < jmlhiterasi):
        alpha_All.append(alpha_old)
        result_Ei = FindEi(alpha_old, matriks_Hessian)
        result_BA = CalculateBA(learningRate, result_Ei, alpha_old, slack)
        alpha_old = FindAlphaNew(alpha_old, result_BA)
        iterasi += 1
    data_new["Alpha"] = alpha_All[-1]

    bias = FindBias(data_new)
    return bias, data_new