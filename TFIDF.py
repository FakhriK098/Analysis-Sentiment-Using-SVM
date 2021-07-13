import numpy as np
import pandas as pd

def calc_TF(dokumen):
    TF_dict = {}
    for term in dokumen:
        if term in TF_dict:
            TF_dict[term] += 1
        else:
            TF_dict[term] = 1

    for term in TF_dict:
        TF_dict[term] = TF_dict[term] / len(dokumen)
    return TF_dict

def calc_DF(tfDict):
    count_DF = {}
    for dokumen in tfDict:
        for term in dokumen:
            if term in count_DF:
                count_DF[term] +=1
            else:
                count_DF[term] = 1
    return count_DF

def calc_IDF(dokumen, DF):
    IDF_Dict = {}
    for term in DF:
        IDF_Dict[term] = np.log(dokumen/ (DF[term]+1))
    return IDF_Dict


def tfidf(data):
    document_count = len(data)
    all_dokument = []
    for d in range(len(data['Tweet'])):
        data['Tweet'][d] = str(data['Tweet'][d]).split()


    data['TF_dict'] = data['Tweet'].apply(calc_TF)

    DF = calc_DF(data['TF_dict'])
    IDF = calc_IDF(document_count, DF)

    def calc_TF_IDF(TF):
        TF_IDF_Dict = {}

        for key in TF:
            TF_IDF_Dict[key] = TF[key] * IDF[key]
        return TF_IDF_Dict

    data['TF-IDF'] = data['TF_dict'].apply(calc_TF_IDF)

    unique_term = pd.read_csv("D:\\TUGAS AKHIR\\bahan\\Bahan Unique.csv")

    def calc_TF_IDF_Vec(TF_IDF_Dict):
        TF_IDF_vector = [0.0] * len(unique_term['unique term'])

        for i, term in enumerate(unique_term['unique term']):
            if term in TF_IDF_Dict:
                TF_IDF_vector[i] = TF_IDF_Dict[term]
        return TF_IDF_vector

    data['TF-IDF-Vec'] = data['TF-IDF'].apply(calc_TF_IDF_Vec)
    return data