import mysql.connector
import pandas as pd

db = mysql.connector.connect(
    host = 'localhost',
    user = 'root',
    passwd = 'database123',
    database = 'analysis_sentiment'
)

if db.is_connected():
    cursor = db.cursor()

def showStopword():
    sql = 'SELECT kata_stopword FROM stopword'
    cursor.execute(sql)
    result = cursor.fetchall()

    data = pd.DataFrame(result, columns=['stopword'])
    return data

def showKamusAlay():
    sql = 'SELECT kata_alay, kata_normal FROM kamus_alay'
    cursor.execute(sql)
    result = cursor.fetchall()

    data = pd.DataFrame(result, columns=['kata alay', 'kata normal'])
    return data

def inputHasil(lamda, learningRate, slack, iterasi_max, alpha, akurasi, prec, recall):
    values = (lamda, learningRate, slack, iterasi_max, alpha, akurasi, prec, recall)
    sql = 'INSERT INTO hasil (lambda, learning_rate, slack, iterasi_max, alpha, akurasi, prec, recall) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)'
    cursor.execute(sql,values)
    db.commit()

def showHasil():
    sql = 'SELECT lambda, learning_rate, slack, iterasi_max, alpha, akurasi, prec, recall FROM hasil'
    cursor.execute(sql)
    result = cursor.fetchall()

    data = pd.DataFrame(result, columns=['Lambda', 'Learning Rate', 'Slack', 'Jumlah Iterasi', 'Alpha', 'Akurasi', 'Precision', 'Recall'])
    return data