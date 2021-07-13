import re
import Connection as cn

punc = '''()-[]{};:'"\,<>./@#$%^&*~?!'''
under = '_'

kamus_alay = cn.showKamusAlay()
alay_dict_map = dict(zip(kamus_alay['kata alay'], kamus_alay['kata normal']))

id_stopwords = cn.showStopword()

def lower_case(txt):
    return txt.lower()

def remove_under(txt):
    return ''.join(tx for tx in txt if tx not in under)

def remove_unnecessary(txt):
    txt = re.sub('\n',' ',txt)
    txt = re.sub('rt',' ',txt)
    txt = re.sub('@[\w]+',' ',txt)
    txt = re.sub('lgbt',' ',txt)
    txt = re.sub('((ww,w\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))',' ',txt)
    txt = re.sub(' +',' ',txt)
    return txt

def remove_punc(txt):
    return ''.join(tx for tx in txt if tx not in punc)

def normalize_alay(txt):
    return ' '.join([alay_dict_map[word] if word in alay_dict_map else word for word in txt.split(' ')])

def remove_stopword(txt):
    txt = ' '.join([' ' if word in id_stopwords.stopword.values else word for word in txt.split(' ')])
    txt = re.sub(' +',' ',txt)
    txt = txt.strip()
    return txt

factory = cn.StemmerFactory()
stemmer = factory.create_stemmer()

def stemming(txt):
    return stemmer.stem(txt)

def preprocessing(txt):
    txt = lower_case(txt)
    txt = remove_under(txt)
    txt = remove_unnecessary(txt)
    txt = remove_punc(txt)
    txt = normalize_alay(txt)
    txt = stemming(txt)
    txt = remove_stopword(txt)
    return txt