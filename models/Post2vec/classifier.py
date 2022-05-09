# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import torch
import ast
import sklearn
import numpy
import csv
from itertools import islice
from utils.time_util import get_current_time
from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC, LinearSVC
import sklearn.tree as st
from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import precision_recall_curve
# from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score



# 读取
fpath = "vecter.csv"
fpath1 = "vecter5.csv"
fpath2 = "vecter6.csv"


def openfile(file_name, state):
    with open(file_name, 'r', encoding='utf-8', errors='surrogatepass') as all:
        df = pd.read_csv(all)
        x = list()
        y = list()
        cnt = 0
        filter_cnt = 0
        f = 0
        for idx, row in df.iterrows():
            try:
                qid = row['id']
                title = row['title'].replace('\n', ' ').replace('[', ' ').replace(']', ' ').split()
                text = row['desc_text'].replace('\n', ' ').replace('[', ' ').replace(']', ' ').split()
                code = row['desc_code'].replace('\n', ' ').replace('[', ' ').replace(']', ' ').split()
                x1 = numpy.array(title+text+code)
                x.append(x1)
                if state == 0:
                    y1 = row['state']
                    y.append(y1)
            except Exception as e:
                print("Skip qid %s because %s" % (qid, e))
                filter_cnt += 1
        return x, y

# 加载数据集，切分数据集80%训练，20%测试


x1, y1 = openfile(fpath, 0)
x2, y2 = openfile(fpath1, 1)
print(len(x2))
# 调用MultinomialNB分类器
#     clf = st.DecisionTreeClassifier(max_depth=4).fit(x_train, y_train)
#     clf = SVC(kernel='rbf', C=1.0, gamma='auto').fit(x_train, y_train)
#     clf = MultinomialNB().fit(x_train, y_train)
clf = RandomForestClassifier().fit(x1, y1)
predict = clf.predict(x2)
with open(fpath1, 'r', encoding='utf-8', errors='surrogatepass') as all:
    rd = csv.reader(all)
    row_num = 0
    cnt = 0
    corpus_header = ["id", "phase"]
    with open(fpath2, 'w') as out:
        wr = csv.writer(out)
        wr.writerow(corpus_header)
        for row in islice(rd, 0, 100000):
            phase = predict[cnt]
            row_num += 1
            if row_num == 1:
                continue
            qid = row[0]
            # title = row[1]
            # desc = row[2]
            # code = row[3]
            # creation_date = row[4]
            # tags = row[5]


            try:
                wr.writerow(
                    [qid, phase])
                     # title, desc, code, creation_date, tags])
                cnt += 1
            except Exception as e:
                print("Skip id=%s" % qid)
                print("Error msg: %s" % e)
            if row_num % 10000 == 0:
                print("Processing %s row..." % row_num, get_current_time())
    # print(doc_class_predicted)
    # print(accuracy_score(y_test, doc_class_predicted))



