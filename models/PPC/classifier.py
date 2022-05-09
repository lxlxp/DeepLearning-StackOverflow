# -*- coding: utf-8 -*-
from inspect import isdatadescriptor
from os import write
import numpy as np
import pandas as pd
# import torch
import ast
import sklearn
import numpy
import csv
import scipy.io as scio
from itertools import islice
# from utils.time_util import get_current_time
from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC, LinearSVC
import sklearn.tree as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import  AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
# from sklearn.metrics import precision_recall_curve
# from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, confusion_matrix



# 读取
fpath = "result.csv"
fpath1 = "result4.csv"
fpath2 = "result5.csv"


def openfile(file_name, state,qidif):
    with open(file_name, 'r', encoding='utf-8', errors='surrogatepass') as all:
        df = pd.read_csv(all)
        x = list()
        y = list()
        qids = list()
        cnt = 0
        filter_cnt = 0
        f = 0
        for idx, row in df.iterrows():
            try:
                qid = row['id']
                vector = row['vector'].replace('\n', ' ').replace('[', ' ').replace(']', ' ').split()
                # title = row['title'].replace('\n', ' ').replace('[', ' ').replace(']', ' ').split()
                # text = row['desc_text'].replace('\n', ' ').replace('[', ' ').replace(']', ' ').split()
                # code = row['desc_code'].replace('\n', ' ').replace('[', ' ').replace(']', ' ').split()
                # x1 = numpy.array(title+text+code)
                vector = list(map(float,vector))
                x.append(vector)
                if state == 0:
                    phase = row['phase']
                    # y1 = row['state']
                    y.append(phase)
                if qidif == 0:
                    qids.append(qid)
            except Exception as e:
                print("Skip qid %s because %s" % (qid, e))
                filter_cnt += 1
        return x, y, qids

# 加载数据集，切分数据集80%训练，20%测试


x_train, y_train, qid_train= openfile(fpath, 0, 1)
x_test, y_test, qid_test= openfile(fpath1, 0, 1)
# print(len(x_test))
# print(len(qid_test))
# print(len(x2))

# 2022.01.15
# file = "question_embedding.mat"
# data = scio.loadmat(file)
# x_test , qid_test= np.split(data['name'],[768],1)
# print(x[0])
# file_name = "hotquestion1.csv"
# with open(file_name, 'r', encoding='ansi', errors='surrogatepass') as all:
#         df = pd.read_csv(all)
#         y = list()
#         cnt = 0
#         filter_cnt = 0
#         f = 0
#         ids = list()
#         for idx, row in df.iterrows():
#             try:
#                 qid = row['post_id']
#                 y1 = row['pipe_cate']
#                 ids.append(qid)
#                 y.append(y1)
#             except Exception as e:
#                 print("Skip qid %s because %s" % (qid, e))
#                 filter_cnt += 1
# x_train, x_test, y_train, y_test , id_train, id_test= train_test_split(x, y, ids, test_size=0.2)


#     clf = st.DecisionTreeClassifier(max_depth=4).fit(x_train, y_train)
# model = KNeighborsClassifier()
# model = LogisticRegression(penalty='l2')
# model = GradientBoostingClassifier(n_estimators=200)
# model = GaussianNB()
# model = LinearDiscriminantAnalysis()
# model = QuadraticDiscriminantAnalysis()
# model = LinearSVC(probability=True)
model = SVC(kernel='linear',probability=True)
model.fit(x_train, y_train)
#     clf = MultinomialNB().fit(x_train, y_train)
# clf = RandomForestClassifier().fit(x1, y1)
predict = model.predict(x_test)
pre = model.predict_proba(x_test)
print(len(predict))
print(len(pre))
corpus_header = ["id", "phase", "p0","p1","p2","p3","p4","p5"]
cnt = 0
with open(fpath2, 'wt',newline='') as out:
    wr = csv.writer(out)
    wr.writerow(corpus_header)
    length = len(qid_test)
    cnt = 0
    while(cnt<length):
        phase = predict[cnt]
        qid = qid_test[cnt]
        vector = np.asarray(x_test[cnt])
        p0 = pre[cnt][0]
        p1 = pre[cnt][1]
        p2 = pre[cnt][2]
        p3 = pre[cnt][3]
        p4 = pre[cnt][4]
        p5 = pre[cnt][5]
            # title = row[1]
            # desc = row[2]
            # code = row[3]
            # creation_date = row[4]
            # tags = row[5]
        cnt += 1

        try:
            wr.writerow([qid, phase, p0,p1,p2,p3,p4,p5])
                     # title, desc, code, creation_date, tags])
        
        except Exception as e:
            print("Skip id=%s" % qid)
            print("Error msg: %s" % e)
        if cnt % 10000 == 0:
            print("Processing %s row..." % cnt)

# print(accuracy_score(y_train, y_test))
# print(confusion_matrix(y_train,y_test))


# file2 = "result.txt"
# file1 = "vector.txt"
# i = 0
# with open(file2,'w') as wf:
#     while(i<len(pre)):
#         wf.write(str(predict[i]))
#         wf.write(' ')
#         wf.write(str(pre[i][predict[i]]))
#         wf.write(' ')
#         wf.write(str(id_test[i]))
#         wf.write('\n')
#         i = i + 1
