import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score , roc_auc_score , roc_curve
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
 
 
 
def create_model(d_train , d_test , all_post):
    print("训练样本 = %d" % len(d_train))
    print("测试样本 = %d" % len(d_test))
    vectorizer = TfidfVectorizer(ngram_range=(1,2),min_df=2 ) #tf-idf特征抽取ngram_range=(1,2)
    train_features = []
    train_label = []
    texts = []
    for line in all_post:
        texts = texts + line["clean_text"].split()
    vectorizer.fit_transform(texts)
    for line in d_train:
        feature = vectorizer.transform(line["clean_text"].split()).mean(axis=0)
        train_features.append(np.squeeze(feature.tolist(), 0))
        train_label.append(line["pipe_cate"])
    # print("训练样本特征表长度为 " + str(train_features.shape))
    # print(vectorizer.get_feature_names()[3000:3050]) #特征名展示
    print(train_features[0])

    test_features = []
    test_label = []
    for line in d_train:
        feature = vectorizer.transform(line["clean_text"].split()).mean(axis=0)
        test_features.append(np.squeeze(feature.tolist(), 0))
        test_label.append(line["pipe_cate"])
    # print("测试样本特征表长度为 "+ str(test_features.shape))
    #支持向量机
    #C: 目标函数的惩罚系数C，用来平衡分类间隔margin和错分样本的，default C = 1.0
    svmmodel = SVC(C = 1.0 , kernel= "linear") #kernel：参数选择有rbf, linear, poly, Sigmoid, 默认的是"RBF";
 
    nn = svmmodel.fit(train_features , train_label)
    # predict = svmmodel.score(test_features ,d_test.sku)
    # print(predict)
    pre_test = svmmodel.predict(test_features)
    print(accuracy_score(test_label,pre_test))
    # d_test["pre_skuno"] = pre_test
    # .to_excel("wr60_svm_pre1012.xlsx", index=False)

train = []
test = []
all_post = []
with open("train.csv", 'r', encoding='utf-8', errors='surrogatepass') as all:
    df1 = pd.read_csv(all)
    for idx, row in df1.iterrows():
        clean_text = row["clean_text"]
        cate = row["pipe_cate"]
        id = row["post_id"]
        post = {
            "clean_text" :clean_text,
            "pipe_cate" :cate,
            "post_id":id
        }
        train.append(post)
with open("test.csv", 'r', encoding='utf-8', errors='surrogatepass') as all:
    df2 = pd.read_csv(all)
    for idx, row in df2.iterrows():
        clean_text = row["clean_text"]
        cate = row["pipe_cate"]
        id = row["post_id"]
        post = {
            "clean_text" :clean_text,
            "pipe_cate" :cate,
            "post_id":id
        }
        test.append(post)
        train.append(post)
with open("all_questions.csv", 'r', encoding='utf-8', errors='surrogatepass') as all:
    df3 = pd.read_csv(all)
    i= 0
    for idx, row in df3.iterrows():
        if i < 10000 :
            clean_text = row["clean_text"]
            cate = row["pipe_cate"]
            id = row["post_id"]
            post = {
                "clean_text" :clean_text,
                "pipe_cate" :cate,
                "post_id":id
            }
            all_post.append(post)
        i += 1
        
 
create_model(train, test, all_post)