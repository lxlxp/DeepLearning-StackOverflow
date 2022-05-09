import pandas as pd
import numpy as np
import csv

file_name = "result5.csv"
fpath2 = "phase5.csv"
def get_mean(x):
    x = np.array(x)
    #print(x)
    return np.mean(x)

with open(file_name, 'r', encoding='utf-8', errors='surrogatepass') as all:
    df = pd.read_csv(all)
    x = list()
    x0 = list()
    x1 = list()
    x2 = list()
    x3 = list()
    x4 = list()
    x5 = list()
    cnt = 0
    filter_cnt = 0
    f = 0
    corpus_header = ["phase","probability"]
    
    with open(fpath2, 'wt',newline='') as out:
        wr = csv.writer(out)
        wr.writerow(corpus_header)
        for idx, row in df.iterrows():
            try:
                qid = row['id']
                if qid == 0 :
                    continue
                p0 = row['p0']
                p1 = row['p1']
                p2 = row['p2']
                p3 = row['p3']
                p4 = row['p4']
                p5 = row['p5']
                # title = row['title'].replace('\n', ' ').replace('[', ' ').replace(']', ' ').split()
                # text = row['desc_text'].replace('\n', ' ').replace('[', ' ').replace(']', ' ').split()
                # code = row['desc_code'].replace('\n', ' ').replace('[', ' ').replace(']', ' ').split()
                # x1 = numpy.array(title+text+code)
                phase = row['phase']
                    # y1 = row['state']
                if phase == 0:
                    wr.writerow([0,p0])
                    wr.writerow([1,p1])
                    wr.writerow([2,p2])
                    wr.writerow([3,p3])
                    wr.writerow([4,p4])
                    wr.writerow([5,p5])
            except Exception as e:
                print("Skip qid %s because %s" % (qid, e))
                filter_cnt += 1