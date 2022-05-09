import regex as re
import itertools
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import csv

# bert_transformers = SentenceTransformer("sentence-transformers/all-MiniLM-L12-v2")
bert_transformers = SentenceTransformer('all-distilroberta-v1')

def mean(z): # used for BERT (word version) and Word2Vec
    return sum(itertools.chain(z))/len(z)

def embeddToBERT(text):
    sentences = re.split('!|\?|\.',text)
    sentences = list(filter(None, sentences))
    result = bert_transformers.encode(sentences)
    feature = [mean(x) for x in zip(*result)]
    return feature

file_name = "all_questions.csv"
with open(file_name, 'r', encoding='utf-8', errors='surrogatepass') as all:
    df = pd.read_csv(all)
    x = list()
    y = list()
    cnt = 0
    filter_cnt = 0
    f = 0
    corpus_header = ["id", "phase","vector"]
    fpath2 = "all_vec4.csv"
    fpath3 = "test.txt"
    with open(fpath2, 'w') as out:
        wr = csv.writer(out)
        wr.writerow(corpus_header)
        for idx, row in df.iterrows():
            try:
                qid = row['post_id']
                text = row['clean_text']
                phase = row['pipe_cate']
                bert_sentence_training_features = embeddToBERT(text)
                #print(bert_sentence_training_features)
               #feature = [x for x in bert_sentence_training_features.transpose()]
                bert_sentence_training_features = np.asarray(bert_sentence_training_features)
                #print(bert_sentence_training_features.shape)
                try:
                    wr.writerow([qid, phase,str(bert_sentence_training_features)])
                    # title, desc, code, creation_date, tags])
                    cnt += 1
                except Exception as e:
                    print("Skip id=%s" % qid)
                    print("Error msg: %s" % e)
                if cnt % 1000 == 0:
                    print("Processing %s row..." % cnt)

                # title = row['title'].replace('\n', ' ').replace('[', ' ').replace(']', ' ').split()
                # text = row['desc_text'].replace('\n', ' ').replace('[', ' ').replace(']', ' ').split()
                # code = row['desc_code'].replace('\n', ' ').replace('[', ' ').replace(']', ' ').split()
                # x2 = numpy.array(title+text+code)
            except Exception as e:
                print("Skip qid %s because %s" % (qid, e))
                filter_cnt += 1
  





