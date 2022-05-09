# -*- coding: utf-8 -*-
import csv
import os
import pandas as pd
import ast
import argparse
import torch
import datetime

# from pathConfig import data_dir
from torchviz import make_dot
from utils.pkl_util import load_pickle
from utils.time_util import get_current_time
from post2vec_util import load_model, load_args
from utils.vocab_util import vocab_to_index_dict
from data_structure.question import Question

task = 'tagRec'
dataset = "SO-05-Sep-2018"
dataset_dir = 'tasks' + os.sep + task + os.sep + dataset
# ts dir
ts = 50
ts_dir = dataset_dir + os.sep + "ts%s" % ts
# sample_K dir
sample_K = "test100000"
sample_K_dir = ts_dir + os.sep + "data-%s" % sample_K
vocab_dir = os.path.join(sample_K_dir, "vocab")

# input files
# len_dict_fpath = os.path.join(vocab_dir, "len.pkl")
title_vocab_fpath = os.path.join(vocab_dir, "title_vocab.pkl")
desc_text_vocab_fpath = os.path.join(vocab_dir, "desc_text_vocab.pkl")
desc_code_vocab_fpath = os.path.join(vocab_dir, "desc_code_vocab.pkl")
tag_vocab_fpath = os.path.join(vocab_dir, "tag_vocab.pkl")

# basic path
train_dir = sample_K_dir + os.sep + "train"
test_dir = sample_K_dir + os.sep + "test"
print("Setting:\ntasks : %s\ndataset : %s\nts : %s\n" % (task, dataset, ts))
parser = argparse.ArgumentParser(description='Multi-label Classifier based on Multi-component')
# basic settings
parser.add_argument('-batch-size', type=int, default=128, help='batch size for training [default: 128]')
parser.add_argument('-epochs', type=int, default=16, help='number of epochs for train [default: 24]')
parser.add_argument('-log-interval', type=int, default=10,
                    help='how many steps to wait before logging [default: 10]')
parser.add_argument('-dev-ratio', type=float, default=0.0, help='ratio of development set')
parser.add_argument('-dev-interval', type=int, default=500,
                    help='how many steps to wait before testing [default: 1000]')
parser.add_argument('-dev-metric', type=str, default='ori', help='evaluation metric for development set')
parser.add_argument('-dev-metric-topk', type=list, default=[1, 2, 3, 4, 5], help='topk for development set')
parser.add_argument('-test-interval', type=int, default=1000,
                    help='how many steps to wait before testing [default: 1000]')
parser.add_argument('-early-stop', type=int, default=1000,
                    help='iteration numbers to stop without performance increasing')
parser.add_argument('-save-interval', type=int, default=1000,
                    help='how many steps to wait before saving [default:1000]')
parser.add_argument('-save-dir', type=str, default='snapshot', help='where to save the snapshot')
parser.add_argument('-save-best', type=bool, default=True, help='whether to save when get best performance')
# data
parser.add_argument('-shuffle', action='store_true', default=False, help='shuffle the data every epoch')
# model
parser.add_argument('-static', action='store_true', default=False, help='fix the embedding')
# device
parser.add_argument('-device', type=int, default=-1,
                    help='device to use for iterate data, -1 mean cpu [default: -1]')
parser.add_argument('-no-cuda', action='store_true', default=False, help='disable the gpu')
# option
parser.add_argument('-snapshot', type=str, default=None, help='filename of model snapshot [default: None]')

############################# default parameter #############################
parser.add_argument('-title-kernel-num', type=int, default=100, help='number of each kind of kernel')  # 100
parser.add_argument('-desc-text-kernel-num', type=int, default=100, help='number of each kind of kernel')  # 100
parser.add_argument('-desc-code-kernel-num', type=int, default=100, help='number of each kind of kernel')  # 100
parser.add_argument('-title-kernel-sizes', type=list, default=[1, 2, 3],
                    help='comma-separated kernel size to use for convolution')
parser.add_argument('-desc-text-kernel-sizes', type=list, default=[1, 2, 3],  # 1,2,3 or 2,3,4 or 3,4,5
                    help='comma-separated kernel size to use for convolution')
parser.add_argument('-desc-code-kernel-sizes', type=list, default=[2, 3, 4],  # 1,2,3 or 2,3,4 or 3,4,5
                    help='comma-separated kernel size to use for convolution')
############################################################################

############################# tuned parameter #############################
parser.add_argument('-model-selection', type=str, default='separate_all_cnn',
                    help='model selection [default: separate_all_cnn]')  # separate_title_desctext_cnn
parser.add_argument('-lr', type=float, default=0.0001, help='initial learning rate [default: 0.001]')
parser.add_argument('-dropout', type=float, default=0.5, help='the probability for dropout [default: 0.5]')
parser.add_argument('-embed-dim', type=int, default=128,
                    help='number of embedding dimension [default: 128]')
parser.add_argument('-hidden-dim', type=int, default=512,
                    help='number of hidden dimension of fully connected layer [default: 512]')
############################################################################


args = parser.parse_args()
len_dict = dict()
len_dict["max_title_len"] = 100
len_dict["max_desc_text_len"] = 1000
len_dict["max_desc_code_len"] = 1000
args.max_title_len = len_dict["max_title_len"]
args.max_desc_text_len = len_dict["max_desc_text_len"]
args.max_desc_code_len = len_dict["max_desc_code_len"]
# title vocab
title_vocab = load_pickle(title_vocab_fpath)
title_vocab = vocab_to_index_dict(vocab=title_vocab, ifpad=True)
args.title_embed_num = len(title_vocab)

# desc_text vocab
desc_text_vocab = load_pickle(desc_text_vocab_fpath)
desc_text_vocab = vocab_to_index_dict(vocab=desc_text_vocab, ifpad=True)
args.desc_text_embed_num = len(desc_text_vocab)

# desc_code_vocab
desc_code_vocab = load_pickle(desc_code_vocab_fpath)
desc_code_vocab = vocab_to_index_dict(vocab=desc_code_vocab, ifpad=True)
args.desc_code_embed_num = len(desc_code_vocab)

# tag vocab
tag_vocab = load_pickle(tag_vocab_fpath)
tag_vocab = vocab_to_index_dict(vocab=tag_vocab, ifpad=False)
args.class_num = len(tag_vocab)

# Device configuration
args.cuda = (not args.no_cuda) and torch.cuda.is_available()
del args.no_cuda

snap_shot_dir = os.path.join(sample_K_dir, "approach", "post2vec", "snapshot-train", "cnn")
if not os.path.exists(snap_shot_dir):
    os.makedirs(snap_shot_dir)
args.save_dir = os.path.join(snap_shot_dir,
                             args.model_selection + "#" + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))


def get_post_vec(raw_qlist):
    from utils.padding_and_indexing_util import padding_and_indexing_qlist_without_tag
    import copy

    model, len_dict, title_vocab, desc_text_vocab, desc_code_vocab = load_p2v_model_and_vocab()

    # get corresponding vector
    print("Get corresponding vector...", get_current_time())
    raw_qlist = copy.deepcopy(raw_qlist)

    # y = model(t=torch.rand(100, 1000).long(), dt=torch.rand(100, 1000).long(),
    #           dc=torch.rand(100, 1000).long())
    # g = make_dot(y, params=dict(model.named_parameters()))
    # g.view()
    # processed qlist
    qlist = padding_and_indexing_qlist_without_tag(raw_qlist, len_dict, title_vocab, desc_text_vocab, desc_code_vocab)
    qvec = model.get_output_vector(qlist=qlist)

    print("qvec size %s" % len(qvec))

    q_dict = dict()
    for q in qvec:
        q_dict[q.qid] = q

    return q_dict


def load_p2v_model_and_vocab():
    # path setting
    task = 'tagRec'
    dataset = "SO-05-Sep-2018"
    dataset_dir = os.path.join("tasks", task, dataset)
    ts = 50
    ts_dir = os.path.join(dataset_dir, "ts%s" % ts)
    sample_K = "test100000"
    sample_K_dir = os.path.join(ts_dir, "data-%s" % sample_K)
    vocab_dir = os.path.join(sample_K_dir, "vocab")

    # approach setting
    app_name = "post2vec"
    # load param
    app_dir = os.path.join(sample_K_dir, "approach", app_name)
    snapshot_dirname = "separate_all_cnn#2021-12-30_17-15-11"
    app_type = "cnn" if "cnn" in snapshot_dirname else "lstm"
    snapshot_dir = os.path.join(app_dir, "snapshot-train", app_type, snapshot_dirname)
    param_name = "snapshot_steps_%s.pt" % "512000"
    ############################## setting end #########################################

    # load vocab
    # initial
    len_dict_fpath = os.path.join(vocab_dir, "len.pkl")
    title_vocab_fpath = os.path.join(vocab_dir, "title_vocab.pkl")
    desc_text_vocab_fpath = os.path.join(vocab_dir, "desc_text_vocab.pkl")
    desc_code_vocab_fpath = os.path.join(vocab_dir, "desc_code_vocab.pkl")

    # len
    len_dict = load_pickle(len_dict_fpath)

    # title vocab
    title_vocab = load_pickle(title_vocab_fpath)
    title_vocab = vocab_to_index_dict(vocab=title_vocab, ifpad=True)

    # desc_text vocab
    desc_text_vocab = load_pickle(desc_text_vocab_fpath)
    desc_text_vocab = vocab_to_index_dict(vocab=desc_text_vocab, ifpad=True)

    # desc_code_vocab
    desc_code_vocab = load_pickle(desc_code_vocab_fpath)
    desc_code_vocab = vocab_to_index_dict(vocab=desc_code_vocab, ifpad=True)

    print("Processing %s" % param_name)
    best_param_fpath = os.path.join(snapshot_dir, param_name)

    # load approach
    print("Load args and model...", get_current_time())
    model = load_model(args, best_param_fpath)

    return model, len_dict, title_vocab, desc_text_vocab, desc_code_vocab


if __name__ == '__main__':
    fpath = 'processed.csv'
    print("Preprocessing corpus %s" % fpath, get_current_time())
    with open(fpath, 'r', encoding='utf-8', errors='surrogatepass') as all:
        df = pd.read_csv(all)
        q_list = list()
        cnt = 0
        filter_cnt = 0
        for idx, row in df.iterrows():
            try:
                qid = row['id']
                title = ast.literal_eval(row['title'])
                desc_text = ast.literal_eval(row['desc_text'])
                desc_code = ast.literal_eval(row['desc_code'])
                creation_date = row['creation_date']
                tags = ast.literal_eval(row['tags'])
                # remove rare tags

                try:
                    q_list.append(Question(qid, title, desc_text, desc_code, creation_date, tags))
                    cnt += 1
                except Exception as e:
                    print("Skip id=%s" % qid)
                    print("Error msg: %s" % e)

                if cnt % 10000 == 0:
                    print("Writing %d instances, filter %d instances..." % (cnt, filter_cnt), get_current_time())
            except Exception as e:
                print("Skip qid %s because %s" % (qid, e))
                filter_cnt += 1

        result = get_post_vec(q_list)
        with open("vecter2.csv", 'w') as out:
            wr = csv.writer(out)
            for q in result.values():
                qid = q.qid
                title = q.title
                desc = q.desc_text
                code = q.desc_code
                creation_date = q.creation_date
                tags = q.tags
                try:
                    wr.writerow(
                        [qid, title, desc, code, creation_date, tags])
                    cnt += 1
                except Exception as e:
                    print("Skip id=%s" % qid)
                    print("Error msg: %s" % e)
