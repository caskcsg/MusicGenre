import pickle
from collections import Counter
from itertools import chain
import jieba
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
import re
import scipy.sparse as sp
np.random.seed(1)


def cut_doc(task, text):
    text = re.sub("\s+", " ", text)
    if task.__contains__('douban'):
        return list(jieba.cut(text))
    else:
        return text.split()


def read_data_from_txt(rootpath, file):
    X_label = []
    X_composer = []
    X_audience = []
    X_comment = []
    with open(file, 'r', encoding='utf-8') as input:
        for line in input.readlines():
            jobj = eval(line)
            X_label.append(jobj["tags"])
            X_composer.append(jobj["composer_name"])

            X_i_audience = []
            X_i_comments = []
            for j, reviewobj in enumerate(jobj["all_reviews"]):
                if j == 40: break
                audience = reviewobj['audience_id'] if 'audience_id' in reviewobj else reviewobj['name']
                X_i_audience.append(audience)
                X_i_comments.append(cut_doc(rootpath, reviewobj["comment"]))
            X_audience.append(X_i_audience)
            X_comment.append(X_i_comments)

    assert len(X_label) == len(X_composer) == len(X_audience) == len(X_comment), "The first dimension must be same."
    return X_label, X_composer, X_audience, X_comment


def gen_label_dic(X_label, rootpath):
    counter = Counter(chain(*X_label))
    print(counter.items())
    labels = [x[0] for x in counter.most_common()]
    label_dic = {word:i for i, word in enumerate(labels)}
    print("labels dict: ", labels)
    pickle.dump(label_dic, open(rootpath+"label_dic.pkl", 'wb'))
    return label_dic


def read_dataset(rootpath):
    X_train_label, X_train_composer, X_train_audience, X_train_comment = read_data_from_txt(rootpath, rootpath + "train.txt")
    X_dev_label, X_dev_composer, X_dev_audience, X_dev_comment = read_data_from_txt(rootpath, rootpath + "dev.txt")
    X_test_label, X_test_composer, X_test_audience, X_test_comment = read_data_from_txt(rootpath, rootpath + "test.txt")

    X_music = range(len(X_train_label)+len(X_dev_label)+len(X_test_label))

    label_dic = gen_label_dic(X_train_label, rootpath)
    X_train_label = [[label_dic[style] for style in labels] for labels in X_train_label]
    X_dev_label = [[label_dic[style] for style in labels] for labels in X_dev_label]
    X_test_label = [[label_dic[style] for style in labels] for labels in X_test_label]
    R_style = get_style_relation(label_dic, X_train_label)

    composers = X_train_composer + X_dev_composer + X_test_composer
    composers = [x[0] for x in Counter(composers).most_common()]
    composer_dic = {comp:i for i, comp in enumerate(composers)}
    X_train_composer = [composer_dic[c] for c in X_train_composer]
    X_dev_composer = [composer_dic[c] for c in X_dev_composer]
    X_test_composer = [composer_dic[c] for c in X_test_composer]
    pickle.dump(composer_dic, open(rootpath+"composer_dic.pkl", 'wb'))

    audiences = X_train_audience + X_dev_audience + X_test_audience
    audiences = [x[0] for x in Counter(chain(*audiences)).most_common()] #  if x[1] >= 2
    audience_dic = {r:i+1 for i, r in enumerate(audiences)}  #  if r in audience_dic else audience_dic["<PAD>"]
    audience_dic["<PAD>"] = 0
    X_train_audience = [[audience_dic["<PAD>"]]*(40-len(audiences)) + [audience_dic[r] for r in audiences] for audiences in X_train_audience]
    X_dev_audience = [[audience_dic["<PAD>"]]*(40-len(audiences)) + [audience_dic[r] for r in audiences] for audiences in X_dev_audience]
    X_test_audience = [[audience_dic["<PAD>"]]*(40-len(audiences)) + [audience_dic[r] for r in audiences] for audiences in X_test_audience]
    pickle.dump(audience_dic, open(rootpath+"audience_dic.pkl", 'wb'))

    X_composer = X_train_composer + X_dev_composer + X_test_composer
    X_audience = X_train_audience + X_dev_audience + X_test_audience

    R_cm = np.zeros(shape=[len(composer_dic), len(X_music)])
    R_am = np.zeros(shape=[len(audience_dic), len(X_music)])

    for i in range(len(X_music)):
        R_cm[X_composer[i]][X_music[i]] = 1.0
        for a in X_audience[i]:
            R_am[a][X_music[i]] = 1.0

    return X_train_label, X_train_composer, X_train_audience, X_train_comment, \
           X_dev_label, X_dev_composer, X_dev_audience, X_dev_comment, \
           X_test_label, X_test_composer, X_test_audience, X_test_comment, \
           R_cm, R_am, R_style



def construct_normalized_adj(R):
    adj = sp.coo_matrix(R, dtype=np.float32)

    # Row-normalize sparse matrix
    rowsum = np.array(adj.sum(1))
    D_row = np.power(rowsum, -0.5).flatten()
    D_row[np.isinf(D_row)] = 0.
    D_row = sp.diags(D_row)

    colsum = np.array(adj.sum(0))
    D_col = np.power(colsum, -0.5).flatten()
    D_col[np.isinf(D_col)] = 0.
    D_col = sp.diags(D_col)

    return adj.dot(D_col).transpose().dot(D_row).transpose().tocoo()



def get_style_relation(label_dic, X_label):
    C = len(label_dic)
    style_adj = np.zeros(shape=(C, C))
    relation = []
    for labels in X_label:
        for i in range(len(labels)):
            j = i + 1
            while j < len(labels):
                relation.append(str(labels[i]) + "-" + str(labels[j]))
                relation.append(str(labels[j]) + "-" + str(labels[i]))
                j += 1

    rel_con = Counter(relation)
    for rel in rel_con.most_common():
        h, t = rel[0].strip().split("-")
        style_adj[int(h), int(t)] = rel[1]

    return style_adj


def preprocess(rootpath):
    X_train_label, X_train_composer, X_train_audience, X_train_comment, \
    X_dev_label, X_dev_composer, X_dev_audience, X_dev_comment, \
    X_test_label, X_test_composer, X_test_audience, X_test_comment, R_cm, R_am, R_style = read_dataset(rootpath)

    R_cm = construct_normalized_adj(R_cm)
    R_am = construct_normalized_adj(R_am)

    mlb = MultiLabelBinarizer()
    X_train_label = mlb.fit_transform(X_train_label)
    X_dev_label = mlb.transform(X_dev_label)
    X_test_label = mlb.transform(X_test_label)

    X_train_music = range(len(X_train_label))
    X_dev_music = range(len(X_train_label), len(X_train_label) + len(X_dev_label))
    X_test_music = range(len(X_train_label) + len(X_dev_label), len(X_train_label) + len(X_dev_label) + len(X_test_label))

    pickle.dump([R_cm, R_am, R_style], open(rootpath+"Relation.pkl", 'wb'))
    pickle.dump([X_train_label.tolist(), X_train_music, X_train_composer, X_train_audience], open(rootpath+"X_train.pkl", 'wb'))
    pickle.dump([X_dev_label.tolist(),   X_dev_music, X_dev_composer, X_dev_audience], open(rootpath+"X_dev.pkl", 'wb'))
    pickle.dump([X_test_label.tolist(),  X_test_music, X_test_composer, X_test_audience], open(rootpath+"X_test.pkl", 'wb'))


if __name__ == '__main__':
    preprocess("douban_music/")
    preprocess("amazon_music/")


# douban
# dict_items([('ost', 554), ('jazz', 226), ('jpop', 462), ('punk', 285), ('rock', 1502), ('folk', 1103), ('indie', 1511),
# ('postpunk', 120), ('alternative', 505), ('electronic', 698), ('pop', 1352), ('hiphop', 156), ('postrock', 185),
# ('r&b', 590), ('metal', 172), ('newage', 309), ('piano', 343), ('darkwave', 63), ('soul', 219), ('classical', 199),
# ('britpop', 433), ('country', 128)])


# amazon
# dict_items([('classical', 1137), ('rock', 4161), ('electronic', 969), ('pop', 1798), ('blues', 690), ('folk', 838), ('hiphop', 329),
# ('newage', 291), ('r&b', 905), ('jazz', 1074), ('alternative', 2061), ('country', 664), ('soul', 616), ('indie', 482), ('punk', 589),
# ('postpunk', 294), ('metal', 980), ('piano', 20), ('britpop', 77), ('ost', 312)])


