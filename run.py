import os
import time
import pickle
import torch
import numpy as np
from sklearn.metrics import f1_score
# from PGAN import PGAN
from HALPN import HALPN
from metrics import Metrics

os.environ['CUDA_ENABLE_DEVICES'] = '0'


def loadData(task):
    path = "./data/" + task + "/"
    R_cm, R_am, R_style = pickle.load(open(path + "Relation.pkl", mode='rb'))
    X_train_label, X_train_music, X_train_composer, X_train_audience = pickle.load(open(path + "X_train.pkl", mode='rb'))
    X_dev_label,   X_dev_music, X_dev_composer, X_dev_audience = pickle.load(open(path + "X_dev.pkl", mode='rb'))
    X_test_label,  X_test_music, X_test_composer, X_test_audience = pickle.load(open(path + "X_test.pkl", mode='rb'))
    config["C"] = 22 if task == 'douban_music' else 20
    config['R_cm'] = R_cm
    config['R_am'] = R_am
    config['R_style'] = R_style
    config['num_composers'] = R_cm.shape[0]
    config['num_audiences'] = R_am.shape[0]

    return  X_train_label, X_train_music, X_train_composer, X_train_audience, \
            X_dev_label,   X_dev_music, X_dev_composer, X_dev_audience, \
            X_test_label, X_test_music, X_test_composer, X_test_audience


def train_and_test(model_class, task):
    model_suffix = model_class.__name__.lower().strip("text")
    config['save_path'] = 'checkpoints/weights.best.'+ model_suffix

    X_train_label, X_train_music, X_train_composer, X_train_audience, \
    X_dev_label,   X_dev_music, X_dev_composer, X_dev_audience, \
    X_test_label,  X_test_music, X_test_composer, X_test_audience = loadData(task)

    model = model_class(config)
    model.fit(X_train_label, X_dev_label,
              X_train_music, X_dev_music,
              X_train_composer, X_dev_composer,
              X_train_audience, X_dev_audience)

    pickle.dump(model.composer_embedding.weight.data.cpu().numpy(),
                file=open("data/"+task+"/composer_embedding.pkl", 'wb'), protocol=4)

    pickle.dump(model.audience_embedding.weight.data.cpu().numpy(),
                file=open("data/"+task+"/audience_embedding.pkl", 'wb'), protocol=4)

    pickle.dump(model.style_embedding.weight.data.cpu().numpy(),
                file=open("data/"+task+"/style_embedding.pkl", 'wb'), protocol=4)

    print("================================================")
    model.load_state_dict(state_dict=torch.load(config['save_path']))
    y_pred, y_pred_top = model.predict(X_test_music, X_test_composer, X_test_audience)

    metric = Metrics()
    metric.calculate_all_metrics(X_test_label, y_pred, y_pred_top)
    if task == 'douban_music':
        X_test_label = np.array(X_test_label)
        y_pred = np.array(y_pred)
        F1_top1 = f1_score(X_test_label, y_pred, labels=[1], average = 'micro')
        F1_top2 = f1_score(X_test_label, y_pred, labels=[0], average = 'micro')
        F1_top3 = f1_score(X_test_label, y_pred, labels=[2], average = 'micro')
        F1_top4 = f1_score(X_test_label, y_pred, labels=[3], average = 'micro')
        F1_top5 = f1_score(X_test_label, y_pred, labels=[4], average = 'micro')

        F1_few1 = f1_score(X_test_label, y_pred, labels=[13], average='micro')
        F1_few2 = f1_score(X_test_label, y_pred, labels=[17], average='micro')
        F1_few3 = f1_score(X_test_label, y_pred, labels=[18], average='micro')
        F1_few4 = f1_score(X_test_label, y_pred, labels=[20], average='micro')
        F1_few5 = f1_score(X_test_label, y_pred, labels=[21], average='micro')

        print('--'*20)
        print("F1_top1: ", F1_top1)
        print("F1_top2: ", F1_top2)
        print("F1_top3: ", F1_top3)
        print("F1_top4: ", F1_top4)
        print("F1_top5: ", F1_top5)
        print('--' * 20)
        print("F1_few1: ", F1_few1)
        print("F1_few2: ", F1_few2)
        print("F1_few3: ", F1_few3)
        print("F1_few4: ", F1_few4)
        print("F1_few5: ", F1_few5)


config = {
    'n_heads': 4,
    'beta': 0.4,
    'lr':1e-3,
    'reg':0,
    'batch_size':64,
    'dropout': 0.5,
    'embeding_size':100,
    'epochs':40,
}


if __name__ == '__main__':
    model = HALPN
    task = "douban_music"  # "amazon_music"  douban_music

    if task == "amazon_music":
        config['beta'] = 0.5
        config['n_heads'] = 3

    start = time.time()
    train_and_test(model, task=task)
    end = time.time()

    print("use time: ", (end-start)/60, "min" )

