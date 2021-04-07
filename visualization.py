import pickle
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.preprocessing import Normalizer

path = "./data/douban_music/"

composer_dic = pickle.load(open(path + "composer_dic.pkl", mode='rb'))
audience_dic = pickle.load(open(path + "audience_dic.pkl", mode='rb'))

X_composer = [
    'Michael Jackson',
    'Avril Lavigne',
    'Lady Gaga',
    'Justin Bieber',
    'Leona Lewis',
    'Lana Del Rey',
    'Whitney Houston',
    'Taylor Swift',
    'Tori Amos',
    'Mary J Blige',
    'Tom Waits',
    'Sam Smith',
    'Marilyn Manson',
    'Mariah Carey',
    'YUI',
    'Neil Young',
    'Keith Urban',
    'Justin Timberlake',
    'John Mayer',
    'Jason Mraz'
]

X_cid = [composer_dic[c] for c in X_composer]

composer_embedding = pickle.load(open(path + "composer_embedding.pkl", mode='rb'))
style_embedding = pickle.load(open(path + "style_embedding.pkl", mode='rb'))
X_c_ = composer_embedding[X_cid]
X_style = style_embedding

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))  #
    return e_x / e_x.sum(axis=1, keepdims=1)

X_c_s = softmax(X_c_.dot(X_style.T))

# np.random.seed(0)
label_dict = pickle.load(open(path + "label_dic.pkl", 'rb'))
label_name = [0] * len(label_dict)
for k, v in label_dict.items():
    label_name[v] = k
# label_dict = [item for item in label_name]
# data = pickle.load(open("label_embedding.pkl", 'rb'))
# data = data.cpu().detach().numpy().T
#
# scaler = Normalizer(norm='l2')
# scaler.fit(data)
# data = scaler.transform(data)
# XX= np.dot(data,data.T)

print(X_c_s.shape)


sns.set()
ax = sns.heatmap(X_c_s, square=True, cmap='Blues', vmin=0, vmax=0.4)
#设置坐标字体方向

ax.set_xticklabels( label_name)
ax.set_yticklabels(X_composer)

label_y = ax.get_yticklabels()
plt.setp(label_y, rotation=360, horizontalalignment='right')
label_x = ax.get_xticklabels()
plt.setp(label_x, rotation=90, horizontalalignment='center')

plt.show()




