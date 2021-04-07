import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from GCN import GCN
from NeuralNetwork import NeuralNetwork


class HALPN(NeuralNetwork):

    def __init__(self, config):
        super(HALPN, self).__init__()
        self.best_f1 = 0
        self.patience = 0
        self.config = config
        self.bsz = config['batch_size']
        self.n_heads = config['n_heads']
        self.beta = config['beta']

        self.R_cm = self.np_coo_to_torch_coo(config['R_cm'])
        self.R_am = self.np_coo_to_torch_coo(config['R_am'])

        self.music_embedding = nn.Embedding(config['R_cm'].shape[1], config['embeding_size'])
        self.composer_embedding = nn.Embedding(config['R_cm'].shape[0], config['embeding_size'])
        self.audience_embedding = nn.Embedding(config['R_am'].shape[0], config['embeding_size'], padding_idx=0)
        self.style_embedding = nn.Embedding(config["C"], config['embeding_size'])

        self.Wcm = [nn.Parameter(torch.FloatTensor(100, 100)).cuda() for _ in range(self.n_heads)]
        self.Wam = [nn.Parameter(torch.FloatTensor(100, 100)).cuda() for _ in range(self.n_heads)]
        self.Wma = nn.Parameter(torch.FloatTensor(100, 100))

        self.W1 = nn.Parameter(torch.FloatTensor(100*self.n_heads, 100))
        self.W2 = nn.Parameter(torch.FloatTensor(100*self.n_heads, 100))

        self.gcn = GCN(adj_matrix=config['R_style'])
        self.dropout = nn.Dropout(config['dropout'])
        self.relu = nn.ReLU()
        self.elu = nn.ELU()

        self.fc1 = nn.Linear(100, 100)
        self.fc2 = nn.Linear(100, config["C"])
        self.init_weights()
        print(self)


    def init_weights(self):
        init.xavier_normal_(self.music_embedding.weight)
        init.xavier_normal_(self.composer_embedding.weight)
        init.xavier_normal_(self.audience_embedding.weight)
        for i in range(self.n_heads):
            init.xavier_normal_(self.Wcm[i])
            init.xavier_normal_(self.Wam[i])

        init.xavier_normal_(self.Wma)
        init.xavier_normal_(self.W1)
        init.xavier_normal_(self.W2)
        init.xavier_normal_(self.fc1.weight)


    def np_coo_to_torch_coo(self, X):
        data = torch.FloatTensor(X.data)
        edge = torch.LongTensor([X.row, X.col])
        transformed_tensor = torch.sparse.FloatTensor(edge, data, torch.Size([X.shape[0], X.shape[1]])).cuda()
        return transformed_tensor

    def composer_multi_head(self, X_composer, X_composer_id, Wcm):
        M = self.music_embedding.weight
        linear1 = torch.tanh(torch.einsum("md,dd,bd->mb", M, Wcm, X_composer))  # m x bsz
        R_cm = self.R_cm.to_dense().permute(1, 0)
        tmp = R_cm[:, X_composer_id]
        alpha = F.softmax(tmp * linear1, dim=-1) # m x bsz
        alpha = self.dropout(alpha)
        return torch.einsum("mb,md->bd", alpha, M)

    def composer_encoder(self, X_composer, X_composer_id):
        m_hat = []
        for i in range(self.n_heads):
            m_hat.append(self.composer_multi_head(X_composer, X_composer_id, self.Wcm[i]))
        m_hat = self.dropout(torch.cat(m_hat, dim=-1))
        m_hat = self.elu(m_hat).matmul(self.W1)
        c_hat = m_hat + X_composer # bsz x d
        return c_hat


    def audience_multi_head(self, X_audience, X_audience_id, Wam):
        M = self.music_embedding.weight
        linear1 = torch.tanh(torch.einsum("bnd,dd,md->bnm", X_audience, Wam, M))  # m x bsz
        R_am = self.R_am.to_dense().permute(1, 0)
        R_am = R_am[:, X_audience_id].permute(1, 2, 0)
        alpha = F.softmax(R_am * linear1, dim=-1) # bsz x 40 x m
        alpha = self.dropout(alpha)
        return torch.einsum("bnm,md->bnd", alpha, M)


    def audience_encoder(self, X_audience, X_audience_id):
        m_hat = []
        for i in range(self.n_heads):
            m_hat.append(self.audience_multi_head(X_audience, X_audience_id, self.Wam[i]))

        m_hat = self.dropout(torch.cat(m_hat, dim=-1))
        m_hat = self.elu(m_hat).matmul(self.W2)
        a_hat = m_hat + X_audience # bsz x 40 x d
        return a_hat

    def music_representation(self, X_music, composer_rep, audiences_rep):
        alpha = F.softmax(torch.einsum("bd,dd,bnd->bn", X_music, self.Wma, audiences_rep))
        aud_rep = torch.einsum("bn,bnd->bd", alpha, audiences_rep)

        music_rep = self.beta * composer_rep + (1 - self.beta)* aud_rep
        return music_rep


    def forward(self, X_music_id, X_composer_id, X_audience_id):  # , X_composer_id, X_reviewer_id
        '''
        :param X_text size: (batch_size, max_sents, max_words)
        :return:
        '''
        X_music = self.music_embedding(X_music_id)
        X_composer = self.composer_embedding(X_composer_id)
        X_audience = self.audience_embedding(X_audience_id)

        style_rep = self.gcn(self.style_embedding.weight)  # nclass x d
        composer_rep = self.composer_encoder(X_composer, X_composer_id)
        audience_rep = self.audience_encoder(X_audience, X_audience_id)
        music_rep = self.music_representation(X_music, composer_rep, audience_rep)

        d1 = self.relu(self.fc1(music_rep)) # self.dropout()
        Xt_logit = torch.einsum("bd,dc->bc", d1, style_rep).sigmoid()
        # Xt_logit = self.fc2(self.dropout(d1)).sigmoid()
        return Xt_logit




