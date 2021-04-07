import pickle
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from torch.utils.data.dataloader import DataLoader
from metrics import Metrics

torch.backends.cudnn.deterministic = True
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

class NeuralNetwork(nn.Module):

    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.metrics = Metrics()
        self.best_metric = 0

    def forward(self):
        raise NotImplementedError

    def train_step(self, i, data):
        with torch.no_grad():
            batch_y, batch_x_music, batch_x_composer, batch_x_audience = (elem.cuda() for elem in data)

        self.optimizer.zero_grad()
        Xt_logit = self.forward(batch_x_music, batch_x_composer, batch_x_audience)
        loss = self.loss_func(Xt_logit, batch_y)
        loss.backward()
        self.optimizer.step()

        print('Batch[{}] - loss: {:.6f}'.format(i, loss.item()))
        return loss


    def fit(self, y_train, y_val,
            X_train_music, X_dev_music,
            X_train_composer=None, X_dev_composer=None,
            X_train_audience=None, X_dev_audience=None):

        if torch.cuda.is_available():
            self.cuda()

        batch_size = self.config['batch_size']
        X_train_music = torch.LongTensor(X_train_music)
        X_train_composer = torch.LongTensor(X_train_composer)
        X_train_audience = torch.LongTensor(X_train_audience)
        y_train = torch.FloatTensor(y_train)

        dataset = TensorDataset(y_train, X_train_music, X_train_composer, X_train_audience)
        dataiter = DataLoader(dataset, shuffle=True, batch_size=batch_size)

        self.loss_func = nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.config['lr'], weight_decay=self.config['reg'])

        for epoch in range(self.config['epochs']):
            print("\nEpoch ", epoch+1,"/", self.config['epochs'])
            avg_loss = 0
            avg_acc = 0

            self.train()
            for i, data in enumerate(dataiter):
                loss = self.train_step(i, data)

                if (i+1) % 50 == 0:
                    self.evaluate(y_val, X_dev_music, X_dev_composer, X_dev_audience)
                    self.train()

                avg_loss += loss.item()
            cnt = y_train.size(0) // batch_size + 1
            print("Average loss:{:.6f} average acc:{:.6f}%".format(avg_loss/cnt, avg_acc/cnt))

            self.evaluate(y_val, X_dev_music, X_dev_composer, X_dev_audience)
            if epoch > 30 and self.patience >= 2 and self.config['lr'] >= 1e-5:
                self.load_state_dict(torch.load(self.config['save_path']))
                self.adjust_learning_rate()
                print("Decay learning rate to: ", self.config['lr'])
                print("Reload the best model...")
                self.patience = 0


    def adjust_learning_rate(self, decay_rate=.5):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * decay_rate
            self.config['lr'] = param_group['lr']


    def evaluate(self, y_val, X_dev_music, X_dev_composer, X_dev_audience):
        y_pred, y_pred_top = self.predict(X_dev_music, X_dev_composer, X_dev_audience)
        OE, HL, MacroF1, MicroF1 = self.metrics.calculate_all_metrics(y_val, y_pred, y_pred_top)
        metric = MicroF1 # + MicroF1 - HL - OE

        if metric > self.best_metric:
            self.best_metric = metric
            self.patience = 0
            torch.save(self.state_dict(), self.config['save_path'])
            print("save model!!!")
        else:
            self.patience += 1
        print("Val set metric:", metric)
        print("Best val set metric:", self.best_metric)


    def predict(self, X_test_music, X_test_composer, X_dev_audience):
        if torch.cuda.is_available():
            self.cuda()

        self.eval()
        X_test_music = torch.LongTensor(X_test_music)
        X_test_composer = torch.LongTensor(X_test_composer)
        X_dev_audience = torch.LongTensor(X_dev_audience)
        dataset = TensorDataset(X_test_music, X_test_composer, X_dev_audience)
        dataiter = DataLoader(dataset, batch_size=64)

        y_pred = []
        y_pred_top = []
        for i, data in enumerate(dataiter):
            batch_x_music, batch_x_composer, batch_x_audience = (elem.cuda() for elem in data)
            logit = self.forward(batch_x_music, batch_x_composer, batch_x_audience)
            predicted = logit > 0.5

            _, predicted_top = torch.max(logit, dim=1)

            y_pred_top += predicted_top.data.cpu().numpy().tolist()
            y_pred += predicted.data.cpu().numpy().tolist()
        return y_pred, y_pred_top

