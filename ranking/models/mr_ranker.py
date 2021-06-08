import torch, random
import numpy as np
from torch.autograd import Variable
from models.base_ranker import BaseRanker


class MRRanker(BaseRanker):
    def __init__(self, epochs, lr, device):
        self.device = device
        super().__init__(epochs, lr)

    def train(self, all_features, all_labels):
        train_x_1, train_x_2, train_y = self._get_pairwise_features(all_features, all_labels)

        self.set_model(train_x_1.size(1))
        self.model.to(self.device)
        self.model.training = True

        loss_fn = torch.nn.MarginRankingLoss(margin=1.0)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        print("Started training.")

        batch_size = 4096 * 16
        for epoch in range(self.epochs):
            print("Final data size", train_x_1.size(0), train_x_1.size(1))
            permutation = torch.randperm(train_x_1.size(0))
            for i in range(0, train_x_1.size(0), batch_size):
                indices = permutation[i:i + batch_size]
                y_pred_1 = self.model(train_x_1[indices])
                y_pred_2 = self.model(train_x_2[indices])
                loss = loss_fn(y_pred_1, y_pred_2, train_y[indices])

                print("Epoch ", epoch, "Loss", loss.data.cpu().numpy().tolist())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        print("Done training.")
        self.model.eval()
        self.model.to("cpu")

    def _get_pairwise_features(self, all_features, all_labels):
        train_labels = []
        train_features_1, train_features_2 = [], []

        for feats, ls in zip(all_features, all_labels):
            for i, sf1 in enumerate(feats):
                for j, sf2 in enumerate(feats):
                    if abs(ls[i] - ls[j]) > 0.1:
                        train_features_1.append(sf1)
                        train_features_2.append(sf2)
                        train_labels.append(float(np.sign(ls[i] - ls[j])))


        self.binner.fit(np.array(train_features_1))
        train_features_1 = self.binner.transform(np.array(train_features_1))
        train_features_2 = self.binner.transform(np.array(train_features_2))
        assert len(train_labels) == len(train_features_1) == len(train_features_2)
        print("Pairwise data size: ", len(train_features_2))

        train_x_1 = Variable(torch.FloatTensor(train_features_1).to(self.device))
        train_x_2 = Variable(torch.FloatTensor(train_features_2).to(self.device))
        train_y = Variable(torch.FloatTensor(train_labels).to(self.device), requires_grad=False)
        train_y = torch.unsqueeze(train_y, 1)
        return  train_x_1, train_x_2, train_y
