from torch.utils.data import DataLoader
from news_data import NewsData
import torch
import os
from tqdm import tqdm
import torch.nn as nn
from model import Net
import pickle

root = 'data/'

# device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyperparameters
num_epochs = 100
emb_dim = 30
input_size = 52 * emb_dim + 3       # max_len * embedding dim + 3
learning_rate = 0.01


def load_pickle(filename):
    """

    Args:
        filename (str): name of the pickle file to be loaded

    Returns:
        file:

    """
    with open(filename, 'rb') as f:
        file = pickle.load(f)
    return file


if __name__ == '__main__':
    train_dataset = NewsData(features=os.path.join(root, 'x_train.pkl'), labels=os.path.join(root, 'x_test.pkl'))
    word2idx = load_pickle(os.path.join(root, 'word2idx.pkl'))
    idx2word = load_pickle(os.path.join(root, 'idx2word.pkl'))
    vocab_size = len(word2idx.keys())

    train_loader = DataLoader(train_dataset, batch_size=512)

    # Loss and optimizer
    loss = nn.MSELoss()
    model = Net(emb_dim, input_size, vocab_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    for epoch in range(num_epochs):
        total_cost = 0
        for n, sample in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()
            features, label = sample['feature'], sample['label']
            features = features.to(device)
            output = model(features)
            cost = loss(label, output)
            cost.backward()
            total_cost += cost
            optimizer.step()
            torch.cuda.empty_cache()

        # save the model at every 10 epochs
        if epoch % 10 == 0:
            torch.save(model.state_dict(), 'Models/model.pt')

        print('Epoch: {}    total_cost =  {}   '.format(epoch, total_cost))
    torch.save(model.state_dict(), 'Models/model.pt')
