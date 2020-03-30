from torch.utils.data import DataLoader
import torch
from model import Net
import os
from tqdm import tqdm
from news_data import NewsData
import torch.nn as nn
from sklearn.metrics import r2_score
import pickle

root = 'data/'
saved_model_path = 'Models/model.pt'

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyperparameters
emb_dim = 30
input_size = 52 * emb_dim + 3
vocab_size = 125954
learning_rate = 0.0001

if __name__ == '__main__':

    test_dataset = NewsData(features=os.path.join(root, 'x_test.pkl'), labels=os.path.join(root, 'y_test.pkl'))
    test_loader = DataLoader(test_dataset, batch_size=512)
    with open(os.path.join(root, 'word2idx.pkl'), 'rb') as f:
        word2idx = pickle.load(f)
    vocab_size = len(word2idx.keys())
    model = Net(emb_dim, input_size, vocab_size).to(device)
    model.load_state_dict(torch.load('Models/model.pt'))
    model.eval()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    acc = []
    with torch.no_grad():
        for n, sample in enumerate(tqdm(test_loader)):
            features, label = sample['feature'], sample['label']
            features = features.to(device)
            output = model(features)
            acc.append(r2_score(label, output))

        print('Error: {}'.format(sum(acc)/len(acc)))
