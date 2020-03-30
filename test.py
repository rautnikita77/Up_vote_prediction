from torch.utils.data import DataLoader
import torch
from model import Net
import os
from tqdm import tqdm
from news_data import NewsData
import torch.nn as nn

root = ''
saved_model_path = 'Models/model.pt'

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyperparameters
emb_dim = 30
input_size = 52 * emb_dim + 3
vocab_size = 125954
learning_rate = 0.0001

if __name__ == '__main__':

    test_dataset = NewsData(features=os.path.join(root, 'y_train.pkl'), labels=os.path.join(root, 'y_test.pkl'))
    test_loader = DataLoader(test_dataset)
    model = Net(emb_dim, input_size, vocab_size).to(device)
    model.load_state_dict(torch.load('Model/model.pt'), map_location=device)

    loss = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    error = 0
    with torch.no_grad():
        for n, sample in enumerate(tqdm(test_loader)):
            features, label = sample['feature'], sample['label']
            features = features.to(device)
            output = model(features)
            error += loss(label, output)

        print('Error: {}'.format(error))
