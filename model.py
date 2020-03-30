import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Net(nn.Module):

    def __init__(self, emb_dim, input_size, vocab_size):
        """
        Args:
            emb_dim (int): Embedding dimension size
            input_size (int): input size
            vocab_size (int): Size of vocab
        """
        super(Net, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.fc1 = nn.Linear(input_size, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, 1)

    def forward(self, x):
        """
        Args:
            x(tensor): Input features
        Returns:
            predicted upvotes
        """
        x1 = self.embedding(x[:, :-3])
        x2 = x[:, -3:]
        x = torch.cat((x1.view(-1, x1.size(1) * x1.size(2)), x2.float()), dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
