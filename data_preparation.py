from nltk.corpus import stopwords
from tqdm import tqdm
import pickle
import pandas as pd
from itertools import chain
import numpy as np
import nltk
import os
nltk.download('stopwords')

root = 'data/'


def replace_all(text, dict_):
    """

    Args:
        text (str): string in which particular words are to be replaced
        dict_ (dict): key string is replaced with value string

    Returns:
        text (str): string after replacement

    """
    for i, j in dict_.items():
        text = text.replace(i, j)
    return text


def tokenize(column):
    """

    Args:
        column (list of list): column to be tokenized

    Returns:
        A tokenized, lowercased list of lists
    """
    items = []
    to_replace = {"'s": "", "?": "", ",": "", '-': ' ', ':': '', "(": "", ")": "", "/": " "}
    stop_words = [" " + word + " " for word in stopwords.words('english')]
    to_replace.update(**dict.fromkeys(stop_words, ' '))
    max_len = 0
    for _, item in enumerate(tqdm(column)):
        # Remove unwanted punctuations and stopwords
        filtered_item = replace_all(item, to_replace)
        items.append(filtered_item.strip().lower().split())
        # items[-1] = [word for word in items[-1] if word not in stopwords.words()]
        if len(items[-1]) > max_len:
            max_len = len(items[-1])

    return items, max_len


def padding(data, max_len):
    """

    Args:
        max_len: Length of longest headline
        data: headlines to be padded

    Returns:
        padded headlines
    """
    print(max_len)
    for idx, item in enumerate(data):
        data[idx] += ['<UNK>'] * (max_len - len(item))

    return np.array(data)


def construct_word2vec(train_data):
    """

    Args:
        train_data (list of list):

    Returns:
        Two dictionaries of mappings between words and indices
    """
    word_to_idx_dict = {item: i for i, item in enumerate(set(chain(*train_data)))}
    # word_to_idx_dict['<UNK>'] = len(word_to_idx_dict.keys())
    idx_to_word_dict = {i: word for word, i in word_to_idx_dict.items()}
    return word_to_idx_dict, idx_to_word_dict


def train_test_split(data, dataframe, fraction=0.8):
    """

    Args:
        dataframe:
        data: data to be split
        fraction: fraction for splitting

    Returns:
        training and testing set
    """
    np.random.shuffle(data)
    n_data = len(data)
    x_training_set = titles[:int(fraction * n_data)]
    x_testing_set = titles[int(fraction * n_data):]
    y_training_set = dataframe.up_votes[:int(fraction * n_data)]
    y_testing_set = dataframe.up_votes[int(fraction * n_data):]
    date = df.date_created.astype('category').cat.codes
    author = df.author.astype('category').cat.codes
    category = df.author.astype('category').cat.codes
    additional_train_features = np.stack(
        (date[:int(fraction * n_data)], author[:int(fraction * n_data)], category[:int(fraction * n_data)]), axis=1)
    additional_test_features = np.stack(
        (date[int(fraction * n_data):], author[int(fraction * n_data):], category[int(fraction * n_data):]), axis=1)
    return x_training_set, x_testing_set, y_training_set, y_testing_set, \
        additional_train_features, additional_test_features


def to_word_vector(tokenized_data):
    """

    Args:
        tokenized_data: text data

    Returns:
        word vector
    """
    return np.array([[word2idx[word] if word in word2idx.keys() else word2idx['<UNK>']
                      for word in item] for item in tokenized_data])


def to_title(word_vec):
    """

    Args:
        word_vec: word vector

    Returns:
        raw text
    """
    return [[idx2word[word] if word in idx2word.keys() else idx2word['<UNK>'] for word in item] for item in word_vec]


def save_pickle(file, filename):
    """

    Args:
        file: file to be saved
        filename (str): path of the file to be saved

    Returns:

    """
    with open(filename, 'wb') as f:
        pickle.dump(file, f)


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


df = pd.read_csv('data/Eluvio_DS_Challenge.csv')

titles, len_longest_sentence = tokenize(df.title)
padded_titles = padding(titles, len_longest_sentence)

x_train_set, x_test_set, y_train_set, y_test_set, extra_train_features, extra_test_features = train_test_split(titles,
                                                                                                               df)
word2idx, idx2word = construct_word2vec(x_train_set)

save_pickle(word2idx, os.path.join(root, 'word2idx.pkl'))
save_pickle(idx2word, os.path.join(root, 'idx2word.pkl'))

x_train_set = to_word_vector(x_train_set)
x_test_set = to_word_vector(x_test_set)

x_train_set = np.concatenate((x_train_set, extra_train_features), axis=1)
x_test_set = np.concatenate((x_test_set, extra_test_features), axis=1)
save_pickle(x_train_set, os.path.join(root, 'x_train.pkl'))
save_pickle(x_test_set, os.path.join(root, 'x_test.pkl'))
save_pickle(y_train_set.values, os.path.join(root, 'y_train.pkl'))
save_pickle(y_test_set.values, os.path.join(root, 'y_test.pkl'))
