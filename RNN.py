import torch
import pandas as pd
from collections import Counter
from tqdm import tqdm
from torchtext.vocab import vocab as Vocab
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import torchtext

torchtext.disable_torchtext_deprecation_warning()

train_data = pd.read_csv('train/train.tsv', sep='\t', header=None)
train_data.columns = ['Tag', 'Word']

tags = train_data['Tag'].apply(lambda x: x.split())
words = train_data['Word'].apply(lambda x: x.split())

sentences = []
labels = []
for word_list, tag_list in zip(words, tags):
    sentences.append(word_list)
    labels.append(tag_list)


def build_vocab(sentences):
    counter = Counter()
    for sentence in sentences:
        counter.update(sentence)
    return Vocab(counter, specials=["<unk>", "<pad>", "<bos>", "<eos>"])


word_vocab = build_vocab(sentences)
tag_vocab = Vocab(Counter([tag for sublist in labels for tag in sublist]), specials=["<pad>"])

word_vocab.set_default_index(word_vocab["<unk>"])


def data_process(sentences, word_vocab):
    return [torch.tensor([word_vocab[token] for token in sentence], dtype=torch.long) for sentence in sentences]


def labels_process(labels, tag_vocab):
    return [torch.tensor([tag_vocab[tag] for tag in label], dtype=torch.long) for label in labels]


train_tokens_ids = data_process(sentences, word_vocab)
train_labels = labels_process(labels, tag_vocab)


class LSTM(torch.nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super(LSTM, self).__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embed_dim)
        self.lstm = torch.nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = self.fc(x)
        return x


vocab_size = len(word_vocab)
embed_dim = 100
hidden_dim = 256
output_dim = len(tag_vocab)

model = LSTM(vocab_size, embed_dim, hidden_dim, output_dim)

criterion = torch.nn.CrossEntropyLoss(ignore_index=tag_vocab["<pad>"])
optimizer = torch.optim.Adam(model.parameters())


def train_model(train_tokens_ids, train_labels, model, criterion, optimizer, num_epochs=100):
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        for tokens, labels in tqdm(zip(train_tokens_ids, train_labels), total=len(train_tokens_ids)):
            tokens = tokens.unsqueeze(0)
            labels = labels.unsqueeze(0)

            optimizer.zero_grad()
            predictions = model(tokens).squeeze(0)
            predictions = predictions.view(-1, predictions.shape[-1])
            labels = labels.view(-1)
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / len(train_tokens_ids)}')


train_model(train_tokens_ids, train_labels, model, criterion, optimizer, num_epochs=100)

dev_data = pd.read_csv('dev-0/in.tsv', sep='\t', header=None)
test_data = pd.read_csv('test-A/in.tsv', sep='\t', header=None)


def prepare_test_data(data, word_vocab):
    test_sentences = [sentence.split() for sentence in data[0].tolist()]
    test_tokens_ids = data_process(test_sentences, word_vocab)
    return test_tokens_ids, test_sentences


X_dev, dev_sentences = prepare_test_data(dev_data, word_vocab)
X_test, test_sentences = prepare_test_data(test_data, word_vocab)


def decode_predictions(predictions, idx2tag):
    pred_tags = []
    for pred in predictions:
        pred_tags.append([idx2tag[p.item()] for p in pred])
    return pred_tags


def get_predictions(tokens_list, model):
    model.eval()
    all_predictions = []
    with torch.no_grad():
        for tokens in tokens_list:
            tokens = tokens.unsqueeze(0)
            predictions = model(tokens).squeeze(0)
            predicted_labels = torch.argmax(predictions, dim=1)
            all_predictions.append(predicted_labels)
    return all_predictions


y_pred_dev = get_predictions(X_dev, model)
y_pred_test = get_predictions(X_test, model)

idx2tag = {i: tag for tag, i in tag_vocab.get_stoi().items()}
pred_tags_dev = decode_predictions(y_pred_dev, idx2tag)
pred_tags_test = decode_predictions(y_pred_test, idx2tag)


def save_predictions_formatted(pred_tags, file_path):
    with open(file_path, 'w') as f:
        for tags in pred_tags:
            f.write(" ".join(tags) + "\n")


save_predictions_formatted(pred_tags_dev, 'dev-0/DEV-0_out.tsv')
save_predictions_formatted(pred_tags_test, 'test-A/TEST-A_out.tsv')
