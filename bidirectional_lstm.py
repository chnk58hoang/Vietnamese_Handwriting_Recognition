import torch.nn as nn



class Bidirectional_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, bidirectional=True, batch_first=True, dropout=0.25):
        super(Bidirectional_LSTM, self).__init__()
        self.bi_lstm1 = nn.LSTM(input_size=input_size, hidden_size=hidden_size, bidirectional=bidirectional,
                                batch_first=batch_first)
        self.bi_lstm2 = nn.LSTM(input_size=hidden_size * 2, hidden_size=hidden_size, bidirectional=bidirectional,
                                batch_first=batch_first)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x, _ = self.bi_lstm1(x)
        x, _ = self.bi_lstm2(x)
        x = self.dropout(x)
        return x