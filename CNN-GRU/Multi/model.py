class CNNGRUModel(nn.Module):
    def __init__(self, input_dim, cnn_out_channels, cnn_kernel_size, gru_hidden_dim, output_dim=4, dropout_prob):
        super(CNNGRUModel, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=input_dim, out_channels=cnn_out_channels, kernel_size=cnn_kernel_size)
        self.gru = nn.GRU(input_size=cnn_out_channels, hidden_size=gru_hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(gru_hidden_dim, output_dim)
        self.activation = nn.Tanh()

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (batch, seq, feature) -> (batch, feature, seq)
        x = F.relu(self.conv1d(x))
        x = x.permute(0, 2, 1)  # (batch, feature, seq) -> (batch, seq, feature)
        x, _ = self.gru(x)  # x is (batch_size, seq_len, hidden_size)
        x = x[:, -1, :]  # Use the output of the last sequence step
        x = self.dropout(x)
        x = self.fc(x)
        x = self.activation(x)
        return x  # (batch_size, output_dim), i.e., (batch_size, 4)
