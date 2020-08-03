import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ScrewNet(nn.Module):
    def __init__(self, lstm_hidden_dim=1000, n_lstm_hidden_layers=1, drop_p=0.5, n_output=8):
        super(ScrewNet, self).__init__()

        self.fc_res_dim_1 = 512
        self.lstm_input_dim = 1000
        self.lstm_hidden_dim = lstm_hidden_dim
        self.n_lstm_hidden_layers = n_lstm_hidden_layers
        self.fc_lstm_dim_1 = 256
        self.fc_lstm_dim_2 = 128
        self.n_output = n_output
        self.drop_p = drop_p

        self.resnet = models.resnet18()
        self.fc_res_1 = nn.Linear(self.lstm_input_dim, self.fc_res_dim_1)
        self.bn_res_1 = nn.BatchNorm1d(self.fc_res_dim_1, momentum=0.01)
        self.fc_res_2 = nn.Linear(self.fc_res_dim_1, self.lstm_input_dim)

        self.LSTM = nn.LSTM(
            input_size=self.lstm_input_dim,
            hidden_size=self.lstm_hidden_dim,
            num_layers=self.n_lstm_hidden_layers,
            batch_first=True,
        )

        self.fc_lstm_1 = nn.Linear(self.lstm_hidden_dim, self.fc_lstm_dim_1)
        self.bn_lstm_1 = nn.BatchNorm1d(self.fc_lstm_dim_1, momentum=0.01)
        self.fc_lstm_2 = nn.Linear(self.fc_lstm_dim_1, self.fc_lstm_dim_2)
        self.bn_lstm_2 = nn.BatchNorm1d(self.fc_lstm_dim_2, momentum=0.01)
        self.dropout_layer1 = nn.Dropout(p=self.drop_p)
        self.fc_lstm_3 = nn.Linear(self.fc_lstm_dim_2, self.n_output)

        # # Initialization
        # for m in self.modules():
        #     if isinstance(m, nn.Linear):
        #         nn.init.xavier_uniform_(m.weight)
        #         m.bias.data.fill_(0.01)
        #     elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)
        #     elif isinstance(m, nn.LSTM):
        #         for name, param in m.named_parameters():
        #             if 'bias' in name:
        #                 nn.init.constant_(param, 0.0)
        #             elif 'weight' in name:
        #                 nn.init.xavier_normal_(param)

    def forward(self, X_3d):
        # X shape: Batch x Sequence x 3 Channels x img_dims
        # Run resnet sequentially on the data to generate embedding sequence
        cnn_embed_seq = []
        for t in range(X_3d.size(1)):
            x = self.resnet(X_3d[:, t, :, :, :])
            x = x.view(x.size(0), -1)
            x = self.bn_res_1(self.fc_res_1(x))
            x = F.relu(x)
            x = self.fc_res_2(x)
            cnn_embed_seq.append(x)

        # swap time and sample dim such that (sample dim, time dim, CNN latent dim)
        cnn_embed_seq = torch.stack(cnn_embed_seq, dim=0).transpose_(0, 1)

        # run lstm on the embedding sequence
        self.LSTM.flatten_parameters()

        RNN_out, (h_n, h_c) = self.LSTM(cnn_embed_seq, None)
        """ h_n shape (n_layers, batch, hidden_size), h_c shape (n_layers, batch, hidden_size) 
        None represents zero initial hidden state. RNN_out has shape=(batch, time_step, output_size) """

        # FC layers
        x_rnn = RNN_out.contiguous().view(-1, self.lstm_hidden_dim)  # Using Last layer of RNN
        x_rnn = self.bn_lstm_1(self.fc_lstm_1(x_rnn))
        x_rnn = F.relu(x_rnn)
        x_rnn = self.bn_lstm_2(self.fc_lstm_2(x_rnn))
        x_rnn = F.relu(x_rnn)
        x_rnn = self.fc_lstm_3(x_rnn)
        return x_rnn.view(X_3d.size(0), -1)


class ScrewNet_2imgs(nn.Module):
    def __init__(self, n_output=8):
        super(ScrewNet_2imgs, self).__init__()

        self.fc_mlp_dim_1 = 2000
        self.fc_mlp_dim_2 = 512
        self.fc_mlp_dim_3 = 256
        self.n_output = n_output

        self.resnet = models.resnet18()
        self.bn_res_1 = nn.BatchNorm1d(1000, momentum=0.01)

        self.fc_mlp_1 = nn.Linear(self.fc_mlp_dim_1, self.fc_mlp_dim_1)
        self.bn_mlp_1 = nn.BatchNorm1d(self.fc_mlp_dim_1, momentum=0.01)
        self.fc_mlp_2 = nn.Linear(self.fc_mlp_dim_1, self.fc_mlp_dim_2)
        self.bn_mlp_2 = nn.BatchNorm1d(self.fc_mlp_dim_2, momentum=0.01)
        self.fc_mlp_3 = nn.Linear(self.fc_mlp_dim_2, self.fc_mlp_dim_3)
        self.bn_mlp_3 = nn.BatchNorm1d(self.fc_mlp_dim_3, momentum=0.01)
        self.fc_mlp_4 = nn.Linear(self.fc_mlp_dim_3, self.n_output)

    def forward(self, X_3d):
        # X shape: Batch x Sequence x 3 Channels x img_dims
        # Run resnet sequentially on the data to generate embedding sequence
        cnn_embed_seq = []
        for t in range(X_3d.size(1)):
            x = self.resnet(X_3d[:, t, :, :, :])
            x = x.view(x.size(0), -1)
            x = self.bn_res_1(x)
            cnn_embed_seq.append(x)

        # swap time and sample dim such that (sample dim, time dim, CNN latent dim)
        cnn_embed_seq = torch.stack(cnn_embed_seq, dim=0).transpose_(0, 1)
        x_rnn = cnn_embed_seq.contiguous().view(-1, self.fc_mlp_dim_1)

        # FC layers
        x_rnn = self.fc_mlp_1(x_rnn)
        x_rnn = F.relu(x_rnn)
        x_rnn = self.bn_mlp_1(x_rnn)
        x_rnn = self.fc_mlp_2(x_rnn)
        x_rnn = F.relu(x_rnn)
        x_rnn = self.bn_mlp_2(x_rnn)
        x_rnn = self.fc_mlp_3(x_rnn)
        x_rnn = F.relu(x_rnn)
        x_rnn = self.bn_mlp_3(x_rnn)
        x_rnn = self.fc_mlp_4(x_rnn)
        return x_rnn.view(X_3d.size(0), -1)



class ScrewNet_NoLSTM(nn.Module):
    def __init__(self, seq_len=16, fc_replace_lstm_dim=1000, n_output=8):
        super(ScrewNet_NoLSTM, self).__init__()

        self.fc_res_dim_1 = 512
        self.fc_replace_lstm_dim = fc_replace_lstm_dim
        self.fc_replace_lstm_seq_dim = fc_replace_lstm_dim * seq_len
        self.fc_lstm_dim_1 = 256
        self.fc_lstm_dim_2 = 128
        self.n_output = n_output

        self.resnet = models.resnet18()
        self.fc_res_1 = nn.Linear(self.fc_replace_lstm_dim, self.fc_res_dim_1)
        self.bn_res_1 = nn.BatchNorm1d(self.fc_res_dim_1, momentum=0.01)
        self.fc_res_2 = nn.Linear(self.fc_res_dim_1, self.fc_replace_lstm_dim)

        self.fc_replace_lstm = nn.Linear(self.fc_replace_lstm_seq_dim, self.fc_replace_lstm_seq_dim)

        self.fc_lstm_1 = nn.Linear(self.fc_replace_lstm_dim, self.fc_lstm_dim_1)
        self.bn_lstm_1 = nn.BatchNorm1d(self.fc_lstm_dim_1, momentum=0.01)
        self.fc_lstm_2 = nn.Linear(self.fc_lstm_dim_1, self.fc_lstm_dim_2)
        self.bn_lstm_2 = nn.BatchNorm1d(self.fc_lstm_dim_2, momentum=0.01)
        self.fc_lstm_3 = nn.Linear(self.fc_lstm_dim_2, self.n_output)

    def forward(self, X_3d):
        # X shape: Batch x Sequence x 3 Channels x img_dims
        # Run resnet sequentially on the data to generate embedding sequence
        cnn_embed_seq = []
        for t in range(X_3d.size(1)):
            x = self.resnet(X_3d[:, t, :, :, :])
            x = x.view(x.size(0), -1)
            x = self.bn_res_1(self.fc_res_1(x))
            x = F.relu(x)
            x = self.fc_res_2(x)
            cnn_embed_seq.append(x)

        # swap time and sample dim such that (sample dim, time dim, CNN latent dim)
        cnn_embed_seq = torch.stack(cnn_embed_seq, dim=0).transpose_(0, 1)

        # FC replacing LSTM layer
        cnn_embed_seq = cnn_embed_seq.contiguous().view(cnn_embed_seq.size(0), -1)
        x_rnn = F.relu(self.fc_replace_lstm(cnn_embed_seq))
        x_rnn = x_rnn.view(-1, self.fc_replace_lstm_dim)

        # FC layers
        x_rnn = self.bn_lstm_1(self.fc_lstm_1(x_rnn))
        x_rnn = F.relu(x_rnn)
        x_rnn = self.bn_lstm_2(self.fc_lstm_2(x_rnn))
        x_rnn = F.relu(x_rnn)
        x_rnn = self.fc_lstm_3(x_rnn)
        return x_rnn.view(X_3d.size(0), -1)