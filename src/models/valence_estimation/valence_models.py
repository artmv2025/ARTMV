import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import JointSeparator from joint_seperator

class CNNBlock(nn.Module):
    def __init__(self, input_dim, kernel_size_1, kernel_size_2, num_filters=32):
        super(CNNBlock, self).__init__()
        self.conv_layer = nn.Conv2d(1, num_filters, (kernel_size_1, kernel_size_2))
        self.batch_norm = nn.BatchNorm2d(num_filters)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = torch.unsqueeze(x, 1)
        
        feature_maps = self.conv_layer(x)
        feature_maps = self.batch_norm(feature_maps)
        feature_maps = self.activation(feature_maps)
        
        reshaped_output = torch.reshape(feature_maps, (feature_maps.size(0), -1, feature_maps.size(3)))
        reshaped_output = reshaped_output.permute(0, 2, 1)
        
        return reshaped_output

class BiLayerAttentionLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim_1=256, hidden_dim_2=128, dropout=0.25):
        super(ValenceAttentionLSTM, self).__init__()
        self.lstm_1 = nn.LSTM(input_dim, hidden_dim_1, batch_first=True)
        self.dropout_1 = nn.Dropout(dropout)
        self.lstm_2 = nn.LSTM(hidden_dim_1, hidden_dim_2, batch_first=True)
        self.dropout_2 = nn.Dropout(dropout)
        self.self_attention_layer = SelfAttention(hidden_dim_2)

    def forward(self, x):
        lstm_output_1, _ = self.lstm_1(x)
        output_1 = self.dropout_1(lstm_output_1)
        lstm_output_2, _ = self.lstm_2(output_1)
        output_2 = self.dropout_2(lstm_output_2)
        attn_out, _ = self.self_attention_layer(output_2, None)
        return attn_out.view(attn_out.size(0), -1)
    
class BiLayerLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim_1=256, hidden_dim_2=128, dropout=0.25):
        super(ValenceLSTM2, self).__init__()
        self.lstm_1 = nn.LSTM(input_dim, hidden_dim_1, batch_first=True)
        self.dropout_1 = nn.Dropout(dropout)
        self.lstm_2 = nn.LSTM(hidden_dim_1, hidden_dim_2, batch_first=True)
        self.dropout_2 = nn.Dropout(dropout)
    
    def forward(self, x):
        lstm_output_1, _ = self.lstm_1(x)
        output_1 = self.dropout_1(lstm_output_1)
        lstm_output_2, _ = self.lstm_2(output_1)
        output_2 = self.dropout_2(lstm_output_2)
        return output_2


class StackedLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_layers=2, dropout=0.25):
        super(ValenceLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, dropout=dropout, batch_first=True)
    
    def forward(self, x):
        output, _ = self.lstm(x)
        return output
    
    
class CNNLSTMBlock(nn.Module):
    def __init__(self, input_dim, kernel_size_1, kernel_size_2, num_filters=32, 
                 lstm_input_dim, lstm_network='BiLayerAttentionLSTM'):
        super(CNNLSTMBlock, self).__init__()
        self.cnn_network = CNNBlock(input_dim, kernel_size_1, kernel_size_2, num_filters)
        
        if lstm_network_type == 'BiLayerAttentionLSTM':
            self.lstm_network = BiLayerAttentionLSTM(lstm_input_dim)
        elif lstm_network_type == 'BiLayerLSTM:
            self.lstm_network = BiLayerLSTM(lstm_input_dim)
        elif lstm_network_type == 'StackedLSTM:
            self.lstm_network = StackedLSTM(lstm_input_dim)
        
    def forward(self, x):
        out = self.cnn_network(x)
        out, _ = self.lstm_network(out)
        return out
    

class P_CNN_LSTM_SHCA(nn.Module):
    def __init__(self, input_dim, kernel_size_1, kernel_size_2, filter_num, num_heads=1, lstm_network='StackedLSTM'):
        super(P_CNN_LSTM_SHCA, self).__init__()
        
        self.joint_separator = JointSeparator()
        
        # CNN-LSTM blocks for different joint groups
        self.cnn_lstm_left = CNNLSTMBlock(input_dim, kernel_size_1, kernel_size_2, filter_num, 96, lstm_network)
        self.cnn_lstm_right = CNNLSTMBlock(input_dim, kernel_size_1, kernel_size_2, filter_num, 96, lstm_network)
        self.cnn_lstm_mid = CNNLSTMBlock(input_dim, kernel_size_1, kernel_size_2, filter_num, 32, lstm_network)
        
        # Attention layers
        self.right_attention_layer = Cross_Attention(256, 256, num_heads=num_heads)
        self.left_attention_layer = Cross_Attention(256, 256, num_heads=num_heads)
        self.mid_attention_layer = Self_Attention(256, num_heads=num_heads)
        
        self.fc = nn.LazyLinear(1)
        
    def forward(self, x):
        # Separate input joints into left, right, and mid groups
        left_joints, right_joints, mid_joints = self.joint_separator.separate(x)
        
        # CNN-LSTM processing for each joint group
        left_embed_out = self.cnn_lstm_left(left_joints)
        right_embed_out = self.cnn_lstm_right(right_joints)
        mid_embed_out = self.cnn_lstm_mid(mid_joints)
        
        # Cross-Attention between left and right embeddings
        right_out, RcL_weights = self.right_attention_layer(right_embed_out, left_embed_out)
        left_out, LcR_weights = self.left_attention_layer(left_embed_out, right_embed_out)
        
        # Self-Attention on middle embeddings
        mid_out, mid_weights = self.mid_attention_layer(mid_embed_out, None)
        
        # Concatenate processed outputs from all branches
        out = torch.cat([right_out, left_out, mid_out], dim=2)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        out = torch.squeeze(out, 1)
        out = torch.sigmoid(out)
        return out

