import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Conv_Blocks import DilatedConvEncoder
import numpy as np


class Model(nn.Module):
    def __init__(self, configs, hidden_dims=128, output_dims=320, kernel_size=3):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention

        self.encoder = DilatedConvEncoder(
            configs.enc_in,
            [hidden_dims] * configs.e_layers + [output_dims],  # a list here
            kernel_size=kernel_size,
        )

        # Decoder
        if (
            self.task_name == "long_term_forecast"
            or self.task_name == "short_term_forecast"
        ):
            raise NotImplementedError
        if self.task_name == "imputation":
            raise NotImplementedError
        if self.task_name == "anomaly_detection":
            raise NotImplementedError
        if self.task_name == "classification":
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(output_dims, configs.num_class)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        raise NotImplementedError

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        raise NotImplementedError

    def anomaly_detection(self, x_enc):
        raise NotImplementedError

    def classification(self, x_enc, x_mark_enc):  # (batch_size, timestamps, enc_in)
        # conv encoder
        output = self.dropout(
            self.encoder(x_enc.transpose(1, 2))
        )  # (batch_size, output_dims, timestamps)
        output = output.transpose(1, 2)  # (batch_size, timestamps, hidden_dims)

        output = F.max_pool1d(
            output.transpose(1, 2), kernel_size=output.size(1)
        ).transpose(1, 2)  # (batch_size, 1, output_dims)
        output = output.squeeze(1)  # (batch_size, output_dims)
        output = self.projection(output)  # (batch_size, num_classes)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if (
            self.task_name == "long_term_forecast"
            or self.task_name == "short_term_forecast"
        ):
            raise NotImplementedError
        if self.task_name == "imputation":
            raise NotImplementedError
        if self.task_name == "anomaly_detection":
            raise NotImplementedError
        if self.task_name == "classification":
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        return None
