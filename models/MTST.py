import torch
from torch import nn
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import PatchEmbedding


class Model(nn.Module):
    """
    Paper link: https://arxiv.org/pdf/2211.14730.pdf
    """

    def __init__(self, configs):
        """
        patch_len: int, patch len for patch_embedding
        stride: int, stride for patch_embedding
        """
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        patch_len_list = list(map(int, configs.patch_len_list.split(",")))
        stride_list = patch_len_list

        embedding_list = [
            PatchEmbedding(
                configs.d_model, patch_len, stride, stride, configs.dropout
            )
            for patch_len, stride in zip(patch_len_list, stride_list)
        ]

        self.patch_embeddings = nn.ModuleList(embedding_list)

        # Encoder
        encoder_list = [
            Encoder(
                [
                    EncoderLayer(
                        AttentionLayer(
                            FullAttention(
                                False,
                                configs.factor,
                                attention_dropout=configs.dropout,
                                output_attention=configs.output_attention,
                            ),
                            configs.d_model,
                            configs.n_heads,
                        ),
                        configs.d_model,
                        configs.d_ff,
                        dropout=configs.dropout,
                        activation=configs.activation,
                    )
                    for l in range(configs.e_layers)
                ],
                norm_layer=torch.nn.LayerNorm(configs.d_model),
            )
            for _ in range(len(patch_len_list))
        ]

        self.encoders = nn.ModuleList(encoder_list)

        # Prediction Head
        head_nf_list = [
            configs.d_model * int((configs.seq_len - patch_len) / stride + 2)
            for patch_len, stride in zip(patch_len_list, stride_list)
        ]
        self.head_nf = sum(head_nf_list)
        if self.task_name == "classification":
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(
                self.head_nf * configs.enc_in, configs.num_class
            )

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        raise NotImplementedError

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        raise NotImplementedError

    def anomaly_detection(self, x_enc):
        raise NotImplementedError

    def classification(self, x_enc, x_mark_enc):
        outputs = []
        for patch_embedding, encoder in zip(self.patch_embeddings, self.encoders):
            # do patching and embedding
            x_enc_permute = x_enc.permute(0, 2, 1)
            # u: [bs * nvars x patch_num x d_model]
            enc_out, n_vars = patch_embedding(x_enc_permute)

            # Encoder
            # z: [bs * nvars x patch_num x d_model]
            enc_out, attns = encoder(enc_out)
            # z: [bs x nvars x patch_num x d_model]
            enc_out = torch.reshape(
                enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1])
            )
            # z: [bs x nvars x d_model x patch_num]
            enc_out = enc_out.permute(0, 1, 3, 2)

            # Decoder
            enc_out = self.dropout(enc_out)
            output = enc_out.reshape(enc_out.shape[0], -1)
            outputs.append(output)
        outputs = torch.cat(outputs, dim=1)
        output = self.projection(outputs)  # (batch_size, num_classes)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if (
                self.task_name == "long_term_forecast"
                or self.task_name == "short_term_forecast"
        ):
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == "imputation":
            dec_out = self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out  # [B, L, D]
        if self.task_name == "anomaly_detection":
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == "classification":
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        return None
