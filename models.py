
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from custom_transformer import TransformerEncoder, TransformerEncoderLayer
from custom_transformer import TransformerDecoder, TransformerDecoderLayer


class MHAblock(nn.Module):
    def __init__(
        self,
        embed_dim=128,
        num_heads=8,
        feature_dim=5,
    ):
        super(MHAblock, self).__init__()

        self.linear_q = nn.Linear(feature_dim, embed_dim)
        self.linear_k = nn.Linear(feature_dim, embed_dim)
        self.linear_v = nn.Linear(feature_dim, embed_dim)

        self.multihead_attention1 = nn.MultiheadAttention(embed_dim, num_heads)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.linear_q(x)
        k = self.linear_k(x)
        v = self.linear_v(x)

        atten_output, atten_output_weights = self.multihead_attention1(q, k, v)

        return atten_output


class RadarTransformer(nn.Module):
    def __init__(
        self,
        features=7,
        embed_dim=64,
        nhead=8,
        encoder_layers=6,
        decoder_layers=6,
    ):
        super(RadarTransformer, self).__init__()

        # linear projection layer
        self.linear_projection = nn.Linear(features, embed_dim)

        # transformer encder block
        encoder_layer = TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead
        )
        self.transformer_encoder = TransformerEncoder(
            encoder_layer,
            num_layers=encoder_layers
        )

        # decoder queries
        self.quries = nn.Parameter(torch.randn(241, 1, embed_dim))

        # transformer decoder layer
        decoder_layer = TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=nhead,
        )
        self.transformer_decoder = TransformerDecoder(
            decoder_layer,
            num_layers=decoder_layers,
        )

        # linear to output ranges
        self.linear_to_range = nn.Linear(embed_dim, 1)

    def forward(self, x, pad_mask, attention_map=False):
        n, b, _ = x.shape

        x = self.linear_projection(x)

        # transformer encoder
        x, ecd_att_map = self.transformer_encoder(
            x,
            src_key_padding_mask=pad_mask
        )
        # print(ecd_att_map.shape)

        # generate queries
        qur = self.quries.repeat(1, b, 1)

        # transformer decoder
        decoded, dcd_att_map = self.transformer_decoder(
            tgt=qur,
            memory=x,
            memory_key_padding_mask=pad_mask,
        )

        # project to ranges
        ranges = self.linear_to_range(decoded)
        ranges = torch.squeeze(ranges)
        if b>1:
            ranges = torch.transpose(ranges, 0, 1)

        if attention_map:
            return ranges, ecd_att_map, dcd_att_map
        else:
            return ranges


class RadarEncoder(nn.Module):
    def __init__(
        self,
        features=7,
        embed_dim=64,
        nhead=8,
        layers=6,
        output_embeding=512,
    ):
        super(RadarEncoder, self).__init__()
        # linear projection layer
        self.linear_projection = nn.Linear(features, embed_dim)

        # decoder queries
        self.quries = nn.Parameter(torch.randn(output_embeding, 1, embed_dim))

        # transformer decoder layer
        decoder_layer = TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=nhead,
        )
        self.transformer_decoder = TransformerDecoder(
            decoder_layer,
            num_layers=layers,
        )

        # linear to output ranges
        self.linear_to_output = nn.Linear(embed_dim, 1)

    def forward(self, x, pad_mask, need_attention_map=False):
        n, b, _ = x.shape

        x = self.linear_projection(x)

        # generate queries
        qur = self.quries.repeat(1, b, 1)

        # transformer decoder
        decoded, dcd_att_map = self.transformer_decoder(
            tgt=qur,
            memory=x,
            memory_key_padding_mask=pad_mask,
        )

        # project to feature embedding
        embed = self.linear_to_output(decoded)
        embed = torch.squeeze(embed)
        embed = torch.transpose(embed, 0, 1)

        if need_attention_map:
            return embed, dcd_att_map
        else:
            return embed


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        kernel = 3
        stride = 2
        self.conv = nn.Sequential(
            nn.Conv1d(2, 64, kernel_size=kernel, stride=stride),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=kernel, stride=stride),
            nn.ReLU()
        )

        dim = 64*59
        self.linear = nn.Sequential(
            nn.Linear(dim, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):

        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)

        return x


class DiscriminatorPatch(nn.Module):
    def __init__(self):
        super(DiscriminatorPatch, self).__init__()
        kernel = 3
        stride = 2
        self.conv = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=kernel, stride=stride),
            nn.ReLU(),
            nn.Conv1d(64, 32, kernel_size=kernel, stride=stride),
            nn.ReLU(),
            nn.Conv1d(32, 16, kernel_size=kernel, stride=stride),
            nn.ReLU(),
            nn.Conv1d(16, 1, kernel_size=kernel, stride=stride),
            nn.Sigmoid(),
        )

    def forward(self, x):

        x = self.conv(x)

        return x


class DiscriminatorTransform(nn.Module):
    def __init__(
        self,
        features=7,
        embed_dim=64,
        nhead=8,
        layers=6,
        patch_size=14,
    ):
        super(DiscriminatorTransform, self).__init__()

        # linear projection layer
        self.linear_projection = nn.Linear(features, embed_dim)
        self.linear_projection_laser = nn.Linear(1, embed_dim)

        # learnable extra token positioned
        self.patch_size = patch_size
        self.cls = nn.Parameter(torch.randn(patch_size, 1, embed_dim))

        # positional embedding for laser token
        self.pos_embedding = nn.Parameter(
            torch.randn(self.patch_size+241, 1, embed_dim))

        # transformer encder block
        # single_layer = TransformerEncoderLayer(
        #     d_model=embed_dim,
        #     nhead=nhead,
        # )
        # self.transformer_encoder = TransformerEncoder(
        #     single_layer,
        #     num_layers=encoder_layers,
        # )

        # transformer decoder block
        decoder_layer = TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=nhead,
        )
        self.transformer_decoder = TransformerDecoder(
            decoder_layer,
            num_layers=layers,
        )

        self.linear2patch = nn.Linear(embed_dim, 1)

    def forward(self, radar, pad_mask, laser, need_attention_map=False):
        n, b, _ = radar.shape

        radar = self.linear_projection(radar)

        laser = torch.transpose(laser, 0, 1)
        laser = torch.unsqueeze(laser, 2)
        laser = self.linear_projection_laser(laser)

        cls_tokens = self.cls.repeat(1, b, 1)

        # concat tokens to first position
        input_seq = torch.cat((cls_tokens, laser), dim=0)
        input_seq += self.pos_embedding

        # transformer encoder
        output_seq, dcd_att_map = self.transformer_decoder(
            tgt=input_seq,
            memory=radar,
            memory_key_padding_mask=pad_mask,
        )

        # cls head for patch gan
        cls_heads = output_seq[:self.patch_size, :, :]
        cls_heads = self.linear2patch(cls_heads)
        cls_heads = torch.transpose(cls_heads, 0, 1)
        cls_heads = torch.squeeze(cls_heads)

        return cls_heads
