import torch.nn as nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence


class HierarchicalAttentionNetwork(nn.Module):
    def __init__(self, vocab_size, embed_dim, word_gru_hidden_dim, sent_gru_hidden_dim, word_gru_num_layers,
                 sent_gru_num_layers, word_att_dim, sent_att_dim, use_layer_norm, dropout):
        super(HierarchicalAttentionNetwork, self).__init__()

        self.sent_attention = SentenceAttention(
            vocab_size, embed_dim, word_gru_hidden_dim, sent_gru_hidden_dim,
            word_gru_num_layers, sent_gru_num_layers, word_att_dim, sent_att_dim, use_layer_norm, dropout)

        self.fc = nn.Linear(2 * sent_gru_hidden_dim, 1)
        self.sig = nn.Sigmoid()

        self.use_layer_nome = use_layer_norm
        self.dropout = dropout

    def forward(self, code_tensor):

        code_lengths = []
        sent_lengths = []

        for file in code_tensor:
            code_line = []
            code_lengths.append(len(file))
            for line in file:
                code_line.append(len(line))
            sent_lengths.append(code_line)

        code_tensor = code_tensor.type(torch.LongTensor)
        code_lengths = torch.tensor(code_lengths).type(torch.LongTensor).cuda()
        sent_lengths = torch.tensor(sent_lengths).type(torch.LongTensor).cuda()

        code_embeds, word_att_weights, sent_att_weights, sents = self.sent_attention(code_tensor, code_lengths,
                                                                                     sent_lengths)

        scores = self.fc(code_embeds)
        final_scrs = self.sig(scores)

        return final_scrs, word_att_weights, sent_att_weights, sents


class SentenceAttention(nn.Module):
    def __init__(self, vocab_size, embed_dim, word_gru_hidden_dim, sent_gru_hidden_dim,
                 word_gru_num_layers, sent_gru_num_layers, word_att_dim, sent_att_dim, use_layer_norm, dropout):
        super(SentenceAttention, self).__init__()

        self.word_attention = WordAttention(vocab_size, embed_dim, word_gru_hidden_dim, word_gru_num_layers,
                                            word_att_dim, use_layer_norm, dropout)

        self.gru = nn.GRU(2 * word_gru_hidden_dim, sent_gru_hidden_dim, num_layers=sent_gru_num_layers,
                          batch_first=True, bidirectional=True, dropout=dropout)

        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(2 * sent_gru_hidden_dim, elementwise_affine=True)
        self.dropout = nn.Dropout(dropout)

        self.sent_attention = nn.Linear(2 * sent_gru_hidden_dim, sent_att_dim)

        self.sentence_context_vector = nn.Linear(sent_att_dim, 1, bias=False)

    def forward(self, code_tensor, code_lengths, sent_lengths):
        device = code_tensor.device

        code_lengths, code_perm_idx = code_lengths.sort(dim=0, descending=True)
        code_perm_idx = code_perm_idx.to(device)
        code_tensor = code_tensor[code_perm_idx]
        sent_lengths = sent_lengths[code_perm_idx]

        packed_sents = pack_padded_sequence(code_tensor, lengths=code_lengths.tolist(), batch_first=True)

        valid_bsz = packed_sents.batch_sizes

        packed_sent_lengths = pack_padded_sequence(sent_lengths, lengths=code_lengths.tolist(), batch_first=True)

        sents, word_att_weights = self.word_attention(packed_sents.data, packed_sent_lengths.data)

        sents = self.dropout(sents)

        packed_sents, _ = self.gru(PackedSequence(sents, valid_bsz))

        if self.use_layer_norm:
            normed_sents = self.layer_norm(packed_sents.data)
        else:
            normed_sents = packed_sents

        att = torch.tanh(self.sent_attention(normed_sents))
        att = self.sentence_context_vector(att).squeeze(1)

        val = att.max()
        att = torch.exp(att - val)

        att, _ = pad_packed_sequence(PackedSequence(att, valid_bsz), batch_first=True)

        sent_att_weights = att / torch.sum(att, dim=1, keepdim=True)

        code_tensor, _ = pad_packed_sequence(packed_sents, batch_first=True)

        code_tensor = code_tensor * sent_att_weights.unsqueeze(2)
        code_tensor = code_tensor.sum(dim=1)

        word_att_weights, _ = pad_packed_sequence(PackedSequence(word_att_weights, valid_bsz), batch_first=True)

        _, code_tensor_unperm_idx = code_perm_idx.sort(dim=0, descending=False)
        code_tensor = code_tensor[code_tensor_unperm_idx]

        word_att_weights = word_att_weights[code_tensor_unperm_idx]
        sent_att_weights = sent_att_weights[code_tensor_unperm_idx]

        return code_tensor, word_att_weights, sent_att_weights, sents


class WordAttention(nn.Module):
    def __init__(self, vocab_size, embed_dim, gru_hidden_dim, gru_num_layers, att_dim, use_layer_norm, dropout):
        super(WordAttention, self).__init__()

        self.embeddings = nn.Embedding(vocab_size, embed_dim)

        self.gru = nn.GRU(embed_dim, gru_hidden_dim, num_layers=gru_num_layers, batch_first=True, bidirectional=True,
                          dropout=dropout)

        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(2 * gru_hidden_dim, elementwise_affine=True)
        self.dropout = nn.Dropout(dropout)

        self.attention = nn.Linear(2 * gru_hidden_dim, att_dim)

        self.context_vector = nn.Linear(att_dim, 1, bias=False)

    def init_embeddings(self, embeddings):
        self.embeddings.weight = nn.Parameter(embeddings)

    def freeze_embeddings(self, freeze=False):
        self.embeddings.weight.requires_grad = freeze

    def forward(self, sents, sent_lengths):
        device = sents.device
        sent_lengths, sent_perm_idx = sent_lengths.sort(dim=0, descending=True)
        sent_perm_idx = sent_perm_idx.to(device)
        sents = sents[sent_perm_idx]

        sents = self.embeddings(sents.cuda())

        packed_words = pack_padded_sequence(sents, lengths=sent_lengths.tolist(), batch_first=True)

        valid_bsz = packed_words.batch_sizes

        packed_words, _ = self.gru(packed_words)

        if self.use_layer_norm:
            normed_words = self.layer_norm(packed_words.data)
        else:
            normed_words = packed_words

        att = torch.tanh(self.attention(normed_words.data))
        att = self.context_vector(att).squeeze(1)

        val = att.max()
        att = torch.exp(att - val)

        att, _ = pad_packed_sequence(PackedSequence(att, valid_bsz), batch_first=True)

        att_weights = att / torch.sum(att, dim=1, keepdim=True)

        sents, _ = pad_packed_sequence(packed_words, batch_first=True)

        sents = sents * att_weights.unsqueeze(2)
        sents = sents.sum(dim=1)

        _, sent_unperm_idx = sent_perm_idx.sort(dim=0, descending=False)
        sents = sents[sent_unperm_idx]

        att_weights = att_weights[sent_unperm_idx]

        return sents, att_weights