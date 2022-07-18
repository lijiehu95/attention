import torch
import torch.nn as nn

from transformers import BertTokenizer, BertModel, BertForMaskedLM
from attention.model.modelUtils import isTrue
from allennlp.common import Registrable
from allennlp.nn.activations import Activation

class Encoder(nn.Module, Registrable) :
    def forward(self, **kwargs) :
        raise NotImplementedError("Implement forward Model")



@Encoder.register('simple-rnn')
class EncoderRNN(Encoder):
    def __init__(self, vocab_size, embed_size, hidden_size, pre_embed=None):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size

        if pre_embed is not None:
            print("Setting Embedding")
            weight = torch.Tensor(pre_embed)
            weight[0, :].zero_()

            self.embedding = nn.Embedding(vocab_size, embed_size, _weight=weight, padding_idx=0)
        else:
            self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)

        self.hidden_size = hidden_size
        # self.rnn = nn.LSTM(input_size=embed_size, hidden_size=hidden_size, batch_first=True, bidirectional=True)
        self.rnn = torch.nn.RNN(input_size=embed_size, hidden_size=hidden_size, num_layers=1, bias=True,
                           bidirectional=False)
        self.output_size = self.hidden_size

    def forward(self, data, revise_embedding=None):
        seq = data.seq
        lengths = data.lengths
        if revise_embedding is not None:
            embedding = revise_embedding
        else:
            embedding = self.embedding(seq)  # (B, L, E)
        packseq = nn.utils.rnn.pack_padded_sequence(embedding, lengths.cpu(), batch_first=True, enforce_sorted=False)
        output, h = self.rnn(packseq)
        output, lengths = nn.utils.rnn.pad_packed_sequence(output, batch_first=True, padding_value=0)

        data.hidden = output
        data.last_hidden = h
        data.embedding = embedding

        if isTrue(data, 'keep_grads'):
            # data.embedding = embedding
            data.embedding.retain_grad()
            data.hidden.retain_grad()


@Encoder.register('rnn')
class EncoderRNN(Encoder) :
    def __init__(self, vocab_size, embed_size, hidden_size, pre_embed=None) :
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size

        if pre_embed is not None :
            print("Setting Embedding")
            weight = torch.Tensor(pre_embed)
            weight[0, :].zero_()

            self.embedding = nn.Embedding(vocab_size, embed_size, _weight=weight, padding_idx=0)
        else :
            self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)

        self.hidden_size = hidden_size
        self.rnn = nn.LSTM(input_size=embed_size, hidden_size=hidden_size, batch_first=True, bidirectional=True)

        self.output_size = self.hidden_size * 2

    def forward(self, data,revise_embedding=None) :
        seq = data.seq
        lengths = data.lengths
        if revise_embedding is not None:
            embedding = revise_embedding
        else:
            embedding = self.embedding(seq)  # (B, L, E)
        packseq = nn.utils.rnn.pack_padded_sequence(embedding, lengths.cpu(), batch_first=True, enforce_sorted=False)
        output, (h, c) = self.rnn(packseq)
        output, lengths = nn.utils.rnn.pad_packed_sequence(output, batch_first=True, padding_value=0)

        data.hidden = output
        data.last_hidden = torch.cat([h[0], h[1]], dim=-1)
        data.embedding = embedding

        if isTrue(data, 'keep_grads') :
            # data.embedding = embedding
            data.embedding.retain_grad()
            data.hidden.retain_grad()
            

@Encoder.register("average")
class EncoderAverage(Encoder) :
    def __init__(self,  vocab_size, embed_size, projection, hidden_size=None, activation:Activation=Activation.by_name('linear'), pre_embed=None) :
        super(EncoderAverage, self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size

        if pre_embed is not None :
            print("Setting Embedding")
            weight = torch.Tensor(pre_embed)
            weight[0, :].zero_()

            self.embedding = nn.Embedding(vocab_size, embed_size, _weight=weight, padding_idx=0)
        else :
            self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)

        if projection :
            self.projection = nn.Linear(embed_size, hidden_size)
            self.output_size = hidden_size
        else :
            self.projection = lambda s : s
            self.output_size = embed_size

        self.activation = activation

    def forward(self, data,revise_embedding=None) :
        seq = data.seq
        lengths = data.lengths
        # embedding = self.embedding(seq) #(B, L, E)
        if revise_embedding:
            embedding = revise_embedding
        else:
            embedding = self.embedding(seq)  # (B, L, E)

        output = self.activation(self.projection(embedding))
        h = output.mean(1)

        data.hidden = output
        data.last_hidden = h
        data.embedding = embedding

        if isTrue(data, 'keep_grads') :
            # data.embedding = embedding
            data.embedding.retain_grad()
            data.hidden.retain_grad()


@Encoder.register("average")
class EncoderBERT(Encoder) :
    def __init__(self, hidden_size=None, activation:Activation=Activation.by_name('linear')) :
        super(EncoderAverage, self).__init__()

        self.embed_size = 512
        self.output_size = self.embed_size
        self.activation = activation
        self.model = BertModel.from_pretrained('bert-base-uncased')

    def forward(self, data, revise_embedding=None) :
        seq = data.seq
        if revise_embedding:
            embedding = revise_embedding
        else:
            embedding = self.embedding(seq)  # (B, L, E)

        output = self.activation(embedding)
        h = output.mean(1)

        data.hidden = output
        data.last_hidden = h
        data.embedding = embedding

        if isTrue(data, 'keep_grads'):
            # data.embedding = embedding
            data.embedding.retain_grad()
            data.hidden.retain_grad()

# @Encoder.register('bert')
# class BERT(Encoder):
#     def __init__(self, vocab_size, embed_size, hidden_size, pre_embed=None):
#         super().__init__()
#         self.hidden_size = hidden_size
#         self.output_size = self.hidden_size
#         self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#         self.bert = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=self.output_size)
#
#
# @Encoder.register('bert2')
# class BERT2(Encoder):
#    def __init__(self):
#        super(BERT, self).__init__()
#        self.embedding = Embedding()
#        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])
#        self.fc = nn.Linear(d_model, d_model)
#        self.activ1 = nn.Tanh()
#        self.linear = nn.Linear(d_model, d_model)
#        self.activ2 = gelu
#        self.norm = nn.LayerNorm(d_model)
#        self.classifier = nn.Linear(d_model, 2)
#        # decoder is shared with embedding layer
#        embed_weight = self.embedding.tok_embed.weight
#        n_vocab, n_dim = embed_weight.size()
#        self.decoder = nn.Linear(n_dim, n_vocab, bias=False)
#        self.decoder.weight = embed_weight
#        self.decoder_bias = nn.Parameter(torch.zeros(n_vocab))
#
#    def forward(self, input_ids, segment_ids, masked_pos):
#        output = self.embedding(input_ids, segment_ids)
#        enc_self_attn_mask = get_attn_pad_mask(input_ids, input_ids)
#        for layer in self.layers:
#            output, enc_self_attn = layer(output, enc_self_attn_mask)
#        # output : [batch_size, len, d_model], attn : [batch_size, n_heads, d_mode, d_model]
#        # it will be decided by first token(CLS)
#        h_pooled = self.activ1(self.fc(output[:, 0])) # [batch_size, d_model]
#        logits_clsf = self.classifier(h_pooled) # [batch_size, 2]
#
#        masked_pos = masked_pos[:, :, None].expand(-1, -1, output.size(-1)) # [batch_size, max_pred, d_model]
#
#        # get masked position from final output of transformer.
#        h_masked = torch.gather(output, 1, masked_pos) # masking position [batch_size, max_pred, d_model]
#        h_masked = self.norm(self.activ2(self.linear(h_masked)))
#        logits_lm = self.decoder(h_masked) + self.decoder_bias # [batch_size, max_pred, n_vocab]
#
#        return logits_lm, logits_clsf
