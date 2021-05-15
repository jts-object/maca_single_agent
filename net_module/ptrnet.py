"""
Implementation of a Pointer Network using AttentionWrapper.
"""

import numpy as np
import tensorflow as tf
from .helper import GreedyEmbeddingHelper
from .attention_wrapper import AttentionWrapper, BahdanauAttention
from .decoder import dynamic_decode
from .basic_decoder import BasicDecoder

# can be edited (to anything larger than vocab size) if encoding of vocab already uses 0, 1
END_TOKEN = -1
START_TOKEN = 1


class PointerNet(object):
    def __init__(self, n_pointers=1, batch_size=100, seq_max_len=30, learning_rate=0.001,
                 cell=tf.contrib.rnn.GRUCell, n_layers=3, n_units=50, name='pointer_net'):
        """Creates TensorFlow graph of a pointer network.
        Args:
            n_pointers (int):      Number of pointers to generate.
            batch_size (int) :     Batch size for training/inference.
            seq_max_len (int):      Maximum sequence length of inputs to encoder.
            learning_rate (float): Learning rate for Adam optimizer.
            cell (method):         Method to create single RNN cell.
            n_layers (int):        Number of layers in RNN (assumed to be the same for encoder & decoder).
            n_units (int):         Number of units in RNN cell (assumed to be the same for all cells).
        """
        self.n_pointers = n_pointers
        self.batch_size = batch_size
        self.seq_max_len = seq_max_len
        self.learning_rate = learning_rate
        self.cell = cell
        self.n_layers = n_layers
        self.n_units = n_units
        self.name = name
        self.input_lengths = tf.placeholder(tf.int32, [self.batch_size], name=name + 'input_length')

        # word_matrix = np.zeros(shape=(101, 1))
        # for i in range(101):
        #     word_matrix[i][0] = float(i+1)
        
        word_matrix = np.random.normal(size=[13, 6], loc=10000., scale=2000.)
        self.word_matrix = tf.Variable(word_matrix, trainable=True, name=name + 'word_matrix', dtype=tf.float32)

    def encoder(self):
        with tf.variable_scope(self.name + 'inputs'):
            # integer-encoded input passages (e.g. 'She went home' -> [2, 3, 4]) 
            self.encoder_inputs = tf.placeholder(tf.int32, [self.batch_size, self.seq_max_len]) 
            # actual non-padded length of each input passages; used for dynamic unrolling 
            # (e.g. ['She went home', 'She went to the station'] -> [3, 5]) 
            self.input_lengths = tf.placeholder(tf.int32, [self.batch_size]) 

        with tf.variable_scope(self.name + 'outputs'):
            # pointer(s) to answer: (e.g. 'She went home' -> [2])
            self.pointer_labels = tf.placeholder(tf.int32, [self.batch_size, self.n_pointers])
            # token 作为开始符或者终止符
            self.start_tokens = tf.constant(START_TOKEN, shape=[self.batch_size], dtype=tf.int32)

            # outputs of decoder are the word 'pointed' to by each pointer
            self.decoder_labels = tf.stack([tf.gather(inp, ptr) for inp, ptr in
                                           list(zip(tf.unstack(self.encoder_inputs), tf.unstack(self.pointer_labels)))])
    
            # inputs to decoder are inputs shifted over by one, with a <start> token at the front
            # 把 decoder_labels 的前面拼接上了 start_token 
            self.decoder_inputs = tf.concat([tf.expand_dims(self.start_tokens, 1), self.decoder_labels], 1)
            # output lengths are equal to the number of pointers
            self.output_lengths = tf.constant(self.n_pointers, shape=[self.batch_size])

        with tf.variable_scope(self.name + 'embeddings'):
            # lookup embeddings of inputs & decoder inputs
            self.input_embeds = tf.nn.embedding_lookup(self.word_matrix, self.encoder_inputs)
            self.output_embeds = tf.nn.embedding_lookup(self.word_matrix, self.decoder_inputs)

        with tf.variable_scope(self.name + 'encoder'):
            if self.n_layers > 1:
                enc_cell = tf.contrib.rnn.MultiRNNCell([self.cell(self.n_units) for _ in range(self.n_layers)])
            else:
                enc_cell = self.cell(self.n_units)

            self.encoder_outputs, _ = tf.nn.dynamic_rnn(enc_cell, self.input_embeds, self.input_lengths, dtype=tf.float32)


    def decoder(self, memory, num_layers, memory_mask):
        """
        Params:
        memory: encoder 的输出
        num_layers: decoder 的层数
        从 memory 中读取一些参数: batch_size, seq_len 和 num_units，num_units 未必要等于memory的最后一个维度的大小，
        BahdanauAttention 类会在输入层进行全连接层的转化，将最后一个维度转为 num_units 大小。
        """
        assert memory.shape.ndims == 3
        batch_size, seq_length, num_units = memory.shape.as_list()
        # num_units = 17
        # self.memory_mask = tf.placeholder(dtype=tf.float32, shape=[batch_size, seq_length], name=self.name + 'memory_mask')
        with tf.variable_scope(self.name + 'attention'):
            attention = BahdanauAttention(num_units, memory, memory_mask=memory_mask)

        self.start_tokens = tf.constant(START_TOKEN, shape=[self.batch_size], dtype=tf.int32)
        with tf.variable_scope(self.name + 'decoder'):
            helper = GreedyEmbeddingHelper(memory, self.start_tokens, END_TOKEN)
            if self.n_layers > 1:
                dec_cell = tf.contrib.rnn.MultiRNNCell([self.cell(num_units) for _ in range(num_layers)])
            else:
                dec_cell = self.cell(self.n_units)   
            
            attn_cell = AttentionWrapper(dec_cell, attention, alignment_history=True)
            out_cell = tf.contrib.rnn.OutputProjectionWrapper(attn_cell, self.word_matrix.shape[0] - 2)
            decoder = BasicDecoder(out_cell, helper, out_cell.zero_state(batch_size, tf.float32))

            self.decoder_outputs, self.dec_state, _, self.rec_inputs, self.rec_dec_finished, self.final_sample_id_ta = dynamic_decode(
                decoder, maximum_iterations=self.n_pointers)

        with tf.variable_scope(self.name + 'pointers'):
            # tensor of shape (# pointers, batch size, max. input sequence length)
            self.pointer_prob = tf.reshape(self.dec_state.alignment_history.stack(), [self.n_pointers, batch_size, seq_length])
            # self.pointers = tf.unstack(tf.argmax(self.pointer_prob, axis=2, output_type=tf.int32))
            self.pointers = tf.argmax(self.pointer_prob, axis=2, output_type=tf.int32)


# attention = tf.contrib.seq2seq.BahdanauAttention(num_units, memory, memory_sequence_length=self.input_lengths)
# attention = BahdanauAttention(num_units, memory, memory_sequence_length=self.input_lengths)
# helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(self.word_matrix, self.start_tokens, END_TOKEN)
# attn_cell = tf.contrib.seq2seq.AttentionWrapper(dec_cell, attention, alignment_history=True)    # rnn.cell 的子类
# decoder = tf.contrib.seq2seq.BasicDecoder(out_cell, helper, out_cell.zero_state(batch_size, tf.float32))
# self.decoder_outputs, self.dec_state, _ = tf.contrib.seq2seq.dynamic_decode(
#     decoder, maximum_iterations=self.n_pointers)
