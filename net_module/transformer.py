import numpy as np
import tensorflow as tf
import tensorflow.contrib as tc
import json
import os
from .util import BaseModel
# from util import BaseModel

# from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
# from data import recover_sentence, START_ID, PAD_ID

PAD_ID = 0

class Transformer(BaseModel):
    """
    See the architecture spec of Transformer in:
        Vaswani et al. Attention is All You Need. NIPS 2017.

    """
    def __init__(self, _raw_input=None, num_heads=8, d_model=512, d_ff=2048, num_enc_layers=6, num_dec_layers=6,
                 drop_rate=0.1, warmup_steps=400, pos_encoding_type='sinusoid',
                 ls_epsilon=0.1, use_label_smoothing=True, batch_size=100, seq_len=64, 
                 model_name='transformer', tf_sess_config=None, **kwargs):

        """
        Params:
        num_heads: number of heads in transformer.
        d_model: dimention of embedding layer.
        d_ff: dimension of feed_forward layer.
        num_enc_layers: numbers of enocder layer.
        num_dec_layers: numbers of decoder layer.
        drop_rate: probability of dropout.
        warmup_steps: For computing the learning rate.
        pos_encoding_type: type of positional encoding.
        ls_epsilon: parameter in label smoothing.
        use_label_smoothing: whether to use label smoothing.
        kwargs:
        """
        assert d_model % num_heads == 0     # 嵌入维度 d_model 必须是 num_heads 的整数倍
        assert pos_encoding_type in ('sinusoid', 'embedding')       # 目前可选 sin 和 embedding 两种模式
        super().__init__(model_name, tf_sess_config=tf_sess_config)

        self.h = num_heads
        self.d_model = d_model
        self.d_ff = d_ff

        self.num_enc_layers = num_enc_layers
        self.num_dec_layers = num_dec_layers
        self.batch_size = batch_size
        self.seq_len = seq_len

        # Dropout regularization: added in every sublayer before layer_normalization(...) 
        # then applied to embedding + positional encoding. 
        self.drop_rate = drop_rate

        # Label smoothing epsilon
        self.ls_epsilon = ls_epsilon
        self.use_label_smoothing = use_label_smoothing
        self.pos_encoding_type = pos_encoding_type

        # For computing the learning rate
        self.warmup_steps = warmup_steps

        # 配置文件的整合
        self.config = dict(
            num_heads=self.h,
            d_model=self.d_model,
            d_ff=self.d_ff,
            num_enc_layers=self.num_enc_layers,
            num_dec_layers=self.num_dec_layers,
            drop_rate=self.drop_rate,
            warmup_steps=self.warmup_steps,
            ls_epsilon=self.ls_epsilon,
            use_label_smoothing=self.use_label_smoothing,
            pos_encoding_type=self.pos_encoding_type,
            model_name=self.model_name,
            tf_sess_config=self.tf_sess_config,
        )

        # The following variables are inputs for build_model().
        # 这个需要研究如何去除 ？？？
        self._input_id2word = None
        self._target_id2word = None
        self._pad_id = 0

        # The following variables will be constructed in build_model().
        self._learning_rate = None
        self._is_training = None
        self._raw_input = _raw_input 
        self._raw_target = None
        self._output = None
        self._accuracy = None
        self._loss = None
        self._train_op = None

        self._is_init = False
        self.step = 0  # Number of training step.


    def build_model(self, dataset_name='anyone', pad_id=PAD_ID, is_training=True):
        """
        Args:
            dataset_name (str): name of the training dataset.
            pad_id (int): the id of '<pad>' symbol.
            is_training (bool)
            train_params (dict): keys, must include 'lr', 'batch_size', and 'seq_len'.
        """
        # id2word 都是字典？？记录了如何从索引得到单词的嵌入？
        # assert input_id2word[pad_id] == '<pad>'   pad_id 对应的单词必须是'<pad>'
        # assert target_id2word[pad_id] == '<pad>'  pad_id 对应的单词必须是'<pad>'

        # self.config.update(dict(
        #     dataset=dataset_name,
        #     pad_id=pad_id,
        #     train_params=train_params,
        # ))
        # value of batch_size and seq_len is given.
        # batch_size = train_params.get('batch_size', 100)
        # seq_len = train_params.get('seq_len', 30)
        
        batch_size = self.batch_size
        seq_len = self.seq_len
        self._pad_id = np.int32(pad_id)

        input_vocab = 16
        target_vocab = 48

        with tf.variable_scope(self.model_name):
            self._learning_rate = tf.placeholder(tf.float32, shape=None, name='learning_rate')
            self._is_training = tf.placeholder_with_default(False, shape=None, name="is_training")

            # self._raw_input = tf.placeholder(tf.int32, shape=[batch_size, seq_len], name='raw_input')
            # Add the offset on the input and target sentences.
            # seq_len 的维度中有一个开始的 token <s>，此处作了移除，让维度保持一致。原来的代码：enc_inp = self._raw_input[:, 1:]
            enc_inp = self._raw_input

            # For the decoder input, we remove the last element, as no more future prediction is gonna be made based on it.
            # 解码的输入的最后一个单词是不需要的，因为不需要根据最后一个单词作出预测，第二个操作的目的在于制造出错位。。。
            # dec_inp = self._raw_target[:, :-1]  # starts with <s>
            # dec_target = self._raw_target[:, 1:]  # starts with the first word
            # dec_target_ohe = tf.one_hot(dec_target, depth=target_vocab, on_value=1)     # 对 dec_target 作独热编码得到 [batch_size, seq_len, target_vocab] 维度的张量，用于后续损失函数的计算
            
            # what we called label_smoothing refers to smoothing the label in loss calculation related，在损失函数的计算中才需要标签光滑化，暂时不考虑这个
            # if self.use_label_smoothing:
            #     dec_target_ohe = self.label_smoothing(dec_target_ohe)

            # The input mask only hides the <pad> symbol. pading mask 仅仅掩住了<pad> 符号
            input_mask = self.construct_padding_mask(enc_inp)
            
            # The target mask hides both <pad> and future words. 对于最后的输出而言建立 mask 需要同时掩掉 <pad> 符号和未来的单词
            # Mask of decoder needs to consider padding mask and autoregressive mask simultaneously. 解码阶段所需要的，暂时不考虑
            # target_mask = self.construct_padding_mask(dec_inp)
            # print('shape of target_mask after padding mask,', target_mask.shape.as_list())
            # target_mask *= self.construct_autoregressive_mask(dec_inp)  # 重点看下构建自回归的 mask 的做法和结果
            # print('shape of target_mask after autoregressive,', target_mask.shape.as_list())

            # Input embedding + positional encoding，之后就送入 encoder 
            inp_embed = self.preprocess(enc_inp, input_vocab, "input_preprocess")
            self.enc_out = self.encoder(inp_embed, input_mask)

            # Target embedding + positional encoding 
            # decoder 的输入包括 decoder_inp 的嵌入，encoder 的输出，以及 padding mask 和 target_mask 
            # dec_inp_embed = self.preprocess(dec_inp, target_vocab, "target_preprocess")
            # print('shape of dec_inp_embed equals, ', dec_inp_embed.shape.as_list())
            # dec_out = self.decoder(dec_inp_embed, enc_out, input_mask, target_mask) # input_mask 和 traget_mask 所起的作用分别是什么？
            # print('shape of dec_out equals, ', dec_out.shape.as_list())

            # # Make the prediction out of the decoder output. ？？？
            # logits = tf.layers.dense(dec_out, target_vocab)  # [batch, target_vocab]
            # print('shape of logits', logits.shape.as_list())
            # self._output = tf.argmax(logits, axis=-1, output_type=tf.int32)
            # print('shape of self._output equals, ', self._output.shape.as_list())

            # # accuracy 为何是这么计算的呢？
            # target_not_pad = tf.cast(tf.not_equal(dec_target, self._pad_id), tf.float32)
            # print('shape of target_not_pad', target_not_pad.shape.as_list())
            # self._accuracy = tf.reduce_sum(
            #     tf.cast(tf.equal(self._output, dec_target), tf.float32) * target_not_pad /
            #     tf.cast(tf.reduce_sum(target_not_pad), tf.float32)
            # )
            # print('shape of self._accuracy', self._accuracy.shape.as_list())

            # # 独热编码之后的 dec_target 主要是用来计算损失的，交叉熵损失函数
            # self._loss = tf.reduce_mean(
            #     tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=dec_target_ohe))
            # print('shape of self._loss equals, ', self._loss.shape.as_list())

            # optim = tf.train.AdamOptimizer(learning_rate=self._learning_rate,
            #                                beta1=0.9, beta2=0.98, epsilon=1e-9)
            # self._train_op = optim.minimize(self._loss)

        # with tf.variable_scope(self.model_name + '_summary'):
        #     tf.summary.scalar('loss', self._loss)
        #     tf.summary.scalar('accuracy', self._accuracy)
        #     self.merged_summary = tf.summary.merge_all()


    # @classmethod
    # def load_model(cls, model_name, is_training=False):
    #     """Returns a Transformer object, with checkpoint loaded.
    #     """
    #     config_path = os.path.join(REPO_ROOT, 'checkpoints', model_name, 'model.config.json')
    #     with open(config_path, 'r') as fin:
    #         cfg = json.load(fin)

    #     model = cls(**cfg)
    #     model.build_model(cfg['dataset'], cfg['input_id2word'], cfg['target_id2word'],
    #                       pad_id=cfg['pad_id'], is_training=is_training,
    #                       **cfg['train_params'])
        
    #     model.load_checkpoint()
    #     return model

    def old_embedding(self, inp, vocab_size, zero_pad=True):
        """When the `zero_pad` flag is on, the first row in the embedding lookup table is
        fixed to be an all-zero vector, corresponding to the '<pad>' symbol."""
        embed_size = self.d_model
        embed_lookup = tf.get_variable("embed_lookup", [vocab_size, embed_size], tf.float32,
                                       initializer=tf.contrib.layers.xavier_initializer())
        if zero_pad:
            assert self._pad_id == 0
            embed_lookup = tf.concat((tf.zeros(shape=[1, self.d_model]), embed_lookup[1:, :]), 0)
        # out.shape.as_list() should be [inp.shape[0], inp.shape[1], embed_size], [batch_size, seq_len, embed_size]
        out = tf.nn.embedding_lookup(embed_lookup, inp) 
        return out 

    def embedding(self, inp):
        """
        Params: inp has shape: [batchs, seq_len, num_features]
        Reurns: shape: [batch, seq_len, embedding_size]. out.shape.as_list() should be [inp.shape[0], inp.shape[1], embed_size], [batch_size, seq_len, embed_size]
        """
        embed_size = self.d_model
        out = tf.layers.dense(inputs=inp, units=embed_size, activation=None)
        
        return out 

    def _positional_encoding_embedding(self, inp):
        batch_size, seq_len = inp.shape.as_list()

        with tf.variable_scope('positional_embedding'):
            # Copy [0, 1, ..., `inp_size`] by `batch_size` times => matrix [batch, seq_len]
            pos_ind = tf.tile(tf.expand_dims(tf.range(seq_len), 0), [batch_size, 1])
            return self.embedding(pos_ind, seq_len, zero_pad=False)  # [batch, seq_len, d_model]

    def _positional_encoding_sinusoid(self, inp):
        """
        PE(pos, 2i) = sin(pos / 10000^{2i/d_model})
        PE(pos, 2i+1) = cos(pos / 10000^{2i/d_model})
        """
        batch, seq_len = inp.shape.as_list()

        with tf.variable_scope('positional_sinusoid'):
            # Copy [0, 1, ..., `inp_size`] by `batch_size` times => matrix [batch, seq_len]
            pos_ind = tf.tile(tf.expand_dims(tf.range(seq_len), 0), [batch, 1])

            # Compute the arguments for sin and cos: pos / 10000^{2i/d_model})
            # Each dimension is sin/cos wave, as a function of the position.
            pos_enc = np.array([
                [pos / np.power(10000., 2. * (i // 2) / self.d_model) for i in range(self.d_model)]
                for pos in range(seq_len)
            ])  # [seq_len, d_model]

            # Apply the cosine to even columns and sin to odds.
            pos_enc[:, 0::2] = np.sin(pos_enc[:, 0::2])  # dim 2i
            pos_enc[:, 1::2] = np.cos(pos_enc[:, 1::2])  # dim 2i+1

            # Convert to a tensor
            lookup_table = tf.convert_to_tensor(pos_enc, dtype=tf.float32)  # [seq_len, d_model]
            if True:
                lookup_table = tf.concat((tf.zeros(shape=[1, self.d_model]), lookup_table[1:, :]), 0)

            out = tf.nn.embedding_lookup(lookup_table, pos_ind)  # [batch, seq_len, d_model]
            return out

    def positional_encoding(self, inp):
        if self.pos_encoding_type == 'sinusoid':
            pos_enc = self._positional_encoding_sinusoid(inp)
        else:
            pos_enc = self._positional_encoding_embedding(inp)
        return pos_enc

    def preprocess(self, inp, inp_vocab, scope):
        # Pre-processing: embedding + positional encoding；Output shape: [batch, seq_len, d_model]
        with tf.variable_scope(scope):
            # out = self.embedding(inp, inp_vocab, zero_pad=True) + self.positional_encoding(inp)，暂时不考虑位置编码
            out = self.embedding(inp)
            # out = tf.layers.dropout(out, rate=self.drop_rate, training=self._is_training)  # 需要作 dropout 吗，暂时没法控制是否训练，因此无法考虑

        return out

    def layer_norm(self, inp):
        return tc.layers.layer_norm(inp, center=True, scale=True)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """
        Args:
            Q (tf.tensor): of shape (h * batch, q_size, d_model)
            K (tf.tensor): of shape (h * batch, k_size, d_model)
            V (tf.tensor): of shape (h * batch, k_size, d_model)
            mask (tf.tensor): of shape (h * batch, q_size, k_size)
        """
        # third dimension of Q, V and K must equal d = self.d_model//self.h
        d = self.d_model // self.h
        assert d == Q.shape[-1] == K.shape[-1] == V.shape[-1]

        out = tf.matmul(Q, tf.transpose(K, [0, 2, 1]))  # [h * batch, q_size, k_size]
        out = out / tf.sqrt(tf.cast(d, tf.float32))  # scaled by sqrt(d_k)

        # if mask is not None:
        if False:
            # masking out (0.0) => setting to -inf.
            out = tf.multiply(out, mask) + (1.0 - mask) * (-1e10)

        out = tf.nn.softmax(out)  # [h * batch, q_size, k_size]
        # out = tf.layers.dropout(out, training=self._is_training)
        out = tf.matmul(out, V)  # [h * batch, q_size, d_model]

        return out

    def multihead_attention(self, query, memory=None, mask=None, scope='attn'):
        """
        Args:
            query (tf.tensor): of shape (batch, q_size, d_model)
            memory (tf.tensor): of shape (batch, m_size, d_model)
            mask (tf.tensor): shape (batch, q_size, k_size)

        Returns:h
            a tensor of shape (bs, q_size, d_model)
        """
        if memory is None:
            memory = query

        with tf.variable_scope(scope):
            # Linear project to d_model dimension: [batch, q_size/k_size, d_model]
            Q = tf.layers.dense(query, self.d_model, activation=tf.nn.relu, kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=1.0))
            K = tf.layers.dense(memory, self.d_model, activation=tf.nn.relu, kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=1.0))
            V = tf.layers.dense(memory, self.d_model, activation=tf.nn.relu, kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=1.0))

            # Split the matrix to multiple heads and then concatenate to have a larger
            # batch size: [h*batch, q_size/k_size, d_model/num_heads]
            Q_split = tf.concat(tf.split(Q, self.h, axis=2), axis=0)
            K_split = tf.concat(tf.split(K, self.h, axis=2), axis=0)
            V_split = tf.concat(tf.split(V, self.h, axis=2), axis=0)
            mask_split = tf.tile(mask, [self.h, 1, 1])

            # Apply scaled dot product attention
            out = self.scaled_dot_product_attention(Q_split, K_split, V_split, mask=mask_split)
            # Merge the multi-head back to the original shape
            out = tf.concat(tf.split(out, self.h, axis=0), axis=2)  # [batch_size, q_size, d_model]

            # The final linear layer and dropout.
            # out = tf.layers.dense(out, self.d_model)
            # out = tf.layers.dropout(out, rate=self.drop_rate, training=self._is_training)

        return out

    def feed_forwad(self, inp, scope='ff'):
        """
        Position-wise fully connected feed-forward network, applied to each position
        separately and identically. It can be implemented as (linear + ReLU + linear) or
        (conv1d + ReLU + conv1d).

        Args:
            inp (tf.tensor): shape [batch, length, d_model]
        """
        out = inp
        with tf.variable_scope(scope):
            # out = tf.layers.dense(out, self.d_ff, activation=tf.nn.relu)
            # out = tf.layers.dropout(out, rate=self.drop_rate, training=self._is_training)
            # out = tf.layers.dense(out, self.d_model, activation=None)
            # by default, use_bias=True
            out = tf.layers.conv1d(out, filters=self.d_ff, kernel_size=1, activation=tf.nn.relu)
            out = tf.layers.conv1d(out, filters=self.d_model, kernel_size=1)

        return out

    def construct_padding_mask(self, inp):
        """
        Args: Original input of word ids, shape [batch, seq_len]
        Returns: a mask of shape [batch, seq_len, seq_len], where <pad> is 0 and others are 1s.
        """
        seq_len = inp.shape.as_list()[1]
        mask = tf.cast(tf.not_equal(inp, self._pad_id), tf.float32)  # mask '<pad>'
        mask = tf.tile(mask, [1, seq_len, 1])    # why repeat this dimension ? 为何需要变成维度为 `[batch_size, seq_len, seq_len]` ？？
        return mask

    def construct_autoregressive_mask(self, target):
        """
        Args: Original target of word ids, shape [batch, seq_len]
        Returns: a mask of shape [batch, seq_len, seq_len].
        """
        batch_size, seq_len = target.shape.as_list()

        # Method np.tril_indices() returns indices of element 1 in lower triangular matrix. 
        # Then reconstruct corresponding triangular matrix of dimension seq_len * seq_len. 
        tri_matrix = np.zeros((seq_len, seq_len))
        tri_matrix[np.tril_indices(seq_len)] = 1

        mask = tf.convert_to_tensor(tri_matrix, dtype=tf.float32)
        masks = tf.tile(tf.expand_dims(mask, 0), [batch_size, 1, 1])  # copies
        
        return masks

    def encoder_layer(self, inp, input_mask, scope):
        """
        Args:
            inp: tf.tensor of shape (batch, seq_len, embed_size)
            input_mask: tf.tensor of shape (batch, seq_len, seq_len)
        """
        out = inp
        with tf.variable_scope(scope):
            # One multi-head attention + one feed-forword
            # 这个加法运算算是用了残差连接吗？
            out = self.layer_norm(out + self.multihead_attention(out, mask=input_mask))
            out = self.layer_norm(out + self.feed_forwad(out))
        return out

    def encoder(self, inp, input_mask, scope='encoder'):
        """
        Args:
            inp (tf.tensor): shape (batch, seq_len, embed_size)
            input_mask (tf.tensor): shape (batch, seq_len, seq_len)
            scope (str): name of the variable scope.
        """
        out = inp  # now, (batch, seq_len, embed_size)
        with tf.variable_scope(scope):
            for i in range(self.num_enc_layers):
                out = self.encoder_layer(out, input_mask, f'enc_{i}')
        return out

    def decoder_layer(self, target, enc_out, input_mask, target_mask, scope):
        out = target
        with tf.variable_scope(scope):
            out = self.layer_norm(out + self.multihead_attention(
                out, mask=target_mask, scope='self_attn'))
            out = self.layer_norm(out + self.multihead_attention(
                out, memory=enc_out, mask=input_mask))
            out = self.layer_norm(out + self.feed_forwad(out))
        return out

    def decoder(self, target, enc_out, input_mask, target_mask, scope='decoder'):
        out = target
        with tf.variable_scope(scope):
            for i in range(self.num_enc_layers):
                out = self.decoder_layer(out, enc_out, input_mask, target_mask, f'dec_{i}')
        return out

    def label_smoothing(self, inp):
        """
        From the paper: "... employed label smoothing of epsilon = 0.1. This hurts perplexity,
        as the model learns to be more unsure, but improves accuracy and BLEU score."

        Args:
            inp (tf.tensor): one-hot encoding vectors, [batch, seq_len, vocab_size]
        """
        vocab_size = inp.shape.as_list()[-1]
        smoothed = (1.0 - self.ls_epsilon) * inp + (self.ls_epsilon / vocab_size)       # 对输入作光滑化
        return smoothed

    def init(self):
        """Call .init() before training starts.
        - Initialize the variables.
        - Save the model config into json file.
        """
        self.sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
        self._is_init = True
        self.step = 0

        self.save_checkpoint()  # make sure saver is created.
        # Save the model config into a json.
        config_path = os.path.join(self.checkpoint_dir, 'model.config.json')
        with open(config_path, 'w') as fout:
            json.dump(self.config, fout)

    def done(self):
        """Call .done() after training is complete.
        """
        self.writer.close()
        self.save_checkpoint()  # Final checkpoint.

    # def train(self, input_ids, target_ids):
    #     """
    #     One train step with one mini-batch.

    #     Args:
    #         input_ids (np.array): same shape as raw input placeholder.
    #         target_ids (np.array): same shape as raw target placeholder.

    #     Returns:
    #         A dict of some meta information, including 'loss'.
    #     """
    #     assert self._is_init, "Please call .init() before training starts."
    #     self.step += 1

    #     lr = np.power(self.d_model, -0.5) * min(
    #         np.power(self.step, -0.5),
    #         self.step * np.power(self.warmup_steps, -1.5)
    #     )

    #     train_loss, train_accu, summary, _ = self.sess.run(
    #         [self._loss, self._accuracy, self.merged_summary, self.train_op],
    #         feed_dict={
    #             self._learning_rate: lr,
    #             self.raw_input_ph: input_ids.astype(np.int32),
    #             self.raw_target_ph: target_ids.astype(np.int32),
    #             self.is_training_ph: True,
    #         })
    #     self.writer.add_summary(summary, global_step=self.step)

    #     if self.step % 10000 == 0:
    #         # Save the model checkpoint every 1000 steps.
    #         self.save_checkpoint(step=self.step)

    #     return {'train_loss': train_loss,
    #             'train_accuracy': train_accu,
    #             'learning_rate': lr,
    #             'step': self.step}

    # def predict(self, input_ids):
    #     """
    #     Make predict in an autoregressive way.

    #     Args:
    #         input_ids (np.array): same shape as raw input placeholder.

    #     Returns:
    #         a np.array of the same shape as the raw target placeholder.
    #     """
    #     assert list(input_ids.shape) == self.raw_input_ph.shape.as_list()
    #     batch_size, inp_seq_len = self.raw_input_ph.shape.as_list()

    #     input_ids = input_ids.astype(np.int32)
    #     pred_ids = np.zeros(input_ids.shape, dtype=np.int32)
    #     pred_ids[:, 0] = START_ID

    #     # Predict one output a time autoregressively.
    #     for i in range(1, inp_seq_len):
    #         # The decoder does not output <s>
    #         next_pred = self.sess.run(self._output, feed_dict={
    #             self.raw_input_ph: input_ids,
    #             self.raw_target_ph: pred_ids,
    #             self.is_training_ph: False,
    #         })
    #         # Only update the i-th column in one step.
    #         pred_ids[:, i] = next_pred[:, i - 1]
    #         # print(f"i={i}", pred_ids)

    #     return pred_ids

    # def evaluate(self, input_ids, target_ids):
    #     """Make a prediction and compute BLEU score.
    #     """
    #     pred_ids = self.predict(input_ids)

    #     refs = []
    #     hypos = []
    #     for truth, pred in zip(target_ids, pred_ids):
    #         truth_sent = recover_sentence(truth, self._target_id2word)
    #         pred_sent = recover_sentence(pred, self._target_id2word)

    #         refs.append([truth_sent])
    #         hypos.append(pred_sent)

    #     # Print the last pair for fun.
    #     source_sent = recover_sentence(input_ids[-1], self._input_id2word)
    #     print("[Source]", source_sent)
    #     print("[Truth]", truth_sent)
    #     print("[Translated]", pred_sent)

    #     smoothie = SmoothingFunction().method4
    #     bleu_score = corpus_bleu(refs, hypos, smoothing_function=smoothie)
    #     return {'bleu_score': bleu_score * 100.}

    # ============================= Utils ===============================

    def _check_variable(self, v, name):
        if v is None:
            raise ValueError(f"Call build_model() to initialize {name}.")
        return v

    @property
    def raw_input_ph(self):
        return self._check_variable(self._raw_input, 'input placeholder')

    @property
    def raw_target_ph(self):
        return self._check_variable(self._raw_target, 'target placeholder')

    @property
    def is_training_ph(self):
        return self._check_variable(self._is_training, 'is_training placeholder')

    @property
    def train_op(self):
        return self._check_variable(self._train_op, 'train_op')

    @property
    def loss(self):
        return self._check_variable(self._loss, 'loss')



if __name__ == '__main__':
    train_params = {
        'lr': 0.001, 
        'batch_size': 25, 
        'seq_len': 30, 
    }

    transformer_net = Transformer()
    transformer_net.build_model(dataset_name='test', train_params=train_params)



