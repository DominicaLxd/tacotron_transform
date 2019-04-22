import tensorflow as tf
import numpy as np
from utils.symbols import symbols
from models.modules import *
#from .dynamic_decoder import dynamic_decode

class Tacotron2():
    def __init__(self, hparams):
        self._hparams = hparams

    def initialize(self, inputs, input_lengths, mel_targets=None, attention_style=None):
        with tf.variable_scope('inference') as scope:

            self.is_training = mel_targets is not None
            # character Embedding

            embedding_table = tf.get_variable(name='embedding_table', shape=[len(symbols), 512], dtype=tf.float32,
                                              initializer=tf.truncated_normal_initializer(stddev=0.5))
            embedding_input = tf.nn.embedding_lookup(embedding_table, inputs)

            # Encoder
            encoder_conv3_output = encoder_conv3(inputs, self.is_training)
            encoder_output = encoder_BiLSTM(encoder_conv3_output, input_lengths, is_training=self.is_training)

            # Decoder
            if attention_style is not None:
                #TODO: define it
                attention_mechanism = None

            attention_decoder = MyAttentionWrapper()





