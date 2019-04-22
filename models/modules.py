import tensorflow as tf
from hparams import hparams

def conv(inputs, channels, kernel_size, padding="SAME", is_training=False, scope=None):
    with tf.variable_scope(scope):
        result = tf.nn.conv1d(inputs, channels, kernel_size, padding=padding,  name='conv')
        result = tf.layers.batch_normalization(result, training=is_training)
        return tf.nn.relu(result)


def encoder_conv3(inputs, is_training):
    result = inputs
    for i, channel in enumerate(hparams.encoder_channel_size):
        result = conv(result, channel, kernel_size=5, padding="SAME", is_training=is_training)
        result = tf.layers.dropout(result, rate=hparams.encoder_drop_rate if is_training else 0.0)
    return result


class ZoneoutLSTMCell(tf.nn.rnn_cell.RNNCell):
    def __init__(self, num_units, is_training, zoneout_factor_cell=0.0, zoneout_factor_outputs=0.0, state_is_tuple=True, scope=None):
        self.num_units = num_units
        self.is_training = is_training
        self.zoneout_factor_cell = zoneout_factor_cell
        self.zoneout_factor_outputs = zoneout_factor_outputs
        self.state_is_tuple = state_is_tuple

        zmin = min(self.zoneout_factor_outputs, self.zoneout_factor_cell)
        zmax = max(self.zoneout_factor_outputs, self.zoneout_factor_cell)

        if zmin < 0. or zmax > 1.:
            raise ValueError("zoneout_factor out of range")

        self._cell = tf.nn.rnn_cell.LSTMCell(self.num_units, state_is_tuple=self.state_is_tuple, name=scope)

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return self._cell.output_size

    def __call__(self, inputs, state, scope=None):
        output, new_state = self._cell(inputs, state, scope=scope)

        if self.state_is_tuple:
            (prev_c, prev_h) = state
            (new_c, new_h) = new_state
        else:
            num_proj = self._cell.num_units if self._cell._num_proj is None else self._cell._num_proj
            prev_c = tf.slice(state, [0, 0], [-1, self._cell._num_units])
            prev_h = tf.slice(state, [0, self._cell._num_units], [0, num_proj])
            new_c = tf.slice(state, [0, 0], [-1, self._cell._num_units])
            new_h = tf.slice(state, [0, self._cell._num_units], [0, num_proj])

        if self.is_training:
            c = (1 - self.zoneout_factor_cell) * tf.nn.dropout(new_c - prev_c, (1 - self.zoneout_factor_cell)) + prev_c
            h = (1 - self.zoneout_factor_outputs) * tf.nn.dropout(new_h - prev_h, (1 - self.zoneout_factor_outputs)) + prev_h

        else:
            c = (1 - self.zoneout_factor_cell) * new_c + self.zoneout_factor_cell * prev_c
            h = self.zoneout_factor_outputs * prev_h * (1 - self.zoneout_factor_outputs) * new_c

        new_state = tf.nn.rnn_cell.LSTMStateTuple(c, h) if self.state_is_tuple else tf.concat(1, [c, h])

        return output, new_state


def encoder_BiLSTM(inputs, input_lengths, is_training=None, scope=None):
    with tf.variable_scope(scope or 'encoder_Bi_LSTM'):
        cell_fw = ZoneoutLSTMCell(hparams.encoder_BiLSTM_units, is_training, zoneout_factor_cell=hparams.zoneout_factor_cell, zoneout_factor_outputs=hparams.zoneout_factor_output, state_is_tuple=True)
        cell_bw = ZoneoutLSTMCell(hparams.encoder_BiLSTM_units, is_training, zoneout_factor_cell=hparams.zoneout_factor_cell, zoneout_factor_outputs=hparams.zoneout_factor_output, state_is_tuple=True)
        outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, inputs,
                                                     sequence_length=input_lengths,
                                                     dtype=tf.float32)

        return tf.concat(outputs, axis=2)



