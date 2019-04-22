import numpy as np
import tensorflow as tf
"""
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

lr = 0.001
epoch = 10
batch_size = 32

n_input = 28
n_steps = 28

n_hidden = 128
n_classes = 10

x = tf.placeholder(dtype=tf.float32, shape=[None, n_steps, n_input])
y = tf.placeholder(dtype=tf.float32, shape=[None, n_classes])

class ZoneoutLSTMCell(tf.nn.rnn_cell.RNNCell):
    def __init__(self, num_units, is_training, zoneout_factor_cell=0., zoneout_factor_output=0., state_is_tuple=True, name=None):
        zm = min(zoneout_factor_output, zoneout_factor_cell)
        zs = max(zoneout_factor_output, zoneout_factor_cell)

        if zm < 0. or zs > 1.:
            raise ValueError('One/both provided Zoneout factors are not in [0, 1]')

        self._cell = tf.nn.rnn_cell.LSTMCell(num_units, state_is_tuple=state_is_tuple, name=name)
        self._zoneout_cell = zoneout_factor_cell
        self._zoneout_outputs = zoneout_factor_output
        self.is_training = is_training
        self.state_is_tuple = state_is_tuple

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return self._cell.output_size

    def __call__(self, inputs, state, scope=None):
        output, new_state = self._cell(inputs, state, scope)

        if self.state_is_tuple:
            (prev_c, prev_h) = state
            (new_c, new_h) = new_state
        else:
            num_proj = self._cell._num_units if self._cell._num_proj is None else self._cell._num_proj
            # [batch_size, c + h]
            prev_c = tf.slice(state, [0, 0], [-1, self._cell._num_units])
            prev_h = tf.slice(state, [0, self._cell._num_units], [-1, num_proj])
            new_c = tf.slice(new_state, [0, 0], [-1, self._cell._num_units])
            new_h = tf.slice(new_state, [0, self._cell._num_units], [-1, num_proj])

        # apply zoneout
        if self.is_training:
            c = (1 - self._zoneout_cell) * tf.nn.dropout(new_c - prev_c, (1 - self._zoneout_cell)) + prev_c
            h = (1 - self._zoneout_outputs) * tf.nn.dropout(new_h - prev_h, (1 - self._zoneout_outputs)) + prev_h

        else:
            c = (1 - self._zoneout_cell) * new_c + self._zoneout_cell * prev_c
            h = (1 - self._zoneout_outputs) * new_h + self._zoneout_outputs * prev_h

        new_state = tf.nn.rnn_cell.LSTMStateTuple(c, h) if self.state_is_tuple else tf.concat(1, [c, h])

        return output, new_state
"""
"""
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# normalization
x_train = x_train / 255
x_test = x_test / 255

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(512, activation=tf.nn.relu),  # (输出维度， 激活函数)
    tf.keras.layers.Dropout(0.2),  # 丢弃的输入比例
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test)


a = tf.constant(3.0, dtype=tf.float32, name='a')
b = tf.constant(4.0, name='b')
total = a + b
print(a)
print(b)
print(total)

#writer = tf.summary.FileWriter('.')
#writer.add_graph(tf.get_default_graph())

sess = tf.Session()
#print(sess.run({'ab': (a, b), 'total': total}))

vec = tf.random_uniform(shape=(3, ))
out1 = vec + 1
out2 = vec + 2
print(sess.run(vec))
print(sess.run(vec))
print(sess.run((out1, out2)))

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
z = x + y
sess = tf.Session()
print(sess.run(z, feed_dict={x: 3, y: 4.5}))
print(sess.run(z, feed_dict={x: [1, 3], y: [2, 4]}))

my_data = [
    [0, 1],
    [2, 3],
    [4, 5],
    [6, 7],
]  # [4, 2]

slices = tf.data.Dataset.from_tensor_slices(my_data)
next_item = slices.make_one_shot_iterator().get_next()
sess = tf.Session()
while True:
    try:
        print(sess.run(next_item))
    except tf.errors.OutOfRangeError:
        break
""""""
from tensorflow.keras import layers

inputs = tf.keras.Input(shape=(32, ))
x = layers.Dense(64, activation='relu')(inputs)
x = layers.Dense(64, activation='relu')(x)

predictions = layers.Dense(10, activation='softmax')(x)

model = tf.keras.Model(inputs=inputs, outputs=predictions)

model.compile(optimizer=tf.train.RMSPropOptimizer(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

data = np.random.random([1000, 32])
labels = np.random.random([1000, 10])

model.fit(data, labels, batch_size=32, epochs=5)

one = tf.Variable(1, tf.int16)
mystr = tf.Variable(['H'], tf.string)
cloo = tf.Variable([1, 2, 3, 4, 5], tf.int32)
mymat = tf.Variable([
    [[2, 3], [2, 4]],
    [[1, 3], [1, 3]]
], tf.int16)

print(one)
print(mystr)
print(cloo)
print(mymat)
features = {
    'sales': [[5], [10], [8], [9]],
    'department': ['sports', 'sports', 'gardening', 'gardening']}
department_colunm = tf.feature_column.categorical_column_with_vocabulary_list(
    'department', ['sports', 'gardening'])
department_colunm = tf.feature_column.indicator_column(department_colunm)
columns = [
    tf.feture_column.numeric_column('sales'),
    department_colunm
]
inputs = tf.feature_column.input_layer(features, columns)


x = tf.constant([[1], [2], [3], [4]], dtype=tf.float32)
y_true = tf.constant([[0], [-1], [-2], [-3]], dtype=tf.float32)

tv = tf.Variable()
tc = tf.constant()
tst = tf.SparseTensor()

linear_model = tf.layers.Dense(units=1)
y_pred = linear_model(x)
loss = tf.losses.mean_squared_error(labels=y_true, predictions=y_pred)
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)


sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
for i in range(100):
    _, loss_value = sess.run((train, loss))
    print(loss_value)
print(sess.run(y_pred))

""""""
model = tf.keras.Sequential()
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))


model.compile(optimizer=tf.train.AdamOptimizer(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.compile(optimizer='优化器',
              loss='损失类型',
              metrics='监控训练')

data = np.random.random([1000, 32])
labels = np.random.random([1000, 10])

# 测试数据这种感觉
val_data = np.random.random([100, 32])
val_labels = np.random.random([100, 10])

dataset = tf.data.Dataset.from_tensor_slices((data, labels))
dataset = dataset.batch(32).repeat()

#val_dataset = tf.data.Dataset.from_tensor_slices((val_data, val_labels))
#val_dataset = val_dataset.batch(32).repeat()

#model.fit(data, labels, epochs=10, batch_size=32, validation_data=(val_data, val_labels))

model.evaluate(data, labels, batch_size=32)
model.evaluate(dataset, steps=30)

from tensorflow import keras
from tensorflow.keras import layers
data = np.random.random([1000, 32])
labels = np.random.random([1000, 10])

class MyLayer(layers.Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(MyLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        shape = tf.TensorShape((input_shape[1], self.output_dim))
        self.kernel = self.add_weight(name='kernel',
                                      shape=shape,
                                      initializer='uniform',
                                      trainable=True)
        super(MyLayer, self).build(input_shape)

    def call(self, inputs):
        return tf.matmul(inputs, self.kernel)

    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape).as_list()
        shape[-1] = self.output_dim
        return tf.TensorShape(shape)

    def get_config(self):
        base_config = super(MyLayer, self).get_config()
        base_config['output_dim'] = self.output_dim
        return base_config

    @classmethod
    def from_config(cls, config):
        return cls(**config)



class MyModel(keras.Model):
    def __init__(self, num_classes=10):
        super(MyModel, self).__init__(name='my_model')
        self.num_claases = num_classes
        #
        self.dense_1 = layers.Dense(32, activation='relu')
        self.dense_2 = layers.Dense(num_classes, activation='softmax')

    def call(self, inputs):
        x = self.dense_1(inputs)
        return self.dense_2(x)

    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape).as_list()
        shape[-1] = self.num_claases
        return tf.TensorShape(shape)

model = tf.keras.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10,)))
model.add(layers.Dense(1, activation='sigmoid'))

#model = MyModel(10)
model.compile(optimizer=tf.train.AdamOptimizer(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()
#model.fit(data, labels, batch_size=32, epochs=5)"""
"""
tf.Variable()
my_variable = tf.get_variable("my_variable", [1, 2, 3])

# 多设备共享变量
tf.GraphKeys.GLOBAL_VARIABLES
# 计算梯度的变量
tf.GraphKeys.TRAINABLE_VARIABLES

tf.train.replica_device_setter()

v = tf.get_variable("v", shape=(), initializer=tf.zeros_initializer())

assignment = v.assign_add(1)
with tf.control_dependencies([assignment]):
    w = v.read_value()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    print(sess.run([w]))




def conv_relu(input, kernel_shape, bias_shape):
    weights = tf.get_variable('weights', kernel_shape,
                              initializer=tf.random_normal_initializer())
    biases = tf.get_variable('biases', bias_shape,
                              initializer=tf.random_normal_initializer(0.0))
    conv = tf.nn.conv2d(input, weights, strides=[1, 1, 1, 1], padding='SAME')

    return tf.nn.relu(conv + biases)

input1 = tf.random_normal([1, 10, 10, 32])
input2 = tf.random_normal([1, 20, 20, 32])

with tf.variable_scope('conv_1'):
    x = conv_relu(input1, kernel_shape=[5, 5, 32, 32], bias_shape=[32])
with tf.variable_scope('conv_2'):
    x = conv_relu(x, kernel_shape=[5, 5, 32, 32], bias_shape=[32])





x = tf.placeholder(tf.float32, shape=[3])
y = tf.square(x)
#y = tf.matmul(x, w)
output = tf.nn.softmax(y)
#init_op = w.initializer

with tf.Session() as sess:

    print(sess.run(y, feed_dict={x: [1.0, 2.0, 3.0]}))
    print(sess.run(y, feed_dict={x: [0.0, 0.0, 5.0]}))
    sess.run(y)
    sess.run(y, {x: 37})




y = tf.matmul([[37.0, -23.0], [1.0, 4.0]], tf.random_uniform([2, 2]))

with tf.Session() as sess:
    options = tf.RunOptions()
    options.output_partition_graphs = True
    options.trace_level = tf.RunOptions.FULL_TRACE

    metadata = tf.RunMetadata()

    sess.run(y, options=options, run_metadata=metadata)

    print(metadata.partition_graphs)

    print(metadata.step_stats)


x = tf.constant([[37.0, -23.0], [1.0, 4.0]])
w = tf.Variable(tf.random_uniform([2, 2]))
y = tf.matmul(x, w)

loss = tf.losses.mean_squared_error(labels=)
""""""≈
max_iters = 10000 if not (is_training or is_evaling) else None
class CustomDecoderOutput(collections.namedtuple("CustomDecoderOutput", ("rnn_output", "token_output", "sample_id"))):
    pass
class CustomDecoder(decoder.Decoder):

    与tacotron定制的helper匹配，输出一个stop_token



        CustomDecoder(decoder_cell, self.helper, decoder_init_state),
        :param cell: RNNCell
        :param helper: 'Helper'
        :param initial_state:

        rnn_cell_impl.assert_like_rnncell(type(cell), cell)

        self._cell = cell
        self._helper = helper
        self._initial_state = initial_state
        self._output_layer = output_layer

    @property
    def batch_size(self):
        return self._helper.batch_size

    def _rnn_output_size(self):
        size = self._cell.output_size
        if self._output_layer is None:
            return size

        else:
            pass

    @property
    def output_size(self):
        return CustomDecoderOutput(
            rnn_output=self._rnn_output_size(),
            token_output=self._helper.token_output_size,
            sample_id=self._helper.sample_ids_shape
        )

    @property
    def output_dtype(self):
        dtype = nest.flatten(self._initial_state)[0].dtype  # 嵌套结构压平
        return CustomDecoderOutput(
            nest.map_structure(lambda _: dtype, self._rnn_output_size()),
            tf.float32,
            self._helper.sample_ids_shape
        )

    def initialize(self, name=None):
        return self._helper.initialize() + (self._initial_state,)

    def step(self, time, inputs, state, name=None):
        with ops.name_scope(name, "CustomDecoderStep", (time, inputs, state)):
            (cell_outputs, stop_token), cell_state = self._cell(inputs, state)

            if self._output_layer is not None:
                pass
            sample_ids = self._helper.sample(
                time=time, outputs=cell_outputs, state=cell_state)
            (finished, next_inputs, next_state) = self._helper.next_inputs(
                time=time,
                outputs=cell_outputs,
                state=cell_state,
                sample_ids=sample_ids,
                stop_token_prediction=stop_token)

        outputs = CustomDecoderOutput(cell_outputs, stop_token, sample_ids)
        return (outputs, next_state, next_inputs, finished)
(frames_prediction, stop_token_prediction, _), final_decoder_state, _ = dynamic_decode(
    CustomDecoder(decoder_cell, self.helper, decoder_init_state),
    impute_finished=False,
    maximum_iterations=None,  # 最大解码步数
)
""""""
from tensorflow.python.layers import base as layers_base
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
class AttentionMechanism(object):
    pass

def _maybe_mask_score(score, memory_sequence_length, score_mask_value):
    if memory_sequence_length is None:
        return score
    message = ("All values in memory_sequence_length must greater than zero.")
    with ops.control_dependencies([check_ops.assert_positive(memory_sequence_length, message=message)]):
        score_mask = array_ops.sequence_mask(memory_sequence_length, maxlen=array_ops.shape[score][1])
        score_mask_values = score_mask_value * array_ops.ones_like(score)
        return array_ops.where(score_mask, score, score_mask_value)

class _BaseAttentionMechanism(AttentionMechanism):
    def __init__(self,
                 query_layer,
                 memory,
                 probability_fn,
                 memory_sequence_length=None,
                 memory_layer=None,
                 check_inner_dims_defined=True,
                 score_mask_value=float("-inf"),
                 name=None):

        if (query_layer is not None and not isinstance(query_layer, layers_base.Layer)):
            raise TypeError("query_layer is not a Layer: %s" % type(query_layer).__name__)
        if (memory_layer is not None and not isinstance(memory_layer, layers_base.Layer)):
            raise TypeError("query_layer is not a Layer: %s" % type(memory_layer).__name__)

        self._query_layer = query_layer
        self._memory_layer = memory_layer

        if not callable(probability_fn):
            pass
        self._probability_fn = lambda score, prev(
            probability_fn(_maybe_mask_score())
        )


class HybridAttention(_BaseAttentionMechanism):
    pass

class LocationBasedAttention(_BaseAttentionMechanism):
    pass

# attention_style None = LocationBasedAttention
if attention_style is None:
    attention_mechanism = LocationBasedAttention(hparams.attention_size, encoder_outputs)
                                                # 128
else:
    attention_mechanism = HybridAttention(hp.attention_size, encoder_outputs)
"""
from tensorflow.contrib import rnn
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.ops.rnn_cell import RNNCell
from tensorflow.contrib.seq2seq import dynamic_decode
from tensorflow.contrib.seq2seq.python.ops import decoder
from tensorflow.python.ops import rnn_cell_impl
import collections
from tensorflow.python.util import nest
from tensorflow.python.framework import ops, tensor_shape
from tensorflow.contrib.seq2seq.python.ops.attention_wrapper import BahdanauAttention
from tensorflow.python.ops import variable_scope, check_ops
from tensorflow.contrib.rnn import RNNCell
# TODO: decoder_cell
    # TODO: prenet done
    # TODO: attention_mechanism
    # TODO: decoder_LSTM done
    # TODO: frame_proj done
    # TODO: stop_proj   done

# TODO: helper
"""
class Prenet:
    def __init__(self, is_training, layers_sizes=[256, 256], drop_rate=0.5, activation=tf.nn.relu, scope=None):
        super(Prenet, self).__init__()
        self.drop_rate = drop_rate
        self.layers_sizes = layers_sizes
        self.activation = activation
        self.is_training = is_training

        self.scope = 'prenet' if scope is None else scope

    def __call__(self, inputs):
        result = inputs

        with tf.variable_scope(self.scope):
            for i, size in enumerate(self.layers_sizes):
                result = tf.layers.dense(result, size, activation=self.activation,
                                         name='dense_{}'.format(i + 1))
                result = tf.layers.dropout(result, rate=self.drop_rate, training=True,
                                           name='dropout_{}'.format(i + 1))

            return result
class FrameProjection:
    def __init__(self, shape=80, activation=None, scope=None):
        super(FrameProjection, self).__init__()
        self.shape = shape
        self.activation = activation
        self.scope = 'linear_transform_projection' if scope is None else scope
        self.dense = tf.layers.Dense(units=shape, activation=activation, name='prejection_{}'.format(self.scope))

    def __call__(self, inputs):
        with tf.varialbe_scope(self.scope):
            output = self.dense(inputs)
            return output
class StopProjection:
    def __init__(self, is_training, shape=1, activation=tf.nn.sigmoid, scope=None):
        super(StopProjection, self).__init__()
        self.is_training = is_training
        self.shape = shape
        self.activation = activation
        self.scope = 'stop_token_projection' if scope is None else scope

    def __call__(self, inputs):
        with tf.variable_scope(self.scope):
            output = tf.layers.dense(inputs, units=self.shape,
                                     activation=None, name='projection_{}'.format(self.scope))

            if self.is_training:
                return output
            return self.activation(output)
class DecoderRNN:
    def __init__(self, is_training, layers=2, size=1024, zoneout=0.1, scope=None):
        super(DecoderRNN, self).__init__()
        self.is_training = is_training
        self.layers = layers
        self.size = size
        self.zoneout = zoneout
        self.scope = 'decoder_lstm' if scope is None else scope

        # 2层LSTM
        self.rnn_layers = [ZoneoutLSTMCell(
            size, is_training, zoneout_factor_cell=zoneout, zoneout_factor_output=zoneout,
            name='decoder_LSTM_{}'.format(i+1))for i in range(layers)]

        self._cell = tf.contrib.rnn.MultiRNNCell(self.rnn_layers, state_is_tuple=True)

    def __call__(self, inputs, states):
        with tf.variable_scope(self.scope):
            return self._cell(inputs, states)

def _smoothing_normalization(e):
    # 由e计算a
    return tf.nn.sigmoid(e) / tf.reduce_sum(tf.nn.sigmoid(e), axis=-1, keepdims=True)

class LocationSensitiveAttention(BahdanauAttention):
    def __init__(self,
                 num_units,
                 memory,
                 is_training,
                 mask_encoder=True,
                 memory_sequence_length=None,
                 smoothing=False,
                 cumulate_weights=True,
                 name='LocationSensitiveAttention'):

        # hp.attention_dim, encoder_outputs, hparams, is_training, mask_encoder=True, memory_sequence_length=input_length, smoothing=None, cumulative_weight=True
        normalization_function = _smoothing_normalization if smoothing else None
        #  是否需要aij sum == 1
        memory_length = memory_sequence_length
        super(LocationSensitiveAttention, self).__init__(
            num_units=num_units,
            memory=memory,
            memory_sequence_length=memory_length,
            probaility_fn=normalization_function,
            name=name
        )
        self.location_convolution = tf.layers.Conv1D(fileters=32,
                                                     kelnel_size=(31, ),
                                                     padding='SAME',
                                                     use_bias=True,
                                                     bias_initializer=tf.zeros_initializer(),
                                                     name='location_features_convolution')
        self.location_layer = tf.layers.Dense(units=num_units,
                                              use_bias=False,
                                              dtype=tf.float32,
                                              name='location_features_layer')
        self._cumulate = cumulate_weights
        self.synthesis_contraint = False
        self.attention_win_size = tf.convert_to_tensor(7, dtype=tf.int32)
        self.constraint_type = 'window'

    def __call__(self, query, state, prev_max_attentions):
        # query 上一帧的输出，也是输入，  state 输入状态， 之前的最大attention
        previous_alignments = state
        with variable_scope.variable_scope(None, 'Location_Sensitive_Attention', [query]):
            # Dense
            processed_query = self.query_layer(query) if self.query_layer else query
            processed_query = tf.expand_dims(processed_query, 1)
            # [batch_size, 1, attention_dim]

            expanded_alignments = tf.expand_dims(previous_alignments, axis=2)
            # [batch_size, max_time, 1]
            f = self.location_convolution(expanded_alignments)
            processed_location_features = self.location_layer(f)

            energy = _location_sensitive_score(processed_query, processed_location_features, self.keys)



def _location_sensitive_score(W_query, W_fil, W_keys):
    dtype = W_query.dtype  # 上一个时间步的输出
    num_units = W_keys.shape[-1].value
    v_a = tf.get_variable(
        'attention_variable_projection', shape=[num_units], dtype=dtype,
        initializer=tf.contrib.layers.xavier_initializer())
    b_a = tf.get_variable(
        'attention_bias', shape=[num_units], dtype=dtype,
        initializer=tf.zeros_initializer())
    return tf.reduce_sum(v_a * tf.tanh(W_query + W_keys + W_fil + b_a), [2])

def _compute_attention(attention_mechanism, cell_output, attention_state, attention_layer, prev_max_attentions):
    # self._attention_mechanism, LSTM_output, previous_alignments, attention_layers=None, prev_max_attentions=state.max_attentions
    #return  context_vector, alignments, cumulated_alignmnets, max_attentions = _compute_attention()
    alignments, next_attention_state, max_attentions = attention_mechanism(cell_output, state=attention_state, prev_max_attentions=prev_max_attentions)

class TacotronDecoderCell(tf.contrib.rnn.RNNCell):
    def __init__(self, prenet, attention_mechanism, rnn_cell, frame_projection, stop_projection):
        super(TacotronDecoderCell, self).__init__()
        self._prenet = prenet
        self._attention_mechanism = attention_mechanism
        self._cell = rnn_cell
        self._frame_projection = frame_projection
        self._stop_projection = stop_projection
        self._attention_layer_size = self._attention_mechanism.values.get_shape()[-1].value

    def batch_size_checks(self, batch_size, error_message):
        pass

    def __call__(self, inputs, state):
        
        :param inputs: 上一帧的输出/上一帧的真值
        :param state:  上一帧的state
        :return:
        
        prenet_output = self._prenet(inputs)
        LSTM_input = tf.concat([prenet_output, state.attention], axis=-1)

        previous_alignments = state.alignments
        previous_alignment_history = state.alignment_history
        context_vector, alignments, cumulated_alignmnets, max_attentions = _compute_attention(self._attention_mechanism,
                                                                                              LSTM_output,
                                                                                              previous_alignments,
                                                                                              attention_layer=None,
                                                                                              prev_max_attentions=state.max_attentions)

prenet = Prenet(is_training, layers_sizes=[256, 256], drop_rate=0.5, scope='decoder_prenet')
attention_mechanism = LocationSensitiveAttention(128, encoder_outputs, hparams=hp, is_training=is_training,
                                                 mask_encoder=True, memory_sequence_length=tf.reshape(input_lengths, [-1]), smoothing=False,
                                                 cumulate_weights=True)
decoder_lstm = DecoderRNN(is_training, layers=2, size=1024, zoneout=0.1, scope='decoder_LSTM')
frame_proj = FrameProjection(hparams.num_mels * hparams.outputs_per_step, scope='linear_transform_projection')
stop_proj = StopProjection(is_training or is_evaluating, shape=hparams.outputs_per_step, scope='stop_token_projection')


decoder_cell = TacotronDecoderCell(
    prenet,
    attention_mechanism,
    decoder_lstm,
    frame_proj,
    stop_proj
)
"""
"""
def decoder_LSTM(is_training, layers=2):
    rnn_layers = [ZoneoutLSTMCell(512,
                                  is_training,
                                  zoneout_factor_cell=0.1,
                                  zoneout_factor_output=0.1,
                                  exp_proj=hparams.num_mels) for i in range(2)]
    LSTM_Cell = tf.nn.rnn_cell.MultiRNNCell(rnn_layers)
    return LSTM_Cell


class MyAttentionWrapper(rnn_cell_impl.RNNCell):
    def __init__(self,
                 cell,  # 2层LSTM
                 attention_mechanism,
                 attention_layer_size=None,
                 alignment_history=False,
                 cell_input_fn=None,
                 output_attention=True,
                 initial_cell_state=None,
                 name=None):
        super(MyAttentionWrapper, self).__init__(name=name)
        if isinstance(attention_mechanism, (list, tuple)):
            self._is_multi = True
            attention_mechanisms = attention_mechanism
            for attention_mechanism in attention_mechanisms:
                pass

        else:
            self._is_multi = False
            attention_mechanisms = (attention_mechanism, )

        if cell_input_fn is None:
            cell_input_fn = (lambda inputs, attention: array_ops.concat([inputs, attention], -1))

        if attention_layer_size is not None:
            pass
        else:
            self._attention_layers = None
            self._attention_layer_size = attention_mechanism.values.get_shape()[-1].value

        self._cell = cell
        self._attention_mechanisms = attention_mechanisms
        self._cell_input_fn = cell_input_fn
        self._output_attention = output_attention
        self._alignment_history = alignment_history
        with ops.name_scope(name, 'AttentionWrapperInit'):
            if initial_cell_state is None:
                self._initial_cell_state = None
            else:
                final_state_tensor = nest.flatten(initial_cell_state)[-1]
                state_batch_size = (final_state_tensor.shape[0].value)
                error_message = ("err")
                with ops.control_dependencies(self._batch_size_check(state_batch_size, error_message)):
                    self._initial_cell_state = nest.map_structure(lambda s: array_ops.identity(s, name='check_initial_cell_state'),
                                                                  initial_cell_state)

    def call(self, inputs, state):
        inputs = dec_prenet(inputs, [256, 256], scope='decoder_prenet')

        # concat([inputs, state.attention])
        cell_inputs = self._cell_input_fn(inputs, state.attention)
        
        LSTM_output, next_cell_state = self._cell(cell_inputs, state.cell_state)
        
        concat_output_LSTM = tf.concat([LSTM_output, state.attention], axis=-1)
        
        cell_output = projection(concat_output_LSTM, num_mels, scope='decoder_projection_layer')
        
        

    def _batch_size_checks(self, batch_size, error_message):
        return [check_ops.assert_equal(batch_size, attention_mechanism.batch_size, message=error_message) for attention_mechanism in self._attention_mechanisms]

def dec_prenet(inputs, layers_size, scope=None):
    result = inputs
    drop_rate = 0.5
    with tf.variable_scope(scope):
        for i, size in enumerate(layers_size):
            fc = tf.layers.dense(result, units=size, activation=tf.nn.relu,name='dec_FC_%d' % (i + 1))
            result = tf.layers.dropout(fc, rate=drop_rate, name='dropout_%d' % (i + 1))
            return result
        

    if attention_styple is 'Hybrid':
        attention_mechanism = HybridAttention(128, encoder_outputs)
    else:
        attention_mechanism = LocationBasedAttention(128, encoder_outputs)
    
        attention_decoder = MyAttentionWrapper(
            decoder_LSTM(is_training, layers=2),
            attention_mechanism,
            alignment_history=True,
            output_attention=False)


from keras.datasets import imdb

NUM_WORDS = 10000
INDEX_FROM = 3
SEQUENCE_LENGTH = 250
EMBEDDING_DIM = 100
HIDDEN_SIZE = 150
ATTENTION_SIZE = 50
KEEP_PROB = 0.8
BATCH_SIZE = 256
NUM_EPOCHS = 3  # Model easily overfits without pre-trained words embeddings, that's why train for a few epochs
DELTA = 0.5
MODEL_PATH = './model'

(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=NUM_WORDS, index_from=INDEX_FROM)


vocabulary_size = max([max(x) for x in X_train]) + 1
X_test = [[w for w in x if w < vocabulary_size] for x in X_test]
X_train = np.array([x[:SEQUENCE_LENGTH - 1] + [0] * max(SEQUENCE_LENGTH - len(x), 1) for x in X_train])
X_test = np.array([x[:SEQUENCE_LENGTH - 1] + [0] * max(SEQUENCE_LENGTH - len(x), 1) for x in X_test])


with tf.name_scope("inputs"):
    batch_ph = tf.palceholder(tf.int32, [None, SEQUENCE_LENGTH], name='batch_ph')
    target_ph = tf.placeholder(tf.float32, [None], name='target_ph')
    seq_len_ph = tf.placeholder(tf.int32, [None], name='seq_len_ph')
    keep_prob_ph = tf.placeholder(tf.float32, name='keep_prob_ph')

with tf.name_scope("Embedding_layer"):
    embeddings_var = tf.Variable(tf.random_uniform([vocabulary_size, EMBEDDING_DIM], -1.0, 1.0), trainable=True)
    tf.summary.histogram('embeddings_var', embeddings_var)
    batch_embedded = tf.nn.embedding_lookup(embeddings_var, batch_ph)

rnn_outputs, _ = tf.nn.bidirectional_dynamic_rnn(tf.contrib.rnn.GRUCell(HIDDEN_SIZE), tf.contrib.rnn.GRUCell(HIDDEN_SIZE),
                                                 inputs=batch_embedded,
                                                 sequence_length=seq_len_ph,
                                                 dtype=tf.float32)
tf.summary.histogram('RNN_outputs', rnn_outputs)

def attention(inputs, attention_size, time_major=False):
    # rnn_outputs, attention_size=50
    if isinstance(inputs, tuple):
        inputs = tf.concat(inputs, 2)

    if time_major:
        # (time, batch_size, dims) -> (batch_size, time, dims)
        inputs = tf.array_ops.transpose(inputs, [1, 0, 2])

    # dims
    hidden_size = inputs.shape[2].value

    w = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1))
    b = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
    u = tf.Variable(tf.random_normal([attention_size], stddev=0.1))

    with tf.name_scope('v'):
        # e = g(St-1 * hj)
        v = tf.tanh(tf.tensordot(inputs, w, axes=1) + b)

    # e = g(e)
    e = tf.tensordot(v, u, axes=1, name='vu')
    # apphas = softmax(e)
    alphas = tf.nn.softmax(e, name='alphas')

    # c = sum[hj, aij]
    output = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), 1)

    return output, alphas


with tf.name_scope('Attention_layer'):
    attention_output, alphas = attention(rnn_outputs, ATTENTION_SIZE)
    tf.summary.histogram('alphas', alphas)


cell = [rnn.LSTMCell(cell_size) for i in range(num_layers)]
mutli_layer = rnn.MultiRNNCell(cell)

attention = BahdanauAttention(num_units=num_units, memory=context)
att_warpper = AttentionWrapper(cell=mutli_layer,
                               attention_mechanism=attention,
                               attention_layer_size=att_size,
                               cell_input_fn=lambda intput, attention: input)
states = att_warpper.zeros_state(batch_size, tf.float32)
with tf.variable_scope(SCOPE, reuse=tf.AUTO_REUSE):
    for i in range(decoder_time):
        h_bar_withouttanh, states = att_warpper(_X, states)
        h_bar = tf.tanh(h_bar_withouttanh)
        _x = tf.nn.softmax(tf.matmul(h_bar, W), 1)



batch_size = 5
encoder_input = tf.placeholder(shape=[batch_size, None], dtype=tf.int32)
decoder_target = tf.placeholder(shape=[batch_size, None], dtype=tf.int32)

input_vocab_size = 10
target_vocab_size = 10
input_embedding_size = 20
target_embedding_size = 20

encoder_input_ = tf.one_hot(encoder_input, depth=input_vocab_size, dtype=tf.int32)
decoder_target_ = tf.one_hot(decoder_target, depth=target_vocab_size, dtype=tf.int32)

input_embedding = tf.Variable(tf.random_uniform(shape=[input_vocab_size, input_embedding_size]), dtype=tf.float32)
target_embedding = tf.Variable(tf.random_uniform(shape=[target_vocab_size, target_embedding_size]), dtype=tf.float32)

input_embedded = tf.nn.embedding_lookup(input_embedding, encoder_input_)
target_embedded = tf.nn.embedding_lookup(target_embedding, decoder_target_)

rnn_hidden_size = 20
cell = tf.nn.rnn_cell.RGUCell(num_units=rnn_hidden_size)
init_state = cell.zero_state(batch_size, dtype=tf.float32)
encoder_output, encoder_state = tf.nn.dynamic_rnn(cell, input_embedded, init_state)


seq_len = tf.constant([3, 4, 5, 2, 3], tf.int32)
inference = False
if not inference:
    helper = tf.contrib.seq2seq.TrainingHelper(target_embedded, seq_len)
else:
    helper = tf.contrib.seq2seq.InferenceHelper(target_embedded, seq_len)

d_cell = tf.nn.rnn_cell.GRUCell(rnn_hidden_size)

attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(rnn_hidden_size, encoder_output)
decoder_cell = tf.contrib.seq2seq.AttentionWrapper(d_cell, attention_mechanism)
de_state = decoder_cell.zero_state(batch_size, dtype=tf.float32)
out_cell = tf.contrib.rnn.OutputProjectionWrapper(decoder_cell, target_vocab_size)

decoder = tf.contrib.seq2seq.BasicDecoder(
    out_cell, helper, de_state, tf.layers.Dense(target_embedding_size)
)

final_outputs, final_state, final_sequence_lengths = tf.contrib.seq2seq.dynamic_decoder(decoder, swap_memory=True)
"""

class MyAttenttionWrapper(rnn_cell_impl.RNNCell):
    def __init__(self,
                 cell,
                 attention_mechanism,
                 attention_layer_size=None,
                 alignment_history=True,
                 cell_input_fn=None,
                 output_attention=False,
                 initial_cell_state=None,
                 name=None):

        super(MyAttenttionWrapper, self).__init__()
        self._is_multi = False
        attention_mechansims = (attention_mechanism, )

        if cell_input_fn is None:
            cell_input_fn = (lambda inputs, attention: array_ops.concat([inputs, attention], -1))

            self._attention_layer_size = attention_mechanism.values.get_shape()[-1].value

        self._cell = cell
        self._attention_mechanism = attention_mechanism
        self._cell_input_fn = cell_input_fn
        self._output_attention = False
        self.alignment_history = True
        with ops.name_scope(name, "AttentionWrapperInit"):
            if initial_cell_state is None:
                self._initial_cell_state = None
            else:
                pass

    def _batch_size_checks(self, batch_size, error_message):
        return [check_ops.assert_equal(batch_size, attention_mechanism.batch_size, message=error_message)]

    def call(self, inputs, state):
        pass

class AttentionMechanism(object):
    pass

class _BaseAttentionMechanism(AttentionMechanism):
    def __init__(self,
                 query_layer,
                 memory,
                 probability_fn,
                 memory_sequence_length=None,
                 memory_layer=None,
                 check_inner_dims_defined=True,
                 score_mask_value=float("-inf"),
                 name=None):
        self._query_layer = query_layer
        self._memory_layer = memory_layer
        self._probability_fn = lambda score, prev:(
            probability_fn(
                _maybe_mask_score(score, memory_sequence_length, score_mask_value),
                prev))
        with ops.name_scope(name, "BaseAttentionMechanismInit", nest.flatten(memory)):
            self._values = _prepare_memory()


class HybridAttention(_BaseAttentionMechanism):
    @property
    def batch_size(self):
        return self._batch_size



attention_mechanism = HybridAttenttion()

attention_decoder = MyAttentionWrapper(
    tf.nn.rnn_cell.MultiRNNCell([ZoneoutLSTMCell(512, is_training, 0.5, ext_proj=num_mels) for i in range(layer_num)]),
    attention_mechanism,
    alignment_history=True,
    output_attention=False
)


