import tensorflow as tf
from models.tacotron2 import Tacotron2
from hparams import hparams

def train():
    pass

    step_count = 0
    global_step = tf.Variable(step_count, name='global_step', trainable=False)
    with tf.variable_scope('model') as scope:
        model = Tacotron2(hparams)
        model.initialize()
        model.add_loss()
        model.add_optimizer(global_step)



if __name__ == '__main__':
    train()

