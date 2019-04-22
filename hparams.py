import tensorflow as tf

hparams = tf.contrib.training.HParams(






    drop_prob=0.5,

    encoder_channel_size=[512, 512, 512],
    encoder_drop_rate=0.5,

    encoder_BiLSTM_units=256,
    zoneout_factor_cell=0.1,
    zoneout_factor_output=0.1,

)