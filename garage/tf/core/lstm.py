"""lstm."""
import numpy as np
import tensorflow as tf


def lstm(input_var,
         step_input_var,
         hidden_var,
         cell_var,
         output_dim,
         hidden_size,
         hidden_nonlinearity=tf.nn.tanh,
         horizon=None,
         hidden_init_trainable=False,
         forget_bias=1.0,
         use_peepholes=False,
         output_nonlinearity=None,
         output_w_init=tf.initializers.random_normal(),
         output_b_init=tf.initializers.random_normal(),
         **kwargs):
    """To be done."""
    with tf.variable_scope("LSTMCell", reuse=tf.AUTO_REUSE):
        _lstm = tf.nn.rnn_cell.LSTMCell(
            num_units=hidden_size,
            activation=hidden_nonlinearity,
            state_is_tuple=True,
            forget_bias=forget_bias,
            name="lstm_cell")

        # simple step
        # output = hidden
        output, (cell, hidden) = _lstm(step_input_var, (cell_var, hidden_var))
        h_output = tf.layers.dense(
            inputs=output,
            units=output_dim,
            activation=output_nonlinearity,
            kernel_initializer=output_w_init,
            bias_initializer=output_b_init,
            name="step_output")

        # full rnn, i.e. steps for n times
        outputs, _ = tf.nn.dynamic_rnn(_lstm, input_var, dtype=np.float32)
        outputs = tf.reshape(outputs, [-1, hidden_size])
        output = tf.layers.dense(
            inputs=outputs,
            units=output_dim,
            activation=output_nonlinearity,
            kernel_initializer=output_w_init,
            bias_initializer=output_b_init,
            name="output")
        output = tf.reshape(
            output,
            tf.stack((tf.shape(input_var)[0], tf.shape(input_var)[1], -1)))
    return output, hidden, cell, h_output
