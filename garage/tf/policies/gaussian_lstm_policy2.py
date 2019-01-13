"""GaussianLSTMPolicy."""
import numpy as np
import tensorflow as tf

from garage.core import Serializable
from garage.misc.overrides import overrides
from garage.tf.core.lstm import lstm
from garage.tf.core.parameter import parameter
from garage.tf.distributions import RecurrentDiagonalGaussian
from garage.tf.policies import StochasticPolicy
from garage.tf.spaces import Box


class GaussianLSTMPolicy2(StochasticPolicy, Serializable):
    """
    GaussianLSTMPolicy.

    This is GaussianLSTMPolicy.

    :param env_spec: A spec for the env.
    :param hidden_dim: dimension of hidden layer
    :param hidden_nonlinearity: nonlinearity used for each hidden layer

    """

    def __init__(
            self,
            env_spec,
            name="GaussianLSTMPolicy",
            hidden_dim=32,
            feature_network=None,
            state_include_action=True,
            hidden_nonlinearity=tf.tanh,
            learn_std=True,
            init_std=1.0,
            output_nonlinearity=None,
            use_peepholes=False,
            std_share_network=False,
    ):
        assert isinstance(env_spec.action_space, Box)

        self._mean_network_name = "mean_network"
        self._std_network_name = "std_network"
        self._hidden_dim = hidden_dim
        with tf.variable_scope(name, "GaussianLSTMPolicy"):
            Serializable.quick_init(self, locals())
            super().__init__(env_spec)

            obs_dim = env_spec.observation_space.flat_dim
            action_dim = env_spec.action_space.flat_dim
            self.action_dim = action_dim

            if state_include_action:
                input_dim = obs_dim + action_dim
            else:
                input_dim = obs_dim

            self.input_ph = tf.placeholder(tf.float32, (None, None, input_dim),
                                           "input_ph")
            self.step_input_ph = tf.placeholder(tf.float32, (None, input_dim),
                                                "step_input_ph")
            self.hidden_ph = tf.placeholder(tf.float32, (None, hidden_dim),
                                            "hidden_ph")
            self.cell_ph = tf.placeholder(tf.float32, (None, hidden_dim),
                                          "cell_ph")

            self.hidden_init_value = tf.get_variable(
                initializer=tf.zeros_initializer(),
                shape=(hidden_dim, ),
                name="h0")
            self.cell_init_value = tf.get_variable(
                initializer=tf.zeros_initializer(),
                shape=(hidden_dim, ),
                name="c0")

            self.mean, self.step_hidden, self.step_cell, self.step_mean = lstm(
                input_var=self.input_ph,
                step_input_var=self.step_input_ph,
                hidden_var=self.hidden_ph,
                cell_var=self.cell_ph,
                output_dim=action_dim,
                hidden_size=hidden_dim,
                hidden_nonlinearity=hidden_nonlinearity,
                use_peepholes=use_peepholes,
                output_nonlinearity=output_nonlinearity,
                name="lstm_mean_network")

            self.log_std = parameter(
                input_var=self.input_ph,
                length=action_dim,
                initializer=tf.constant_initializer(np.log(init_std)),
                trainable=learn_std,
                name="output_log_std")

            self.step_log_std = parameter(
                input_var=self.step_input_ph,
                length=action_dim,
                initializer=tf.constant_initializer(np.log(init_std)),
                trainable=learn_std,
                name="step_output_log_std")

            self.state_include_action = state_include_action
            self.name = name

            self.input_dim = input_dim
            self.action_dim = action_dim
            self.hidden_dim = hidden_dim

            self.prev_actions = None
            self.prev_hiddens = None
            self.prev_cells = None
            self.dist = RecurrentDiagonalGaussian(action_dim)

            self.params = [self.mean, self.log_std]

    @overrides
    def dist_info_sym(self, obs_var, state_info_vars, name=None):
        """dist_info_sym."""
        with tf.name_scope(name, "dist_info_sym"):
            # n_batches = tf.shape(obs_var)[0]
            # n_steps = tf.shape(obs_var)[1]
            # obs_var = tf.reshape(obs_var, tf.stack([n_batches, n_steps, -1]))
            if self.state_include_action:
                prev_action_var = state_info_vars["prev_action"]
                all_input_var = tf.concat(
                    axis=2, values=[obs_var, prev_action_var])
            else:
                all_input_var = obs_var

        with tf.variable_scope("LSTMCell", reuse=tf.AUTO_REUSE):
            _lstm = tf.nn.rnn_cell.LSTMCell(
                num_units=self._hidden_dim, name="lstm_cell")
            means, _ = tf.nn.dynamic_rnn(
                _lstm, all_input_var, dtype=np.float32)
            means = tf.reshape(means, [-1, self._hidden_dim])
            means = tf.layers.dense(
                inputs=means, units=self.action_dim, name="output")
            means = tf.reshape(
                means,
                tf.stack((tf.shape(all_input_var)[0],
                          tf.shape(all_input_var)[1], -1)))

            log_stds = parameter(
                input_var=all_input_var,
                length=self.action_dim,
                name="log_std")

            return dict(mean=means, log_std=log_stds)

    def get_params_internal(self, **tags):
        """get_params_internal."""
        return self.params

    @property
    def vectorized(self):
        """vectorized."""
        return True

    def reset(self, dones=None):
        """reset."""
        if dones is None:
            dones = [True]
        dones = np.asarray(dones)
        if self.prev_actions is None or len(dones) != len(self.prev_actions):
            self.prev_actions = np.zeros((len(dones),
                                          self.action_space.flat_dim))
            self.prev_hiddens = np.zeros((len(dones), self.hidden_dim))
            self.prev_cells = np.zeros((len(dones), self.hidden_dim))

        self.prev_actions[dones] = 0.
        self.prev_hiddens[dones] = self.hidden_init_value.eval()
        self.prev_cells[dones] = self.cell_init_value.eval()

    # The return value is a pair. The first item is a matrix (N, A), where each
    # entry corresponds to the action value taken. The second item is a vector
    # of length N, where each entry is the density value for that action, under
    # the current policy
    @overrides
    def get_action(self, observation):
        """get_action."""
        actions, agent_infos = self.get_actions([observation])
        return actions[0], {k: v[0] for k, v in agent_infos.items()}

    @overrides
    def get_actions(self, observations):
        """get_actions."""
        sess = tf.get_default_session()

        flat_obs = self.observation_space.flatten_n(observations)
        if self.state_include_action:
            assert self.prev_actions is not None
            all_input = np.concatenate([flat_obs, self.prev_actions], axis=-1)
        else:
            all_input = flat_obs

        with tf.name_scope(self._mean_network_name):
            means, hidden_vec, cell_vec, log_stds = sess.run(
                [
                    self.step_mean, self.step_hidden, self.step_cell,
                    self.step_log_std
                ],
                feed_dict={
                    self.step_input_ph: all_input,
                    self.hidden_ph: self.prev_hiddens,
                    self.cell_ph: self.prev_cells
                })
            # means = tf.identity(means, "step_mean")
            # out_step_hidden = tf.identity(out_step_hidden, "step_hidden")
            # out_mean_cell = tf.identity(out_mean_cell, "mean_cell")
            # out_step_log_std = tf.identity(out_step_log_std,
            #                                "step_log_std")

        rnd = np.random.normal(size=means.shape)
        actions = rnd * np.exp(log_stds) + means
        prev_actions = self.prev_actions
        self.prev_actions = self.action_space.flatten_n(actions)
        self.prev_hiddens = hidden_vec
        self.prev_cells = cell_vec
        agent_info = dict(mean=means, log_std=log_stds)
        if self.state_include_action:
            agent_info["prev_action"] = np.copy(prev_actions)
        return actions, agent_info

    @property
    @overrides
    def recurrent(self):
        """recurrent."""
        return True

    @property
    def distribution(self):
        """distribution."""
        return self.dist

    @property
    def state_info_specs(self):
        """state_info_specs."""
        if self.state_include_action:
            return [
                ("prev_action", (self.action_dim, )),
            ]
        else:
            return []
