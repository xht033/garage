"""
This modules creates a continuous MLP policy network.

A continuous MLP network can be used as policy method in different RL
algorithms. It accepts an observation of the environment and predicts an
action.
"""
import tensorflow as tf

from garage.core import Serializable
from garage.misc.overrides import overrides
from garage.tf.core.mlp import mlp
from garage.tf.policies import Policy
from garage.tf.spaces import Box


class ContinuousMLPPolicy(Policy, Serializable):
    """
    This class implements a policy network.

    The policy network selects action based on the state of the environment.
    It uses neural nets to fit the function of pi(s).
    """

    def __init__(self,
                 env_spec,
                 hidden_sizes=(64, 64),
                 name="ContinuousMLPPolicy",
                 hidden_nonlinearity=tf.nn.relu,
                 output_nonlinearity=tf.nn.tanh,
                 input_include_goal=False,
                 layer_norm=False):
        """
        Initialize class with multiple attributes.

        Args:
            env_spec():
            hidden_sizes(list or tuple, optional):
                A list of numbers of hidden units for all hidden layers.
            name(str, optional):
                A str contains the name of the policy.
            hidden_nonlinearity(optional):
                An activation shared by all fc layers.
            output_nonlinearity(optional):
                An activation used by the output layer.
            bn(bool, optional):
                A bool to indicate whether normalize the layer or not.
        """
        assert isinstance(env_spec.action_space, Box)

        Serializable.quick_init(self, locals())
        super().__init__(env_spec)

        self.name = name
        if input_include_goal:
            self._obs_dim = env_spec.observation_space.flat_dim_with_keys(
                ["observation", "desired_goal"])
        else:
            self._obs_dim = env_spec.observation_space.flat_dim
        self._action_dim = env_spec.action_space.flat_dim
        self._action_bound = env_spec.action_space.high
        self._hidden_sizes = hidden_sizes
        self._hidden_nonlinearity = hidden_nonlinearity
        self._output_nonlinearity = output_nonlinearity
        self._layer_norm = layer_norm
        self._policy_network_name = "policy_network"

        with tf.name_scope(name):
            self.f_prob, self.obs_ph = self.build_net(name)

    def build_ph(self, input_include_goal=False):
        if input_include_goal:
            obs_dim = self.env_spec.observation_space.flat_dim_with_keys(
                ["observation", "desired_goal"])
        else:
            obs_dim = self.env_spec.observation_space.flat_dim
        obs_ph = tf.placeholder(tf.float32, shape=(None, obs_dim), name="obs")

        return obs_ph

    def build_net(self, name):
        """
        Set up q network based on class attributes.

        This function uses layers defined in garage.tf.

        Args:
            reuse: A bool indicates whether reuse variables in the same scope.
            trainable: A bool indicates whether variables are trainable.
        """
        obs_ph = self.build_ph()

        network = mlp(
            input_var=obs_ph,
            output_dim=self._action_dim,
            hidden_sizes=self._hidden_sizes,
            name=name,
            hidden_nonlinearity=self._hidden_nonlinearity,
            output_nonlinearity=self._output_nonlinearity,
            layer_normalization=self._layer_norm)

        return network, obs_ph

    def get_f_sym(self, input_phs, name=None):
        assert len(input_phs) == 1
        obs_ph = input_phs

        name = name if name else self.name
        mlp_policy = mlp(
            input_var=obs_ph,
            output_dim=self._action_dim,
            hidden_sizes=self._hidden_sizes,
            name=name,
            hidden_nonlinearity=self._hidden_nonlinearity,
            output_nonlinearity=self._output_nonlinearity,
            layer_normalization=self._layer_norm,
            reuse=True)

        with tf.name_scope(self._policy_network_name):
            scaled_action = tf.multiply(
                mlp_policy, self._action_bound, name="scaled_action")

        return scaled_action

    @overrides
    def get_action(self, observation):
        """Return a single action."""
        sess = tf.get_default_session()
        return sess.run(
            self.f_prob, feed_dict={self.obs_ph: observation})[0], dict()

    @overrides
    def get_actions(self, observations):
        """Return multiple actions."""
        sess = tf.get_default_session()
        return sess.run(
            self.f_prob, feed_dict={self.obs_ph: observations}), dict()

    @property
    def vectorized(self):
        return True

    def log_diagnostics(self, paths):
        pass

    def get_trainable_vars(self, scope=None):
        scope = scope if scope else self.name
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)

    def get_global_vars(self, scope=None):
        scope = scope if scope else self.name
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)

    def get_regularizable_vars(self, scope=None):
        scope = scope if scope else self.name
        reg_vars = [
            var for var in self.get_trainable_vars(scope=scope)
            if 'W' in var.name and 'output' not in var.name
        ]
        return reg_vars
