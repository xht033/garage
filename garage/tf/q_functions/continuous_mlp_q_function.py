import tensorflow as tf

from garage.core import Serializable
from garage.misc.overrides import overrides
from garage.tf.core.mlp import mlp_concat
from garage.tf.q_functions import QFunction


class ContinuousMLPQFunction(QFunction, Serializable):
    """
    This class implements a q value network to predict q based on the input
    state and action. It uses an MLP to fit the function of Q(s, a).
    """

    def __init__(self,
                 env_spec,
                 name="ContinuousMLPQFunction",
                 hidden_sizes=(32, 32),
                 hidden_nonlinearity=tf.nn.relu,
                 action_merge_layer=-2,
                 output_nonlinearity=None,
                 input_include_goal=False,
                 layer_norm=False):
        """
        Initialize class with multiple attributes.

        Args:
            name(str, optional): A str contains the name of the policy.
            hidden_sizes(list or tuple, optional):
                A list of numbers of hidden units for all hidden layers.
            hidden_nonlinearity(optional):
                An activation shared by all fc layers.
            action_merge_layer(int, optional):
                An index to indicate when to merge action layer.
            output_nonlinearity(optional):
                An activation used by the output layer.
            layer_norm(bool, optional):
                A bool to indicate whether normalize the layer or not.
        """
        super().__init__()
        Serializable.quick_init(self, locals())

        self.env_spec = env_spec
        self.name = name
        self._hidden_sizes = hidden_sizes
        self._hidden_nonlinearity = hidden_nonlinearity
        self._output_nonlinearity = output_nonlinearity
        self._layer_norm = layer_norm

        with tf.name_scope(name):
            self.q_val, self.obs_ph, self.act_ph = self.build_net(name)

    def build_ph(self, input_include_goal=False):
        if input_include_goal:
            obs_dim = self.env_spec.observation_space.flat_dim_with_keys(
                ["observation", "desired_goal"])
        else:
            obs_dim = self.env_spec.observation_space.flat_dim
        obs_ph = tf.placeholder(
            tf.float32, shape=(None, obs_dim), name="input_observation")
        act_ph = tf.placeholder(
            tf.float32,
            shape=(None, self.env_spec.action_space.flat_dim),
            name="input_action")

        return obs_ph, act_ph

    @overrides
    def build_net(self, name):
        """
        Set up q network based on class attributes. This function uses layers
        defined in garage.tf.

        Args:
            name: Variable scope of the network.
        """
        obs_ph, act_ph = self.build_ph()

        network = mlp_concat(
            input_var=obs_ph,
            input_var_2=act_ph,
            output_dim=1,
            hidden_sizes=self._hidden_sizes,
            name=name,
            concat_layer=-2,
            hidden_nonlinearity=self._hidden_nonlinearity,
            output_nonlinearity=self._output_nonlinearity,
            layer_normalization=self._layer_norm)

        return network, obs_ph, act_ph

    @overrides
    def get_qval_sym(self, input_phs):
        assert len(input_phs) == 2
        obs_ph, act_ph = input_phs

        return mlp_concat(
            input_var=obs_ph,
            input_var_2=act_ph,
            output_dim=1,
            hidden_sizes=self._hidden_sizes,
            name=self.name,
            concat_layer=-2,
            hidden_nonlinearity=self._hidden_nonlinearity,
            output_nonlinearity=self._output_nonlinearity,
            layer_normalization=self._layer_norm,
            reuse=True)
