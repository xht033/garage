"""Soft Actor Critic."""

import tensorflow as tf

from garage.misc.overrides import overrides
from garage.tf.algos.off_policy_rl_algorithm import OffPolicyRLAlgorithm

class SAC(OffPolicyRLAlgorithm):
    def __init__(self, 
                 vf,
                 lr=1e-3,
                 optimizer=tf.train.AdamOptimizer,
                 tau=0.1,
                 name='SAC',
                 **kwargs):
        self.name = name
        self.vf = vf
        self.optimizer = optimizer(learning_rate=lr, name='AdamOptimizer')
        super(SAC, self).__init__(
            replay_buffer=None,
            use_target=True,
            **kwargs)

        @overrides
        def init_opt(self):
            obs_dim = self.env.observation_space.shape
            action_dim = self.env.action_space.shape

            obs = tf.placeholder(name='observation_batch', shape=(None, ) + obs_dim, dtype=tf.float32)
            actions = tf.placeholder(name='action_batch', shape=(None, ) + action_dim, dtype=tf.float32)
            rewards = tf.placeholder(name='reward_batch', shape=(None, ), dtype=tf.float32)
            next_obs = tf.placeholder(name='next_observation_batch', shape=(None, ) + obs_dim, dtype=tf.float32)

            target_vf = self.vf.model.build()
            action_prob = self.policy.model.network['default'].dist.log_prob(actions)

            with tf.name_scope('soft_qf_bellman_error'):
                target = rewards + self.discount * target_vf.model.network['default'].sample
                bellman_error = tf.square(self.qf.model.network['default'].qvalue - target)
                qf_loss = tf.reduce_mean(0.5 * bellman_error, axis=1)

            with tf.name_scope('soft_vf_bellman_error'):
                target = self.qf.model.network['default'].qvalue - action_prob
                bellman_error = tf.square(self.vf.model.network['default'].value - target)
                vf_loss = tf.reduce_mean(0.5 * bellman_error, axis=1)

            with tf.name_scope('soft_policy_loss'):
                policy_loss = action_prob - self.qf.model.network['default'].qvalue

        def train_once(self, itr, paths):
            pass

        @overrides
        def optimize_policy(self, itr, samples_data):
            pass