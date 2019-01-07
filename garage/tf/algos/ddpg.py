"""
This module implements a DDPG model.

DDPG, also known as Deep Deterministic Policy Gradient, uses actor-critic
method to optimize the policy and reward prediction. It uses a supervised
method to update the critic network and policy gradient to update the actor
network. And there are exploration strategy, replay buffer and target networks
involved to stabilize the training process.
"""
from collections import deque

import numpy as np
import tensorflow as tf
import tensorflow.contrib as tc

import garage.misc.logger as logger
from garage.misc.overrides import overrides
from garage.tf.algos.off_policy_rl_algorithm import OffPolicyRLAlgorithm


class DDPG(OffPolicyRLAlgorithm):
    """
    A DDPG model based on https://arxiv.org/pdf/1509.02971.pdf.

    Example:
        $ python garage/examples/tf/ddpg_pendulum.py
    """

    def __init__(self,
                 env,
                 replay_buffer,
                 target_update_tau=0.01,
                 policy_lr=1e-4,
                 qf_lr=1e-3,
                 policy_weight_decay=0,
                 qf_weight_decay=0,
                 policy_optimizer=tf.train.AdamOptimizer,
                 qf_optimizer=tf.train.AdamOptimizer,
                 clip_pos_returns=False,
                 clip_return=np.inf,
                 discount=0.99,
                 max_action=None,
                 name=None,
                 **kwargs):
        """
        Construct class.

        Args:
            env(): Environment.
            target_update_tau(float): Interpolation parameter for doing the
        soft target update.
            discount(float): Discount factor for the cumulative return.
            policy_lr(float): Learning rate for training policy network.
            qf_lr(float): Learning rate for training q value network.
            policy_weight_decay(float): L2 weight decay factor for parameters
        of the policy network.
            qf_weight_decay(float): L2 weight decay factor for parameters
        of the q value network.
            policy_optimizer(): Optimizer for training policy network.
            qf_optimizer(): Optimizer for training q function network.
            clip_pos_returns(boolean): Whether or not clip positive returns.
            clip_return(float): Clip return to be in [-clip_return,
        clip_return].
            max_action(float): Maximum action magnitude.
            name(str): Name of the algorithm shown in computation graph.
        """
        action_bound = env.action_space.high
        self.max_action = action_bound if max_action is None else max_action
        self.tau = target_update_tau
        self.policy_lr = policy_lr
        self.qf_lr = qf_lr
        self.policy_weight_decay = policy_weight_decay
        self.qf_weight_decay = qf_weight_decay
        self.policy_optimizer = policy_optimizer
        self.qf_optimizer = qf_optimizer
        self.name = name
        self.clip_pos_returns = clip_pos_returns
        self.clip_return = clip_return
        self.success_history = deque(maxlen=100)

        super().__init__(
            env=env,
            replay_buffer=replay_buffer,
            use_target=True,
            discount=discount,
            **kwargs)

    @overrides
    def init_opt(self):
        with tf.name_scope(self.name, "DDPG"):
            # Create target policy and qf network
            self.target_policy_f_prob_online, self._target_policy_obs_ph = self.policy.build_net(  # noqa: E501
                name="target_policy")
            self.target_qf_q_val, self._target_qf_obs_ph, self._target_qf_act_ph = self.qf.build_net(  # noqa: E501
                name="target_qf")

            # Set up target init and update function
            with tf.name_scope("setup_target"):
                policy_init_ops, policy_update_ops = get_target_ops(
                    self.policy.get_global_vars(),
                    self.policy.get_global_vars("target_policy"), self.tau)
                qf_init_ops, qf_update_ops = get_target_ops(
                    self.qf.get_global_vars(),
                    self.qf.get_global_vars("target_qf"), self.tau)
                target_init_op = policy_init_ops + qf_init_ops
                target_update_op = policy_update_ops + qf_update_ops

            self.f_init_target = target_init_op
            self.f_update_target = target_update_op

            # Set up policy training function
            # reuse the qf using action from policy
            self.policy_qf = self.qf.get_qval_sym((self.policy.obs_ph,
                                                   self.policy.f_prob))

            with tf.name_scope("action_loss"):
                action_loss = -tf.reduce_mean(self.policy_qf)
                if self.policy_weight_decay > 0.:
                    policy_reg = tc.layers.apply_regularization(
                        tc.layers.l2_regularizer(self.policy_weight_decay),
                        weights_list=self.policy.get_regularizable_vars())
                    action_loss += policy_reg
            with tf.name_scope("minimize_action_loss"):
                policy_train_op = self.policy_optimizer(
                    self.policy_lr, name="PolicyOptimizer").minimize(
                        action_loss, var_list=self.policy.get_trainable_vars())

            self.f_train_policy = policy_train_op, action_loss

            # Set up qf training function
            self.y = tf.placeholder(
                tf.float32, shape=(None, 1), name="input_y")

            with tf.name_scope("qval_loss"):
                qval_loss = tf.reduce_mean(
                    tf.squared_difference(self.y, self.qf.q_val))
                if self.qf_weight_decay > 0.:
                    qf_reg = tc.layers.apply_regularization(
                        tc.layers.l2_regularizer(self.qf_weight_decay),
                        weights_list=self.qf.get_regularizable_vars())
                    qval_loss += qf_reg

            with tf.name_scope("minimize_qf_loss"):
                qf_train_op = self.qf_optimizer(
                    self.qf_lr, name="QFunctionOptimizer").minimize(
                        qval_loss, var_list=self.qf.get_trainable_vars())

            self.f_train_qf = qf_train_op, qval_loss, self.qf.q_val

    @overrides
    def train(self, sess=None):
        created_session = True if (sess is None) else False
        if sess is None:
            self.sess = tf.Session()
            self.sess.__enter__()

        self.sess.run(tf.global_variables_initializer())
        self.start_worker(self.sess)

        if self.use_target:
            self.sess.run(self.f_init_target, feed_dict=dict())

        episode_rewards = []
        episode_policy_losses = []
        episode_qf_losses = []
        epoch_ys = []
        epoch_qs = []
        last_average_return = None

        for epoch in range(self.n_epochs):
            self.success_history.clear()
            with logger.prefix('epoch #%d | ' % epoch):
                for epoch_cycle in range(self.n_epoch_cycles):
                    paths = self.obtain_samples(epoch)
                    samples_data = self.process_samples(epoch, paths)
                    episode_rewards.extend(
                        samples_data["undiscounted_returns"])
                    self.success_history.extend(
                        samples_data["success_history"])
                    self.log_diagnostics(paths)
                    for train_itr in range(self.n_train_steps):
                        if self.replay_buffer.n_transitions_stored >= self.min_buffer_size:  # noqa: E501
                            self.evaluate = True
                            qf_loss, y, q, policy_loss = self.optimize_policy(
                                epoch, samples_data)

                            episode_policy_losses.append(policy_loss)
                            episode_qf_losses.append(qf_loss)
                            epoch_ys.append(y)
                            epoch_qs.append(q)

                logger.log("Training finished")
                logger.log("Saving snapshot #{}".format(epoch))
                params = self.get_itr_snapshot(epoch, samples_data)
                logger.save_itr_params(epoch, params)
                logger.log("Saved")
                if self.evaluate:
                    logger.record_tabular('Epoch', epoch)
                    logger.record_tabular('AverageReturn',
                                          np.mean(episode_rewards))
                    logger.record_tabular('StdReturn', np.std(episode_rewards))
                    logger.record_tabular('Policy/AveragePolicyLoss',
                                          np.mean(episode_policy_losses))
                    logger.record_tabular('QFunction/AverageQFunctionLoss',
                                          np.mean(episode_qf_losses))
                    logger.record_tabular('QFunction/AverageQ',
                                          np.mean(epoch_qs))
                    logger.record_tabular('QFunction/MaxQ', np.max(epoch_qs))
                    logger.record_tabular('QFunction/AverageAbsQ',
                                          np.mean(np.abs(epoch_qs)))
                    logger.record_tabular('QFunction/AverageY',
                                          np.mean(epoch_ys))
                    logger.record_tabular('QFunction/MaxY', np.max(epoch_ys))
                    logger.record_tabular('QFunction/AverageAbsY',
                                          np.mean(np.abs(epoch_ys)))
                    if self.input_include_goal:
                        logger.record_tabular('AverageSuccessRate',
                                              np.mean(self.success_history))
                    last_average_return = np.mean(episode_rewards)

                if not self.smooth_return:
                    episode_rewards = []
                    episode_policy_losses = []
                    episode_qf_losses = []
                    epoch_ys = []
                    epoch_qs = []

                logger.dump_tabular(with_prefix=False)
                if self.plot:
                    self.plotter.update_plot(self.policy, self.max_path_length)
                    if self.pause_for_plot:
                        input("Plotting evaluation run: Press Enter to "
                              "continue...")

        self.shutdown_worker()
        if created_session:
            self.sess.close()
        return last_average_return

    @overrides
    def optimize_policy(self, itr, samples_data):
        """
        Perform algorithm optimizing.

        Returns:
            action_loss: Loss of action predicted by the policy network.
            qval_loss: Loss of q value predicted by the q network.
            ys: y_s.
            qval: Q value predicted by the q network.

        """
        transitions = self.replay_buffer.sample(self.buffer_batch_size)
        observations = transitions["observation"]
        rewards = transitions["reward"]
        actions = transitions["action"]
        next_observations = transitions["next_observation"]
        terminals = transitions["terminal"]

        rewards = rewards.reshape(-1, 1)
        terminals = terminals.reshape(-1, 1)

        if self.input_include_goal:
            goals = transitions["goal"]
            next_inputs = np.concatenate((next_observations, goals), axis=-1)
            inputs = np.concatenate((observations, goals), axis=-1)
        else:
            next_inputs = next_observations
            inputs = observations

        target_actions = self.sess.run(
            self.target_policy_f_prob_online,
            feed_dict={self._target_policy_obs_ph: next_inputs})
        target_qvals = self.sess.run(
            self.target_qf_q_val,
            feed_dict={
                self._target_qf_obs_ph: next_inputs,
                self._target_qf_act_ph: target_actions
            })

        clip_range = (-self.clip_return, 0.
                      if self.clip_pos_returns else self.clip_return)
        ys = np.clip(
            rewards + (1.0 - terminals) * self.discount * target_qvals,
            clip_range[0], clip_range[1])

        _, qval_loss, qval = self.sess.run(
            self.f_train_qf,
            feed_dict={
                self.y: ys,
                self.qf.obs_ph: inputs,
                self.qf.act_ph: actions
            })

        _, action_loss = self.sess.run(
            self.f_train_policy, feed_dict={self.policy.obs_ph: inputs})

        self.sess.run(self.f_update_target, feed_dict=dict())

        return qval_loss, ys, qval, action_loss

    @overrides
    def get_itr_snapshot(self, itr, samples_data):
        return dict(itr=itr, policy=self.policy, env=self.env)


def get_target_ops(variables, target_variables, tau):
    """Get target network update operations."""
    update_ops = []
    init_ops = []
    assert len(variables) == len(target_variables)
    for var, target_var in zip(variables, target_variables):
        init_ops.append(tf.assign(target_var, var))
        update_ops.append(
            tf.assign(target_var, tau * var + (1.0 - tau) * target_var))
    return init_ops, update_ops
