import tensorflow as tf
import numpy as np
from typing import Mapping

class Continuous:
    def __init__(self, action_dims: int, **kwargs):
        if 'init_log_var' in kwargs:
            self.init_log_var = kwargs['init_log_var']
        else:
            self.init_log_var = -1

        self.__act_dim = action_dims

    def init_vars(self):
        self.log_vars = tf.get_variable('logvars', (self.__act_dim,), tf.float32,
                                        tf.constant_initializer(0.0)) + self.init_log_var
        self.old_log_vars_ph = tf.placeholder(tf.float32, (self.__act_dim,), 'old_log_vars_ph')

    def output_layer(self, input_layer: tf.Tensor, num_units_input: int, out_name: str = 'means') -> tf.Tensor:
        return tf.layers.dense(
            input_layer,
            self.__act_dim,
            None,
            kernel_initializer=tf.random_normal_initializer(stddev=np.sqrt(1 / num_units_input)),
            name=out_name
        )

    def log_probs(self, *args, **kwargs) -> tf.Tensor:
        is_old = kwargs['is_old']
        act_ph = args[0]
        means = args[1]

        if is_old:
            log_vars = self.old_log_vars_ph
        else:
            log_vars = self.log_vars

        logp = -0.5 * tf.reduce_sum(log_vars)
        logp += -0.5 * tf.reduce_sum(tf.square(act_ph - means) / tf.exp(log_vars), axis=1)

        return logp

    def kl(self, *args):
        means = args[0]
        old_means_ph = args[1]

        log_det_cov_old = tf.reduce_sum(self.old_log_vars_ph)
        log_det_cov_new = tf.reduce_sum(self.log_vars)
        tr_old_new = tf.reduce_sum(tf.exp(self.old_log_vars_ph - self.log_vars))

        return 0.5 * tf.reduce_mean(log_det_cov_new - log_det_cov_old + tr_old_new +
                                       tf.reduce_sum(tf.square(means - old_means_ph) /
                                                     tf.exp(self.log_vars), axis=1) -
                                       self.__act_dim)

    def entropy(self, **kwargs):
        return 0.5 * (self.__act_dim * (np.log(2 * np.pi) + 1) + tf.reduce_sum(self.log_vars))

    def sample(self, means: tf.Tensor):
        return (means +
                tf.exp(self.log_vars / 2.0) *
                tf.random_normal(shape=(self.__act_dim,), dtype=tf.float32))

    def run_and_get_old_mean(self,
                             feed_dict: Mapping[tf.Variable, tf.Tensor],
                             means: tf.Tensor,
                             old_means_ph: tf.Variable):
        old_means_np, old_log_vars_np = tf.get_default_session().run([means,
                                                                      self.log_vars],
                                                                     feed_dict)


        feed_dict[self.old_log_vars_ph] = old_log_vars_np
        feed_dict[old_means_ph] = old_means_np

        return feed_dict

class Discret:
    def __init__(self, action_dims: int, **kwargs):
        self.__act_dim = action_dims
        self._class_to_action = kwargs['class_to_action']

    def init_vars(self):
        pass

    def output_layer(self, input_layer: tf.Tensor, num_units_input: int, out_name: str = 'means') -> tf.Tensor:
        return tf.layers.dense(
            input_layer,
            self.__act_dim,
            tf.nn.softmax,
            kernel_initializer=tf.random_normal_initializer(stddev=np.sqrt(1 / num_units_input)),
            name=out_name
        )

    def log_probs(self, *args, **kwargs) -> tf.Tensor:
        act_ph = args[0]
        means = args[1]

        return tf.reduce_sum( act_ph*tf.log(means), axis=1)

    def kl(self, *args):
        means = args[0]
        old_means_ph = args[1]

        return tf.reduce_mean(-tf.reduce_sum(old_means_ph * tf.log(means/old_means_ph), axis=1))

    def entropy(self, **kwargs):
        assert 'means' in kwargs, 'means must be an argument of entropy in the discret case'

        means = kwargs['means']

        return -tf.reduce_sum(means * tf.log(means))

    def sample(self, means: tf.Tensor):
        dist = tf.distributions.Categorical(probs=means)
        samples = tf.to_int32(dist.sample())
        samples = tf.reshape(samples, shape=[])
        return self._class_to_action(samples)

    def run_and_get_old_mean(self,
                             feed_dict: Mapping[tf.Variable, tf.Tensor],
                             means: tf.Tensor,
                             old_means_ph: tf.Variable):
        old_means_np = tf.get_default_session().run(means, feed_dict)

        feed_dict[old_means_ph] = old_means_np

        return feed_dict
