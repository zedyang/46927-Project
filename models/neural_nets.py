import numpy as np
from sklearn.exceptions import NotFittedError
import tensorflow as tf
from influence.emp_risk_optimizer import EmpiricalRiskOptimizer


class ConvNet(EmpiricalRiskOptimizer):

    def __init__(self, **kwargs):
        super(ConvNet, self).__init__(**kwargs)

        # input shape
        self.input_side = kwargs.pop('input_side')
        self.n_channels = kwargs.pop('n_channels')
        self.input_flat_size = self.input_side * self.input_side
        self.input_shape = (self.input_side, self.input_side)

        # network architecture
        self.filter_size1 = kwargs.pop('filter_size1')
        self.n_filters1 = kwargs.pop('n_filters1')
        self.filter_size2 = kwargs.pop('filter_size2')
        self.n_filters2 = kwargs.pop('n_filters2')
        self.fc_size = kwargs.pop('fc_size')
        self.down_sample = kwargs.pop('down_sample')

        self.hyperparams += [
            'filter_size1', 'n_filters1',
            'filter_size2', 'n_filters2',
            'fc_size'
        ]

        self.y_pred = None
        self.y_prob_pred = None

    def get_all_params(self):
        """

        :return:
        """
        # all parameter dimensions
        p_W1 = self.filter_size1 * self.filter_size1 * \
            self.n_channels * self.n_filters1
        p_b1 = self.n_filters1
        p_W2 = self.filter_size2 * self.filter_size2 * \
            self.n_filters1 * self.n_filters2
        p_b2 = self.n_filters2

        assert self.input_side % (self.down_sample**2) == 0
        fc1_input_n_features = self.n_filters2*(self.input_side//(
            self.down_sample**2))**2

        p_Wfc1 = fc1_input_n_features * self.fc_size
        p_bfc1 = self.fc_size
        p_Wfc2 = self.fc_size * self.n_classes
        p_bfc2 = self.n_classes

        all_params_flat_dims = [0, p_W1, p_b1, p_W2, p_b2,
                                p_Wfc1, p_bfc1, p_Wfc2, p_bfc2]
        pos = np.cumsum(all_params_flat_dims)
        self.n_params = np.sum(all_params_flat_dims)

        all_params = tf.get_variable(
            name='flat_params',
            shape=(self.n_params, 1),
            initializer=tf.truncated_normal_initializer)

        # make parameter blocks
        W_conv1 = tf.reshape(
            all_params[pos[0]:pos[1], :],
            [self.filter_size1, self.filter_size1,
             self.n_channels, self.n_filters1],
            name='W_conv1')

        b_conv1 = tf.reshape(
            all_params[pos[1]:pos[2], :],
            [self.n_filters1], name='b_conv1')

        W_conv2 = tf.reshape(
            all_params[pos[2]:pos[3], :],
            [self.filter_size2, self.filter_size2,
             self.n_filters1, self.n_filters2],
            name='W_conv2')

        b_conv2 = tf.reshape(
            all_params[pos[3]:pos[4], :],
            [self.n_filters2], name='b_conv2')

        W_fc1 = tf.reshape(
            all_params[pos[4]:pos[5], :],
            [fc1_input_n_features, self.fc_size],
            name='W_fc1')

        b_fc1 = tf.reshape(
            all_params[pos[5]:pos[6], :],
            [self.fc_size], name='b_fc1')

        W_fc2 = tf.reshape(
            all_params[pos[6]:pos[7], :],
            [self.fc_size, self.n_classes],
            name='W_fc2')

        b_fc2 = tf.reshape(
            all_params[pos[7]:pos[8], :],
            [self.n_classes], name='b_fc2')

        self.all_params_dict['W_conv1'] = W_conv1
        self.all_params_dict['b_conv1'] = b_conv1
        self.all_params_dict['W_conv2'] = W_conv2
        self.all_params_dict['b_conv2'] = b_conv2
        self.all_params_dict['W_fc1'] = W_fc1
        self.all_params_dict['b_fc1'] = b_fc1
        self.all_params_dict['W_fc2'] = W_fc2
        self.all_params_dict['b_fc2'] = b_fc2

        return all_params

    def get_emp_risk(self):
        # convolution layer 1
        conv_layer_1 = tf.nn.conv2d(
            input=tf.reshape(
                self.X_input,
                [-1, self.input_side, self.input_side, self.n_channels]),
            filter=self.all_params_dict['W_conv1'],
            strides=[1, 1, 1, 1],
            padding='SAME', name='conv_layer_1'
        )
        conv_layer_1 += self.all_params_dict['b_conv1']
        # pooling layer 1
        pooling_layer_1 = tf.nn.max_pool(
            value=conv_layer_1,
            ksize=[1, self.down_sample, self.down_sample, 1],
            strides=[1, self.down_sample, self.down_sample, 1],
            padding='SAME', name='pooling_1'
        )
        activation_1 = tf.nn.relu(pooling_layer_1, name='activation_1')

        # convolution layer 2
        conv_layer_2 = tf.nn.conv2d(
            input=activation_1,
            filter=self.all_params_dict['W_conv2'],
            strides=[1, 1, 1, 1],
            padding='SAME', name='conv_layer_2'
        )
        conv_layer_2 += self.all_params_dict['b_conv2']
        # pooling layer 2
        pooling_layer_2 = tf.nn.max_pool(
            value=conv_layer_2,
            ksize=[1, self.down_sample, self.down_sample, 1],
            strides=[1, self.down_sample, self.down_sample, 1],
            padding='SAME', name='pooling_2'
        )
        activation_2 = tf.nn.relu(pooling_layer_2, name='activation_2')

        # fully connected layer 1
        layer_2_shape = activation_2.get_shape()
        layer_2_flat_shape = layer_2_shape[1:4].num_elements()
        flat_layer = tf.reshape(
            activation_2, [-1, layer_2_flat_shape], name='flatten_layer')
        fc_layer_1 = tf.matmul(
            flat_layer,
            self.all_params_dict['W_fc1']) + self.all_params_dict['b_fc1']
        activation_fc = tf.nn.relu(fc_layer_1, name='activation_fc')

        # fully connected layer 2
        fc_layer_2 = tf.matmul(
            activation_fc,
            self.all_params_dict['W_fc2']) + self.all_params_dict['b_fc2']
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=fc_layer_2, labels=self.y_input)
        emp_risk = tf.reduce_mean(cross_entropy, name='emp_risk')

        self.y_prob_pred = tf.nn.softmax(fc_layer_2, name='y_prob_pred')
        self.y_pred = tf.argmax(self.y_prob_pred, axis=1, name='y_pred')

        return cross_entropy, emp_risk

    def fit_with_sklearn(self, **kwargs):
        raise NotFittedError

    def refit_with_sklearn(self, feed_dict, **kwargs):
        raise NotFittedError

    def predict(self, X_test):
        y_pred_val = self.sess.run(
            self.y_pred,
            feed_dict={self.X_input: X_test}
        )
        return y_pred_val
