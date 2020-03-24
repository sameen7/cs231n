# coding=utf-8
import numpy as np
from layers import *

################################ 简易版 ##########################################
# class TwoLayerNet:
#     def __init__(self, input_size, hidden_size, output_size, std=1e-4):
#         self.params = {}
#         self.params['W1'] = std * np.random.randn(input_size, hidden_size)
#         self.params['b1'] = np.zeros(hidden_size)
#         self.params['W2'] = std * np.random.randn(hidden_size, output_size)
#         self.params['b2'] = np.zeros(output_size)
#
#     def loss(self, X, y=None, reg=0.0):
#         W1, b1 = self.params['W1'], self.params['b1']
#         W2, b2 = self.params['W2'], self.params['b2']
#         N, D = X.shape
#         scores = None
#
#         h_output = np.maximum(0, X.dot(W1) + b1)  # 第一层输出(N,H)，Relu激活函数
#         scores = h_output.dot(W2) + b2  # 第二层激活函数前的输出(N,C)
#         if y is None:
#             return scores
#         loss = None
#         shift_scores = scores - np.max(scores, axis=1).reshape((-1, 1))
#         softmax_output = np.exp(shift_scores) / np.sum(np.exp(shift_scores), axis=1).reshape(-1, 1)
#         loss = -np.sum(np.log(softmax_output[range(N), list(y)]))
#         loss /= N
#         loss += 0.5 * reg * (np.sum(W1 * W1) + np.sum(W2 * W2))  # 正则项
#         grads = {}
#
#         # 第二层梯度计算
#         dscores = softmax_output.copy()
#         dscores[range(N), list(y)] -= 1
#         dscores /= N
#         grads['W2'] = h_output.T.dot(dscores) + reg * W2
#         grads['b2'] = np.sum(dscores, axis=0)
#
#         # 第一层梯度计算
#         dh = dscores.dot(W2.T)
#         dh_ReLu = (h_output > 0) * dh
#         grads['W1'] = X.T.dot(dh_ReLu) + reg * W1
#         grads['b1'] = np.sum(dh_ReLu, axis=0)
#
#         return loss, grads
#
#     def train(self, X, y, X_val, y_val, learning_rate=1e-3, learning_rate_decay=0.95, reg=1e-5, num_iters=100,
#               batch_size=200, verbose=False):
#         num_train = X.shape[0]
#         iterations_per_epoch = max(num_train / batch_size, 1)  # 每一轮迭代数目
#         loss_history = []
#         train_acc_history = []
#         val_acc_history = []
#         for it in range(num_iters):
#             X_batch = None
#             y_batch = None
#             idx = np.random.choice(num_train, batch_size, replace=True)
#             X_batch = X[idx]
#             y_batch = y[idx]
#             loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
#             loss_history.append(loss)
#
#             # 参数更新
#             self.params['W2'] += -learning_rate * grads['W2']
#             self.params['b2'] += -learning_rate * grads['b2']
#             self.params['W1'] += -learning_rate * grads['W1']
#             self.params['b1'] += -learning_rate * grads['b1']
#
#             if verbose and it % 100 == 0:  # 每迭代100次，打印
#                 print('iteration %d / %d : loss %f' % (it, num_iters, loss))
#
#             if it % iterations_per_epoch == 0:  # 一轮迭代结束
#                 train_acc = (self.predict(X_batch) == y_batch).mean()
#                 val_acc = (self.predict(X_val) == y_val).mean()
#                 train_acc_history.append(train_acc)
#                 val_acc_history.append(val_acc)
#                 # 更新学习率
#                 learning_rate *= learning_rate_decay
#
#         return {
#             'loss_history': loss_history,
#             'train_acc_history': train_acc_history,
#             'val_acc_history': val_acc_history
#         }
#
#     def predict(self, X):
#         y_pred = None
#         h = np.maximum(0, X.dot(self.params['W1']) + self.params['b1'])
#         scores = h.dot(self.params['W2']) + self.params['b2']
#         y_pred = np.argmax(scores, axis=1)
#
#         return y_pred


############################### 全连接 #####################################
class TwoLayerNet(object):

    def __init__(self, input_dim=3 * 32 * 32, hidden_dim=100, num_classes=10,weight_scale=1e-3, reg=0.0):
        self.params = {}
        self.reg = reg
        self.params['W1'] = weight_scale * np.random.randn(input_dim, hidden_dim)
        self.params['b1'] = np.zeros(hidden_dim)
        self.params['W2'] = weight_scale * np.random.randn(hidden_dim, num_classes)
        self.params['b2'] = np.zeros(num_classes)

    def loss(self, X, y=None):
        scores = None
        # a1_out, a1_cache = affine_forward(X, self.params['W1'], self.params['b1'])
        # r1_out, r1_cache = relu_forward(a1_out)
        ar1_out, ar1_cache = affine_relu_forward(X, self.params['W1'], self.params['b1'])
        a2_out, a2_cache = affine_forward(ar1_out, self.params['W2'], self.params['b2'])
        scores = a2_out

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        loss, dscores = softmax_loss(scores, y)
        loss = loss + 0.5 * self.reg * np.sum(self.params['W1'] * self.params['W1']) + 0.5 * self.reg * np.sum(self.params['W2'] * self.params['W2'])
        dx2, dw2, db2 = affine_backward(dscores, a2_cache)
        grads['W2'] = dw2 + self.reg * self.params['W2']
        grads['b2'] = db2
        # dx2_relu = relu_backward(dx2, r1_cache)
        # dx1, dw1, db1 = affine_backward(dx2_relu, a1_cache)
        dx1, dw1, db1 = affine_relu_backward(dx2, ar1_cache)
        grads['W1'] = dw1 + self.reg * self.params['W1']
        grads['b1'] = db1

        return loss, grads


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3 * 32 * 32, num_classes=10,
                 dropout=0, use_batchnorm=False, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
          the network should not use dropout at all.
        - use_batchnorm: Whether or not the network should use batch normalization.
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.use_batchnorm = use_batchnorm
        self.use_dropout = dropout > 0
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}


        layer_input_dim = input_dim
        for i, hd in enumerate(hidden_dims):
            self.params['W%d' % (i + 1)] = weight_scale * np.random.randn(layer_input_dim, hd)
            self.params['b%d' % (i + 1)] = weight_scale * np.zeros(hd)
            if self.use_batchnorm:
                self.params['gamma%d' % (i + 1)] = np.ones(hd)
                self.params['beta%d' % (i + 1)] = np.zeros(hd)
            layer_input_dim = hd
        self.params['W%d' % (self.num_layers)] = weight_scale * np.random.randn(layer_input_dim, num_classes)
        self.params['b%d' % (self.num_layers)] = weight_scale * np.zeros(num_classes)





        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed



        self.bn_params = []
        if self.use_batchnorm:
            self.bn_params = [{'mode': 'train'} for i in xrange(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.iteritems():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.
        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.dropout_param is not None:
            self.dropout_param['mode'] = mode
        if self.use_batchnorm:
            for bn_param in self.bn_params:
                bn_param['mode'] = mode

        scores = None


        layer_input = X
        ar_cache = {}
        dp_cache = {}

        for lay in xrange(self.num_layers - 1):
            if self.use_batchnorm:
                layer_input, ar_cache[lay] = affine_bn_relu_forward(layer_input,
                                                                    self.params['W%d' % (lay + 1)],
                                                                    self.params['b%d' % (lay + 1)],
                                                                    self.params['gamma%d' % (lay + 1)],
                                                                    self.params['beta%d' % (lay + 1)],
                                                                    self.bn_params[lay])
            else:
                layer_input, ar_cache[lay] = affine_relu_forward(layer_input, self.params['W%d' % (lay + 1)],
                                                                 self.params['b%d' % (lay + 1)])

            if self.use_dropout:
                layer_input, dp_cache[lay] = dropout_forward(layer_input, self.dropout_param)

        ar_out, ar_cache[self.num_layers] = affine_forward(layer_input, self.params['W%d' % (self.num_layers)],
                                                           self.params['b%d' % (self.num_layers)])
        scores = ar_out





        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}


        loss, dscores = softmax_loss(scores, y)
        dhout = dscores
        loss = loss + 0.5 * self.reg * np.sum(
            self.params['W%d' % (self.num_layers)] * self.params['W%d' % (self.num_layers)])
        dx, dw, db = affine_backward(dhout, ar_cache[self.num_layers])
        grads['W%d' % (self.num_layers)] = dw + self.reg * self.params['W%d' % (self.num_layers)]
        grads['b%d' % (self.num_layers)] = db
        dhout = dx
        for idx in xrange(self.num_layers - 1):
            lay = self.num_layers - 1 - idx - 1
            loss = loss + 0.5 * self.reg * np.sum(self.params['W%d' % (lay + 1)] * self.params['W%d' % (lay + 1)])
            if self.use_dropout:
                dhout = dropout_backward(dhout, dp_cache[lay])
            if self.use_batchnorm:
                dx, dw, db, dgamma, dbeta = affine_bn_relu_backward(dhout, ar_cache[lay])
            else:
                dx, dw, db = affine_relu_backward(dhout, ar_cache[lay])
            grads['W%d' % (lay + 1)] = dw + self.reg * self.params['W%d' % (lay + 1)]
            grads['b%d' % (lay + 1)] = db
            if self.use_batchnorm:
                grads['gamma%d' % (lay + 1)] = dgamma
                grads['beta%d' % (lay + 1)] = dbeta
            dhout = dx



        return loss, grads


class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:

    conv - relu - 2x2 max pool - affine - relu - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
                 hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
                 dtype=np.float32):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Size of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype


        C, H, W = input_dim
        self.params['W1'] = weight_scale * np.random.randn(num_filters, C, filter_size, filter_size)
        self.params['b1'] = np.zeros(num_filters)
        self.params['W2'] = weight_scale * np.random.randn((H / 2) * (W / 2) * num_filters, hidden_dim)
        self.params['b2'] = np.zeros(hidden_dim)
        self.params['W3'] = weight_scale * np.random.randn(hidden_dim, num_classes)
        self.params['b3'] = np.zeros(num_classes)


        for k, v in self.params.iteritems():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']

        # pass conv_param to the forward pass for the convolutional layer
        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None


        conv_forward_out_1, cache_forward_1 = conv_relu_pool_forward(X, self.params['W1'], self.params['b1'],conv_param, pool_param)
        affine_forward_out_2, cache_forward_2 = affine_forward(conv_forward_out_1, self.params['W2'], self.params['b2'])
        affine_relu_2, cache_relu_2 = relu_forward(affine_forward_out_2)
        scores, cache_forward_3 = affine_forward(affine_relu_2, self.params['W3'], self.params['b3'])



        if y is None:
            return scores

        loss, grads = 0, {}


        loss, dout = softmax_loss(scores, y)

        # Add regularization
        loss += self.reg * 0.5 * (
                    np.sum(self.params['W1'] ** 2) + np.sum(self.params['W2'] ** 2) + np.sum(self.params['W3'] ** 2))

        dX3, grads['W3'], grads['b3'] = affine_backward(dout, cache_forward_3)
        dX2 = relu_backward(dX3, cache_relu_2)
        dX2, grads['W2'], grads['b2'] = affine_backward(dX2, cache_forward_2)
        dX1, grads['W1'], grads['b1'] = conv_relu_pool_backward(dX2, cache_forward_1)

        grads['W3'] = grads['W3'] + self.reg * self.params['W3']
        grads['W2'] = grads['W2'] + self.reg * self.params['W2']
        grads['W1'] = grads['W1'] + self.reg * self.params['W1']



        return loss, grads
