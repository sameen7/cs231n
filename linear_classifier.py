import numpy as np
from svm import *
from softmax import *
class LinearClassifier:
    def __init__(self):
        self.W=None
    def train(self, X, y, learning_rate=1e-3, reg=1e-5,num_iters=100,batch_size=200, verbose=False):
        num_train,dim=X.shape
        num_classes=np.max(y) +1
        if self.W is None:
            self.W=0.001*np.random.randn(dim, num_classes)
        loss_history= []
        for it in range(num_iters):
            X_batch=None
            y_batch=None
            batch_idx=np.random.choice(num_train,batch_size,replace=True)
            X_batch=X[batch_idx]
            y_batch=y[batch_idx]
            loss, grad=self.loss(X_batch, y_batch, reg)
            loss_history.append(loss)
            self.W+=-1*learning_rate*grad
            if verbose and it%100==0:
                print('iteration %d / %d: loss %f'% (it, num_iters,loss))
        return loss_history

    def predict(self, X):
        y_pred=np.zeros(X.shape[1])
        scores=X.dot(self.W)
        y_pred=np.argmax(scores, axis=1)
        return y_pred

    def loss(self, X_batch, y_batch, reg):
        pass

class LinearSVM(LinearClassifier):
    def loss(self, X_batch, y_batch, reg):
        return svm_loss_vectorized(self.W, X_batch, y_batch, reg)

class Softmax(LinearClassifier):
    def loss(self, X_batch, y_batch, reg):
        return softmax_loss_vectorized(self.W, X_batch, y_batch, reg)