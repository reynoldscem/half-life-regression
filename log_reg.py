from theano import tensor as T
import numpy as np
import theano
import time

class LogisticRegression():
    def __init__(self, input_var=None, n_in=None, n_out=1):
        self.W = theano.shared(
            value=np.zeros(
                (n_in, n_out),
                dtype=theano.config.floatX
            ),
            name='W',
            borrow=True
        )

        self.b = theano.shared(
            value=np.zeros(
                (n_out,),
                dtype=theano.config.floatX
            ),
            name='b',
            borrow=True
        )

        self.pred_prob = T.nnet.sigmoid(T.dot(input_var, self.W) + self.b)
        self.y_pred = T.argmax(self.pred_prob, axis=1)
        self.params = [self.W, self.b]
        self.input_var = input_var

    def binary_cross_entropy(self, target_prob):
        return T.nnet.binary_crossentropy(
            self.pred_prob,
            T.clip(target_prob, 0.001, 0.999)
        ).mean()


def main(args):
    batch_size = 1000000
    learning_rate = 5e-3
    n_epochs = 1000

    with np.load('./dataset.npz') as fd:
        trainX = fd['trainX'].astype(theano.config.floatX)
        trainy = fd['trainy'].squeeze()[:, 0].astype(theano.config.floatX)
        testX = fd['testX'].astype(theano.config.floatX)
        testy = fd['testy'].squeeze()[:, 0].astype(theano.config.floatX)

    X = T.matrix('X', dtype=theano.config.floatX)
    y = T.matrix('y', dtype=theano.config.floatX)

    model = LogisticRegression(input_var=X, n_in=4)

    loss = model.binary_cross_entropy(y)

    g_W = T.grad(cost=loss, wrt=model.W)
    g_b = T.grad(cost=loss, wrt=model.b)

    updates = [
        (model.W, model.W - learning_rate * g_W),
        (model.b, model.b - learning_rate * g_b)
    ]

    train_model = theano.function(
        inputs=[X, y],
        outputs=loss,
        updates=updates
    )

    n_batches = trainX.shape[0] // batch_size

    epoch = 0
    while epoch < n_epochs:
        epoch = epoch + 1
        epoch_loss = 0.
        start_time = time.process_time()
        for index in range(n_batches):
            thisX = trainX[index * batch_size: (index + 1) * batch_size]
            thisy = trainy[
                index * batch_size: (index + 1) * batch_size,
                np.newaxis
            ]

            epoch_loss += train_model(thisX, thisy)
        avg_loss = epoch_loss / index
        time_delta = time.process_time() - start_time
        print('Done epoch {}/{} in {:.2f}, avg loss {:.4f}'.format(
            epoch, n_epochs, time_delta, avg_loss
        ))

if __name__ == '__main__':
    main(None)
