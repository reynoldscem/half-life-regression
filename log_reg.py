from lasagne.updates import nesterov_momentum
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

    def binary_cross_entropy(self, target_prob, lbd=1e-1):
        return T.nnet.binary_crossentropy(
            self.pred_prob,
            T.clip(target_prob, 0.001, 0.999)
        ).mean() + lbd * T.sum(self.W ** 2)

    def mean_abs_err(self, target_prob, lbd=0.):
        return T.mean(abs(
            self.pred_prob - target_prob
        )) + lbd * T.sum(self.W ** 2)

    def mean_squared_err(self, target_prob, lbd=0.):
        return T.mean((
            self.pred_prob - target_prob
        ) ** 2) + lbd * T.sum(self.W ** 2)

    def get_param_values(self):
        return [param.get_value() for param in self.params]


def load_data():
    with np.load('./dataset.npz') as fd:
        trainX = fd['trainX'].astype(theano.config.floatX)
        trainy = fd['trainy'][:, :, 0].astype(theano.config.floatX)
        testX = fd['testX'].astype(theano.config.floatX)
        testy = fd['testy'][:, :, 0].astype(theano.config.floatX)

    return trainX, trainy, testX, testy


def get_model_and_funcs(learning_rate, n_features):
    X = T.matrix('X', dtype=theano.config.floatX)
    y = T.matrix('y', dtype=theano.config.floatX)

    model = LogisticRegression(
        input_var=X,
        n_in=n_features
    )

    # loss = model.binary_cross_entropy(y)
    loss = model.mean_abs_err(y, 5e-2)
    #loss = model.mean_squared_err(y, 5e-2)
    grads = T.grad(loss, model.params)
    updates = nesterov_momentum(grads, model.params, learning_rate)

    train_model = theano.function(
        inputs=[X, y],
        outputs=loss,
        updates=updates
    )

    test_model = theano.function(
        [X, y],
        model.mean_abs_err(y)
    )

    model_loss = theano.function(
        [X, y],
        model.binary_cross_entropy(y, 0.)
    )

    return train_model, test_model, model_loss, model


def train_val_split(trainX, trainy, split_degree=0.9, shuffle=True):
    combined_matrix = np.hstack((trainX, trainy))

    if shuffle:
        np.random.shuffle(combined_matrix)

    n_rows = combined_matrix.shape[0]
    split_ind = int(split_degree * n_rows)

    train_split = combined_matrix[:split_ind, :]
    val_split = combined_matrix[split_ind:, :]

    return (
        train_split[:, :-1], train_split[:, -1:],
        val_split[:, :-1], val_split[:, -1:]
    )


def main(args):
    learning_rate = theano.shared(np.array(1e-2, dtype=theano.config.floatX))
    decay_frequency = 25
    anneal_factor = 0.9
    min_decay = 25
    n_epochs = 1000

    def decay_function(epoch):
        if epoch >= min_decay and epoch % decay_frequency == 0:
            new_lr = (
                learning_rate.get_value() * anneal_factor
            ).astype(theano.config.floatX)
            print('Learning rate annealing to {:.5f}'.format(new_lr))
            learning_rate.set_value(new_lr)

    trainX, trainy, testX, testy = load_data()
    trainX, trainy, valX, valy = train_val_split(trainX, trainy)

    train_function, test_function, model_loss, model = get_model_and_funcs(
        learning_rate, trainX.shape[-1]
    )

    output_format_string = (
        'Done epoch {:03d}/{} in {:.2f}, '
        'train loss {:.8f}, val loss {:.8f}, val err {:.4f}'
    )

    best_val_error = np.inf
    epochs_since_best = 0

    for epoch in range(1, n_epochs + 1):
        start_time = time.process_time()

        decay_function(epoch)

        epoch_loss = float(train_function(trainX, trainy))
        epochs_since_best += 1

        val_loss = float(model_loss(valX, valy))
        val_error = float(test_function(valX, valy))

        if val_error <= best_val_error:
            best_val_error = val_error
            epochs_since_best = 0
            best_params = model.get_param_values()

        if epochs_since_best >= 25:
            print('Early stopping')
            break

        time_delta = time.process_time() - start_time
        print(output_format_string.format(
            epoch, n_epochs, time_delta, epoch_loss, val_loss, val_error
        ))

    test_error = float(test_function(testX, testy))
    print('Finished, test MAE: {}'.format(test_error))
    np.savez('parameters_{}.npz'.format(int(time.time()), *best_params))


if __name__ == '__main__':
    main(None)
