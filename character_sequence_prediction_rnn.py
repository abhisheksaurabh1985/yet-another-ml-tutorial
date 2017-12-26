import numpy as np

data = open('./datasets/kafka.txt', 'r').read()
print type(data)
print len(data)

# Get unique characters in a list
unique_chars = list(set(data))
# Get data size and vocabulary size
data_size = len(data)
vocab_size = len(unique_chars)

print "Data has a total of %d characters. %d of those are unique." % (data_size, vocab_size)

# Encode/ Decode char/ vector
# Get each unique character and its index in the list
char_to_ix = {ch: i for i, ch in enumerate(unique_chars)}
ix_to_char = {i: ch for i, ch in enumerate(unique_chars)}
print char_to_ix
print ix_to_char

# Create vector from a character
vector_for_char_a = np.zeros((vocab_size, 1))
vector_for_char_a[char_to_ix['a']] = 1
print vector_for_char_a.ravel()

# Hyper-parameters
hidden_size = 100
sequence_length = 25  # Output size. Probably refers to the time steps.
learning_rate = 0.1

# Model parameters
Wxh = np.random.randn(hidden_size, vocab_size) * 0.01
Whh = np.random.randn(hidden_size, hidden_size) * 0.01
Why = np.random.randn(vocab_size, hidden_size) * 0.01
bh = np.zeros((hidden_size, 1))
by = np.zeros((vocab_size, 1))


# Loss function
def loss_function(inputs, targets, hprev):
    # Empty dicts to store values at every time step.
    # So for e.g., xs will store the encoded values of the input at each of the 25 time steps.
    xs, hs, ys, ps, = {}, {}, {}, {}
    hs[-1] = np.copy(hprev)

    # Initialize loss as zero
    loss = 0

    for t in xrange(len(inputs)):
        xs[t] = np.zeros((vocab_size, 1))
        xs[t][inputs[t]] = 1
        hs[t] = np.tanh(np.dot(Wxh, xs[t]) + np.dot(Whh, hs[t - 1]) + bh)
        ys[t] = np.dot(Why, hs[t]) + by
        ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t]))
        loss += -np.log(ps[t][targets[t], 0])

    dWxh, dWhh, dWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
    dbh, dby = np.zeros_like(bh), np.zeros_like(by)
    dhnext = np.zeros_like(hs[0])  # Hidden state for the next time step

    for t in reversed(xrange(len(inputs))):
        dy = np.copy(ps[t])
        dy[targets[t]] -= 1  # backprop into y
        dWhy += np.dot(dy, hs[t].T)
        dby += dy
        dh = np.dot(Why.T, dy) + dhnext  # backprop into h
        dhraw = (1 - hs[t] * hs[t]) * dh  # backprop through tanh nonlinearity
        dbh += dhraw  # derivative of hidden bias
        dWxh += np.dot(dhraw, xs[t].T)  # derivative of input to hidden layer weight
        dWhh += np.dot(dhraw, hs[t - 1].T)  # derivative of hidden layer to hidden layer weight
        dhnext = np.dot(Whh.T, dhraw)

    for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
        np.clip(dparam, -5, 5, out=dparam)  # clip to mitigate exploding gradients

    return loss, dWxh, dWhh, dWhy, dbh, dby, hs[len(inputs) - 1]


# Prediction, one full forward pass
def sample(h, seed_ix, n):
    """
    :param h: memory state of size (100,1)
    :param seed_ix: Seed letter for the first time step as integer. For eg. char_to_ix['a'].
    :param n: Number of characters to be predicted
    :return:
    """
    # create vector
    x = np.zeros((vocab_size, 1))
    # customize it for our seed char
    x[seed_ix] = 1
    # list to store generated chars
    ixes = []
    # for as many characters as we want to generate
    for t in xrange(n):
        # a hidden state at a given time step is a function
        # of the input at the same time step modified by a weight matrix
        # added to the hidden state of the previous time step
        # multiplied by its own hidden state to hidden state matrix.
        # x.shape:(81,1); Wxh.shape:(100,81); h.shape:(100,1); Whh.shape:(100,100); bh.shape:(100,1)
        h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h) + bh)
        # compute output (unnormalised). Why.shape:(81,100); by.shape:(81,1)
        y = np.dot(Why, h) + by
        # probabilities for next chars
        p = np.exp(y) / np.sum(np.exp(y))
        print "-------------p---------------------", p
        # index of the character with highest probability. pick one with the highest probability.
        ix = np.random.choice(range(vocab_size), p=p.ravel())
        print "=============ix=====================", ix
        # create a vector
        x = np.zeros((vocab_size, 1))
        # customize it for the predicted char
        x[ix] = 1
        # add it to the list
        ixes.append(ix)
        print "##############ixes##################", ixes
    txt = ''.join(ix_to_char[ix] for ix in ixes)
    print '----\n %s \n----' % (txt,)


hprev = np.zeros((hidden_size, 1))  # reset RNN memory
# predict the 200 next characters given 'a'
sample(hprev, char_to_ix['a'], 4)


p=2
inputs = [char_to_ix[ch] for ch in data[p:p+sequence_length]]
print "inputs", inputs
targets = [char_to_ix[ch] for ch in data[p+1:p+sequence_length+1]]
print "targets", targets


n, p = 0, 0
mWxh, mWhh, mWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
mbh, mby = np.zeros_like(bh), np.zeros_like(by)  # memory variables for Adagrad
smooth_loss = -np.log(1.0/vocab_size)*sequence_length  # loss at iteration 0
while n <= 3:
    # prepare inputs (we're sweeping from left to right in steps sequence_length long)
    # check "How to feed the loss function to see how this part works
    if p+sequence_length+1 >= len(data) or n == 0:
        hprev = np.zeros((hidden_size, 1))  # reset RNN memory
        p = 0  # go from start of data
    inputs = [char_to_ix[ch] for ch in data[p:p+sequence_length]]
    targets = [char_to_ix[ch] for ch in data[p+1:p+sequence_length+1]]

    # forward seq_length characters through the net and fetch gradient
    loss, dWxh, dWhh, dWhy, dbh, dby, hprev = loss_function(inputs, targets, hprev)
    # Smooth_loss doesn't play any role in the training. It is just a low pass filtered version of the loss.
    # It is a way to average the loss on over the last iterations to better track the progress.
    smooth_loss = smooth_loss * 0.999 + loss * 0.001

    # sample from the model now and then
    # if n % 1000 == 0:
    print 'iter %d, loss: %f' % (n, smooth_loss)  # print progress
    sample(hprev, inputs[0], 200)

    # perform parameter update with Adagrad. This is a type of gradient descent strategy.
    for param, dparam, mem in zip([Wxh, Whh, Why, bh, by],
                                [dWxh, dWhh, dWhy, dbh, dby],
                                [mWxh, mWhh, mWhy, mbh, mby]):
        mem += dparam * dparam
        param += -learning_rate * dparam / np.sqrt(mem + 1e-8)  # adagrad update

    p += sequence_length  # move data pointer
    n += 1  # iteration counter
