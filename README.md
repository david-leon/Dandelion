# Dandelion
[![License: MPL 2.0](https://img.shields.io/badge/License-MPL%202.0-brightgreen.svg)](https://github.com/david-leon/Dandelion/blob/master/LICENSE)
[![Python 3.x](https://img.shields.io/badge/python-3.x-brightgreen.svg)](https://www.python.org/downloads/release)
[![PyPI version](https://badge.fury.io/py/Dandelion.svg)](https://badge.fury.io/py/Dandelion)
[![Travis CI](https://travis-ci.org/david-leon/Dandelion.svg?branch=master)](https://travis-ci.org/david-leon/Dandelion)

A quite light weight deep learning framework, on top of Theano, offering better balance between flexibility and abstraction

## Targeted Users
Researchers who need flexibility as well as convenience to experiment all kinds of *nonstandard* network structures, and also the stability of Theano.

## Featuring
* **Aiming to offer better balance between flexibility and abstraction.**
    * Easy to use and extend, support for any neural network structure.  
    * Loose coupling, each part of the framework can be modified independently.
* **More like a handy library of deep learning modules.**
    * Common modules such as CNN, LSTM, GRU, Dense, Dropout, Batch Normalization, and common optimization methods such as SGD, Adam, Adadelta, Rmsprop are ready out-of-the-box.
* **Plug & play, operating directly on Theano tensors, no upper abstraction applied.**
    * Unlike previous frameworks like Keras, Lasagne, etc., Dandelion operates directly on tensors instead of layer abstractions, making it quite easy to plug in 3rd part defined deep learning modules (layer defined by Keras/Lasagne) or vice versa.

## Documentation
Documentation is available online: [https://david-leon.github.io/Dandelion/](https://david-leon.github.io/Dandelion/)

## Install
Use pip channel
```
pip install dandelion --upgrade
```
Dependency
* Theano >=1.0
* Scipy (required by `dandelion.ext.CV`)
* Pillow (required by `dandelion.ext.CV`)
* OpenCV (required by `dandelion.ext.CV`)

## Quick Tour
```python
    import theano
    import theano.tensor as tensor
    from dandelion.module import *
    from dandelion.update import *
    from dandelion.functional import *
    from dandelion.util import gpickle

    class model(Module):
        def __init__(self, batchsize=None, input_length=None, Nclass=6, noise=(0.5, 0.2, 0.7, 0.7, 0.7)):
            super().__init__()
            self.batchsize = batchsize
            self.input_length = input_length
            self.Nclass = Nclass
            self.noise = noise

            self.dropout0 = Dropout()
            self.dropout1 = Dropout()
            self.dropout2 = Dropout()
            self.dropout3 = Dropout()
            self.dropout4 = Dropout() 
            W = gpickle.load('word_embedding(6336, 256).gpkl')
            self.embedding = Embedding(num_embeddings=6336, embedding_dim=256, W=W)
            self.lstm0 = LSTM(input_dims=256, hidden_dim=100)
            self.lstm1 = LSTM(input_dims=256, hidden_dim=100)
            self.lstm2 = LSTM(input_dims=200, hidden_dim=100)
            self.lstm3 = LSTM(input_dims=200, hidden_dim=100)
            self.lstm4 = LSTM(input_dims=200, hidden_dim=100)
            self.lstm5 = LSTM(input_dims=200, hidden_dim=100)
            self.dense = Dense(input_dims=200, output_dim=Nclass)
       
        def forward(self, x):
            self.work_mode = 'train'
            x = self.dropout0.forward(x, p=self.noise[0], rescale=False)
            x = self.embedding.forward(x)         # (B, T, D)

            x = self.dropout1.forward(x, p=self.noise[1], rescale=True)
            x = x.dimshuffle((1, 0, 2))           # (B, T, D) -> (T, B, D)
            x_f = self.lstm0.forward(x, None, None, None)
            x_b = self.lstm1.forward(x, None, None, None, backward=True)
            x = tensor.concatenate([x_f, x_b], axis=2)

            x = pool_1d(x, ws=2, ignore_border=True, mode='average_exc_pad', axis=0)

            x = self.dropout2.forward(x, p=self.noise[2], rescale=True)
            x_f = self.lstm2.forward(x, None, None, None)
            x_b = self.lstm3.forward(x, None, None, None, backward=True)
            x = tensor.concatenate([x_f, x_b], axis=2)

            x = self.dropout3.forward(x, p=self.noise[3], rescale=True)
            x_f = self.lstm4.forward(x, None, None, None, only_return_final=True)
            x_b = self.lstm5.forward(x, None, None, None, only_return_final=True, backward=True)
            x = tensor.concatenate([x_f, x_b], axis=1)

            x = self.dropout4.forward(x, p=self.noise[4], rescale=True)
            y = sigmoid(self.dense.forward(x))
            return y

        def predict(self, x):
            self.work_mode = 'inference'
            x = self.embedding.predict(x)

            x = x.dimshuffle((1, 0, 2))  # (B, T, D) -> (T, B, D)
            x_f = self.lstm0.predict(x, None, None, None)
            x_b = self.lstm1.predict(x, None, None, None, backward=True)
            x = tensor.concatenate([x_f, x_b], axis=2)

            x = pool_1d(x, ws=2, ignore_border=True, mode='average_exc_pad', axis=0)

            x_f = self.lstm2.predict(x, None, None, None)
            x_b = self.lstm3.predict(x, None, None, None, backward=True)
            x = tensor.concatenate([x_f, x_b], axis=2)

            x_f = self.lstm4.predict(x, None, None, None, only_return_final=True)
            x_b = self.lstm5.predict(x, None, None, None, only_return_final=True, backward=True)
            x = tensor.concatenate([x_f, x_b], axis=1)

            y = sigmoid(self.dense.predict(x))
            return y            
```

## Why Another DL Framework
* The reason is more about the lack of flexibility for existing DL frameworks, such as Keras, Lasagne, Blocks, etc.
* By **“flexibility”**, we means whether it is easy to modify or extend the framework. 
    * The famous DL framework Keras is designed to be beginner-friendly oriented, at the cost of being quite hard to modify.
    * Compared to Keras, another less-famous framework Lasagne provides more flexibility. It’s easier to write your own layer by Lasagne for small neural network, however, for complex neural networks it still needs quite manual works because like other existing frameworks, Lasagne operates on abstracted ‘Layer’ class instead of raw tensor variables.

## Project Layout
Python Module     | Explanation
----------------- | ----------------
module            | all neual network module definitions
functional        | operations on tensor with no parameter to be learned
initialization    | initialization methods for neural network modules
activation        | definition of all activation functions
objective         | definition of all loss objectives
update            | definition of all optimizers
util              | utility functions
model             | model implementations out-of-the-box
ext               | extensions

## Credits
The design of Dandelion heavily draws on [Lasagne](https://github.com/Lasagne/Lasagne) and [Pytorch](http://pytorch.org/), both my favorate DL libraries.  
Special thanks to **Radomir Dopieralski**, who transferred the `dandelion` project name on pypi to us. Now you can install the package by simply `pip install dandelion`.
