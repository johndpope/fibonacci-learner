# fibonacci-learner

Inspired by the [Joel Gru's FizzBuzz blog post](http://joelgrus.com/2016/05/23/fizz-buzz-in-tensorflow/), let's try to get a model to learn the Fibonacci sequence!

## How to Use

Make sure you have Keras 1.0.8 installed with Theano backend (and all Keras+Theano dependencies).

Then simply

    $ python fibonacci_lstm.py

and enter how many of the first Fibonacci terms you want, and the program prints them out.

Ex:
    ~~~ Generating Fibonacci Terms ~~~

    First `how many` terms? 
    10

    1
    1
    2
    3
    5
    8
    13
    21
    34
    55

This script:

    1) Loads sample Fibonacci series from fibonacci_sequence.csv
    2) Synthesizes training and test data sets using the terms from the sample file and some other data.
    3) Defines the model
    4) Trains the model
    5) Predicts the Fibonacci sequence using the trained model

I've included a pretrained model, you can modify the program to use Keras' 'load_model' function to use it.

## Caveats

Technically I'm teaching a recurrent neural network how to add two numbers in binary form. I call this iteratively, generating the next term in the Fibonacci sequence then feed that in for the subsequent term.

Maximum integer size limits the number of Fibonacci numbers the model can generate to 92 - which is the same as any run-of-the-mill Fibonacci generating program running on a standard laptop or desktop.

I will be posting a blog post soon on this little project and the things I learned about deep learning representations.
