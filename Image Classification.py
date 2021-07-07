
# importing tensorflow library

import tensorflow as tf

print('Using TensorFlow version', tf.__version__)
tf.logging.set_verbosity(tf.logging.ERROR)


# importing the MNIST data set that has lots of images of handwritten digits and also has test cases for evaluation of model


from tensorflow.keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print('x_train shape: ', x_train.shape)
print('y_train shape: ', y_train.shape)
print('x_test shape: ', x_test.shape)
print('y_test shape: ', y_test.shape)


# show the images fetched as the training set

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.imshow(x_train[0], cmap = 'binary')
plt.show()
y_train[0]
print(set(y_train))


# One Hot Encoding

from tensorflow.python.keras.utils import to_categorical

y_train_encoded = to_categorical(y_train)
y_test_encoded = to_categorical(y_test)


# To make sure the encoding worked, let's check the shape of the encoded labels.

print('y_train shape: ', y_train_encoded.shape)
print('y_test shape: ', y_test_encoded.shape)


# And just like before, let's also take a look at the first label and make sure that encoding is correct:

y_train_encoded[0]


# create a Neural Network which will take 784 dimensional vectors as inputs (28 rows * 28 columns) and will output a 10 dimensional vector (For the 10 classes). We have already converted the outputs to 10 dimensional, one-hot encoded vectors. Now, let's convert the input to the required format as well. We will use numpy to easily unroll the examples from `(28, 28)` arrays to `(784, 1)` vectors.


import numpy as np

x_train_reshaped = np.reshape(x_train, (60000, 784))
x_test_reshaped = np.reshape(x_test, (10000, 784))

print('x_train_reshaped shape: ', x_train_reshaped.shape)
print('x_test_reshaped shape: ', x_test_reshaped.shape)


# Each element in each example is a pixel value. Let's take a look at a few values of just one example.

# In[ ]:


print(set(x_train_reshaped[0]))


# Pixel values, in this dataset, range from 0 to 255. While that's fine if we want to display our images, for our neural network to learn the weights and biases for different layers, computations will be simply much more effective and fast if we *normalized* these values. In order to normalize the data, we can calculate the mean and standard deviation for each example.


x_mean = np.mean(x_train_reshaped)
x_std = np.std(x_train_reshaped)

print('mean: ', x_mean)
print('std: ', x_std)


# Now we will normalise both the training and test set using the mean and standard deviation we just calculated. Notice that we will need to apply the same mean and standard deviation to the test set even though we did not use the test set to calculate these values.

# In[ ]:


epsilon = 1e-10
x_train_norm = (x_train_reshaped - x_mean)/(x_std + epsilon)
x_test_norm = (x_test_reshaped - x_mean)/(x_std + epsilon)


# Note how we added a small value to our denominator. This is because, just in case if our std was close to zero, we'd get very large values as a result. In this case, that's obviously not true but we added this anyway as a good practice since this is typically done to ensure numerical stability.

# We looked at some of the values for the first training example before. Let's take a look at it again, after having normalised the values.


print(set(x_train_norm[0]))


# Creating a Model-Using Sequential class defined in Keras to create our model. All the layers are going to be Dense layers. This means, like our examples above, all the nodes of a layer would be connected to all the nodes of the preceding layer i.e. densely connected.



from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential([
    Dense(128, activation = 'relu', input_shape = (784,)),"relu is the activation function which is a simple linear function that is linear for positive values and 0 for other and is most commonly used in classification"
    Dense(128, activation = 'relu'),
    Dense(10, activation = 'softmax')"softmax gives the probability of each classes of output and all these add upto 1.0"
])


# We compile the code using sgd optimiser and the loss here the difference between our result and the result defined in data test set


model.compile(
    optimizer = 'sgd',
    loss = 'categorical_crossentropy',
    metrics = ['accuracy']
)

model.summary()


# now train the model on the normalised data set against the output encoded y 



h = model.fit(
    x_train_norm,
    y_train_encoded,
    epochs = 3
)


# evaluate the performance on the test set.

loss, accuracy = model.evaluate(x_test_norm, y_test_encoded)
print('test set accuracy: ', accuracy * 100)


# Predictions


preds = model.predict(x_test_norm)
print('shape of preds: ', preds.shape)


# plot the first few test set images along with their predicted and actual labels and see how our trained model actually performed.

plt.figure(figsize = (12, 12))
start_index = 0

for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    pred = np.argmax(preds[start_index + i]) " finds the max probability"
    actual = np.argmax(y_test_encoded[start_index + i])
    col = 'g'
    if pred != actual:
        col = 'r'
    plt.xlabel('i={} | pred={} | true={}'.format(start_index + i, pred, actual), color = col)
    plt.imshow(x_test[start_index + i], cmap='binary')
plt.show()



"""
use this command for checking the probabilty graph
"""
index = 8

plt.plot(preds[index])
plt.show()

