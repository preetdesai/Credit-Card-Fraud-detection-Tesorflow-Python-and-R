import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# Reading the dataset

def read_dataset():
    df = pd.read_csv("C:\\Users\\Test\\Desktop\\deskdocs\\Study\\MSIS\\Project\\Analytics\\TensorFlow\\CCFDuncoded.csv", header = None)
    print(len(df.columns))
    print (df)
    X = df[df.columns[0:30]].values
    yl = df[df.columns[30]]
    # Encode the dependent variable
    encoder = LabelEncoder()
    encoder.fit(yl)
    y = encoder.transform(yl)
    Y = one_hot_encode(y)
    print(X.shape)
    return (X, Y, yl)
# Define the encoder function
def one_hot_encode(labels):
    n_labels = len(labels)
    n_unique_labels = len(np.unique(labels))
    one_hot_encode = np.zeros((n_labels, n_unique_labels))
    one_hot_encode[np.arange(n_labels), labels] = 1
    return one_hot_encode

# Read the dataset
X, Y, yl = read_dataset()
print (Y)
# Shuffle the dataset to mix the rows

#X, Y = shuffle(X, Y, random_state = 1)
model_path = "C:\\Users\\Test\\Desktop\\deskdocs\\Study\\MSIS\\Project\\Analytics\\TensorFlow\\CCED"
learning_rate = 0.3
training_epochs = 100
cost_history = np.empty(shape = [1],dtype=float)
n_dim = X.shape[1]
#print("n_dim", n_dim)
n_class = 2

#Define the number of hidden layers and neurons for each layer
n_hidden_1 = 25
n_hidden_2 = 20
n_hidden_3 = 15
n_hidden_4 = 10

x = tf.placeholder(tf.float32, [None, n_dim])
W = tf.Variable(tf.zeros([n_dim, n_class]))
b = tf.Variable(tf.zeros([n_class]))
y_ = tf.placeholder(tf.float32, [None, n_class])

# Define the model
def multi_perceptron(x, weights, biases):
    # Hidden layer with sigmoid Activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.sigmoid(layer_1)
    
    # Hidden layer with sigmoid Activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.sigmoid(layer_2)
    
    # Hidden layer with sigmoid Activation
    layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
    layer_3 = tf.nn.sigmoid(layer_3)
    
    # Hidden layer with RELU Activation
    layer_4 = tf.add(tf.matmul(layer_3, weights['h4']), biases['b4'])
    layer_4 = tf.nn.sigmoid(layer_4)
    
    # Output layer with linear activation
    out_layer = tf.matmul(layer_4, weights['out']) + biases['out']
    return out_layer

# Define the weights and the biases for each layer

weights = {
    'h1': tf.Variable(tf.truncated_normal([n_dim, n_hidden_1])),
    'h2': tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2])),
    'h3': tf.Variable(tf.truncated_normal([n_hidden_2, n_hidden_3])),
    'h4': tf.Variable(tf.truncated_normal([n_hidden_3, n_hidden_4])),
    'out': tf.Variable(tf.truncated_normal([n_hidden_4, n_class])),
    }

biases = {
    'b1': tf.Variable(tf.truncated_normal([n_hidden_1])),
    'b2': tf.Variable(tf.truncated_normal([n_hidden_2])),
    'b3': tf.Variable(tf.truncated_normal([n_hidden_3])),
    'b4': tf.Variable(tf.truncated_normal([n_hidden_4])),
    'out': tf.Variable(tf.truncated_normal([n_class])),
    }    
    
# Initialize all the variables
init = tf.global_variables_initializer()
saver = tf.train.Saver()

# Call the model
y = multi_perceptron(x, weights, biases)
print (y)
# Define the cost function and optimizer
cost_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = y, labels = y_))
training_step = tf.train.AdamOptimizer(learning_rate).minimize(cost_function)



init = tf.global_variables_initializer()
saver = tf.train.Saver()
sess = tf.Session()
sess.run(init)
saver.restore(sess, model_path)

prediction = tf.argmax(y,1)
correct_prediction = tf.equal(prediction, tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Print (Accuracy)

y_true = []
y_pred = []
for i in range(1, 200):
    prediction_run = sess.run(prediction, feed_dict = {x: X[i].reshape(1,30)})
    #accuracy_run = sess.run(accuracy, feed_dict = {x: X[i].reshape(1,30), y_:y[i]})
    print("Original Class : ", yl[i], "Predicted Values : ", prediction_run )
    y_true.append(yl[i])
    y_pred.append(prediction_run)
  
#print (y_true)
#print (y_pred)  


# print confusion matrix  
cnf_matrix = confusion_matrix(y_true, y_pred)

