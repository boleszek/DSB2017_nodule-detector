# This code assumes that the model has been stored in a checkpoint (.ckpt) file.
# This code restores the model, evaluates it on test set, and saves predictions in csv file

import tensorflow as tf
import numpy as np
import pandas as pd
import os
import pickle

# Initialize variables
IMG_DIM1 = 100
IMG_DIM2 = 120
SLICE_COUNT = 60
n_classes = 2

feat1 = 16
feat2 = 40
feat3 = 300

x = tf.placeholder('float')
y = tf.placeholder('float')


# establish train, val, and test data sets
dfold = '3Darrays_stage1'

# load test data
Ytest_df = pd.read_csv('stage1_sample_submission.csv', index_col=0)
test_data_name = [name +'.npy' for name in Ytest_df.index]
test_data = [np.load(dfold+'/'+name) for name in test_data_name]

print("Test data loaded")

def conv3d(x, W):
    return tf.nn.conv3d(x, W, strides=[1,1,1,1,1], padding='SAME')

def maxpool3d(x):
    #                        size of window         movement of window as you slide about
    return tf.nn.max_pool3d(x, ksize=[1,2,2,2,1], strides=[1,2,2,2,1], padding='SAME')

# This is the network graph
def convolutional_neural_network(x):
    # manually calculate the number of feature maps before the dense layer
    nfm = np.int32(np.ceil(IMG_DIM1/2/2)*np.ceil(IMG_DIM2/2/2)*np.ceil(SLICE_COUNT/2/2)*feat2)
    
    # Initialize weights with a small positive random noise (to avoid "dead" neurons due to ReLU ignoring negative)
    #                # 5 x 5 x 5 patches, 1 channel, 32 features to compute.
    weights = {'W_conv1':tf.Variable(tf.truncated_normal([5,5,5,1,feat1],stddev=0.1)),
               #       5 x 5 x 5 patches, 32 channels, 64 features to compute.
               'W_conv2':tf.Variable(tf.truncated_normal([5,5,5,feat1,feat2],stddev=0.1)),
               #                       nfm fully connected   1024 features
               'W_fc':tf.Variable(tf.truncated_normal([nfm,feat3],stddev=0.1)),
               'out':tf.Variable(tf.truncated_normal([feat3, n_classes],stddev=0.1))}

    # Sentdex used tf.random_normal to initialize biases, but MNIST TF tutorial just defines them with tf.constant
    biases = {'b_conv1':tf.Variable(tf.random_normal([feat1])),
               'b_conv2':tf.Variable(tf.random_normal([feat2])),
               'b_fc':tf.Variable(tf.random_normal([feat3])),
               'out':tf.Variable(tf.random_normal([n_classes]))}

    #                            image X   image Y      image Z   last dim corresponds to # of color chan
    x = tf.reshape(x, shape=[-1, IMG_DIM1, IMG_DIM2, SLICE_COUNT, 1])

    conv1 = tf.nn.relu(conv3d(x, weights['W_conv1']) + biases['b_conv1'])
    conv1 = maxpool3d(conv1)


    conv2 = tf.nn.relu(conv3d(conv1, weights['W_conv2']) + biases['b_conv2'])
    conv2 = maxpool3d(conv2)

    fc = tf.reshape(conv2,[-1, nfm])
    fc = tf.nn.relu(tf.matmul(fc, weights['W_fc'])+biases['b_fc'])
    #fc = tf.nn.dropout(fc, keep_rate)

    output = tf.matmul(fc, weights['out'])+biases['out']

    return output




# Later, launch the model, use the saver to restore variables from disk, and
# do some work with the model.
prediction = convolutional_neural_network(x)
# Only call `softmax_cross_entropy_with_logits` with named arguments (labels=..., logits=...)
cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y) )
optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(cost)
    
saver = tf.train.Saver()
    

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, "./Stage2Model2.ckpt")
    print("Model restored.")
    test_predictions = []
    for i in range(len(test_data)):
        test_predictions.append(sess.run(prediction, feed_dict={x:[test_data[i]]}))
        print(test_data_name[i],' completed | ',str(len(test_data)-i-1),' remain')
        
# save raw predictions just in case
try:
    pickle.dump( test_predictions, open( "test_predictions.p", "wb" ) )
except:
    print('Pickling didn''t work')
pass


# convert predictions to softmax probabilities (also trimming 0,1 edges)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer()) #execute init_op
    softmax_test_predictions = []
    for i in range(len(test_predictions)):
        temp = sess.run(tf.nn.softmax(test_predictions[i]))
        temp[temp==1]=0.95
        temp[temp==0]=0.05
        softmax_test_predictions.append(temp[0][0])
        
# write to csv
import pandas as pd
import numpy as np
labels_df = pd.read_csv('stage1_sample_submission.csv', index_col=0)
names = labels_df.index
test = pd.DataFrame(data=softmax_test_predictions,index=names,columns=['cancer'])
test.to_csv('stage1_submission_BO6.csv')