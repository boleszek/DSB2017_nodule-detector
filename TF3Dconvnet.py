import tensorflow as tf
import numpy as np
import pandas as pd
import os

# Initialize variables
IMG_DIM1 = 100
IMG_DIM2 = 120
SLICE_COUNT = 60

MIN_BOUND = -1000.0 # for image normalization
MAX_BOUND = 400.0

PIXEL_MEAN = 0.25 # use the mean of the ENTIRE DATA SET!

n_classes = 2
batch_size = 10

x = tf.placeholder('float')
y = tf.placeholder('float')

keep_rate = 0.8 # for dropout (not used)

# data folder
dfold = '3Darrays_stage1'


# load training names and labels (no validation)
Y_df = pd.read_csv('stage1_labels.csv', index_col=0)
train_data = [name +'.npy' for name in Y_df.index]


#============== Define Subfunctions ================#

def conv3d(x, W):
    return tf.nn.conv3d(x, W, strides=[1,1,1,1,1], padding='SAME')

def maxpool3d(x):
    #                        size of window         movement of window as you slide about
    return tf.nn.max_pool3d(x, ksize=[1,2,2,2,1], strides=[1,2,2,2,1], padding='SAME')

# This is the network graph
def convolutional_neural_network(x):
    # manually calculate the number of feature maps before the dense layer
    nfm = np.int32(np.ceil(IMG_DIM1/2/2)*np.ceil(IMG_DIM2/2/2)*np.ceil(SLICE_COUNT/2/2)*64)
    
    # Initialize weights with a small positive random noise (to avoid "dead" neurons due to ReLU ignoring negative)
    #                # 5 x 5 x 5 patches, 1 channel, 32 features to compute.
    weights = {'W_conv1':tf.Variable(tf.truncated_normal([5,5,5,1,32],stddev=0.1)),
               #       5 x 5 x 5 patches, 32 channels, 64 features to compute.
               'W_conv2':tf.Variable(tf.truncated_normal([5,5,5,2,64],stddev=0.1)),
               #                       nfm fully connected   1024 features
               'W_fc':tf.Variable(tf.truncated_normal([nfm,1024],stddev=0.1)),
               'out':tf.Variable(tf.truncated_normal([1024, n_classes],stddev=0.1))}

    # Sentdex used tf.random_normal to initialize biases, but MNIST TF tutorial just defines them with tf.constant
    biases = {'b_conv1':tf.Variable(tf.random_normal([32])),
               'b_conv2':tf.Variable(tf.random_normal([64])),
               'b_fc':tf.Variable(tf.random_normal([1024])),
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


def one_hot(Y):
    # force labels to be one-hot encoded
    if Y == 1:
        Y = [1,0]
    else:
        Y = [0,1]
    return Y
    
def normalize(image):
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image>1] = 1.
    image[image<0] = 0.
    return image

def zero_center(image):
    image = image - PIXEL_MEAN
    return image


#=========== MAIN function ==============#
def train_neural_network(x):
    prediction = convolutional_neural_network(x)
    # Only call `softmax_cross_entropy_with_logits` with named arguments (labels=..., logits=...)
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y) )
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(cost)
    
    saver = tf.train.Saver()
    
    hm_epochs = 5
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        successful_runs = 0
        total_runs = 0
        
        for epoch in range(hm_epochs):
            epoch_loss = 0
            for data in train_data:
                total_runs += 1
                try:
                    X = np.load(dfold+'/'+data)
                    X = normalize(X)
                    X = zero_center(X)
                    Y = one_hot(Y_df.get_value(data[:-4],'cancer'))
                    _, c = sess.run([optimizer, cost], feed_dict={x: X, y: Y})
                    epoch_loss += c
                    print('Epoch ', epoch+1, ' | ',data,' completed ','loss:',epoch_loss)
                    successful_runs += 1
                except Exception as e:
                    # I am passing for the sake of notebook space, but we are getting 1 shaping issue from one 
                    # input tensor. Not sure why, will have to look into it. Guessing it's
                    # one of the depths that doesn't come to 20.
                    print('error')
                    #print('epoch '+ epoch + ' ' + data)
                    pass
            
            print('Epoch', epoch+1, 'completed out of',hm_epochs,'cum loss:',epoch_loss)

            correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

            #print('Accuracy:',accuracy.eval({x:[np.load(dfold+'/'+i) for i in val_data], y:[one_hot(Y_df.get_value(data[:-4],'cancer')) for i in val_data]}))
            #print('Accuracy:',accuracy.eval({x:[np.load(dfold+'/'+i) for i in test_data], y:[one_hot(test_df.get_value(data[:-4],'cancer')) for i in test_data]}))
            
        #print('Done. Final accuracy:')
        #print('Accuracy:',accuracy.eval({x:[np.load(dfold+'/'+i) for i in val_data], y:[one_hot(Y_df.get_value(data[:-4],'cancer')) for i in val_data]}))
        #print('Accuracy:',accuracy.eval({x:[np.load(dfold+'/'+i) for i in test_data], y:[one_hot(test_df.get_value(data[:-4],'cancer')) for i in test_data]}))
        
        print('Percent files used:',successful_runs/total_runs)
        
        save_path = saver.save(sess, "Stage1Model.ckpt")
        print("Model saved in file: %s" % save_path)
        
        # Finally, run the model on the test data
        #test_predictions = [sess.run(prediction, feed_dict={x:[np.load(dfold+'/'+i)]}) for i in test_data]
        
    return prediction
        
        
#=========== Run Locally =============#
trainp=train_neural_network(x)
np.save('train_predictions.npy',trainp)

