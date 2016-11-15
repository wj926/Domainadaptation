import numpy as np
import cPickle as pickle
import tensorflow as tf
from sklearn.cross_validation import train_test_split
from sklearn.decomposition import PCA

learning_rate = 0.01
batch_size = 100 
training_epochs = 10
display_step    = 5

#autoencoder book load
f = open('encoded_bag.txt', 'r')
a = pickle.load(f)
f.close()
encoded_book = a[0:6238,:]
"""
#pca load
f = open('reduced_bag.txt', 'r')
b = pickle.load(f)
f.close()
reduced_book = b[0:6238,:]
pca = PCA(n_components=128)
pca_book = pca.fit_transform(reduced_book)
"""
#Feature number 
n_input = encoded_book.shape[1]

#Feature X,weight define
x = tf.placeholder("float", [None, n_input])
w = tf.Variable(tf.random_normal([n_input,1]))
#bias term
b = tf.Variable(tf.zeros([1]))
y = tf.placeholder("float", [None,])


beta1 = 0.001 #regularization parameter

#label define
label = np.zeros(encoded_book.shape[0])
label[3119:]=1
labels = np.zeros((label.shape[0],2))

for i in range(label.shape[0]):
    if label[i]==0:
        labels[i,0]=1
        labels[i,1]=0
    elif label[i]==1:
        labels[i,0]=0
        labels[i,1]=1
        

#data split
train_x, test_x, train_y, test_y = train_test_split(encoded_book, label, test_size=0.3, random_state=0)

#activation, cost
actv = tf.nn.softmax(tf.matmul(x, w) + b)
cost1 = tf.reduce_mean(-tf.reduce_sum(y*tf.log(actv)+ (1-y)*tf.log(1-actv))) 
reg_cost = tf.nn.l2_loss(w)
cost = cost1 + beta1 * (reg_cost)

#optimizer
optm = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

total_batch = int(train_x.shape[0] / batch_size)

#prediction, accuracy
pred = tf.equal(tf.argmax(actv, 1), tf.argmax(y, 1))    
accr = tf.reduce_mean(tf.cast(pred, tf.float32))

#initialization
init = tf.initialize_all_variables()
print ("Network constructed")


# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        num_batch = int(6238*0.7/batch_size)
        print("start!")
        # Loop over all batches
        for i in range(num_batch): 
            batch_xs = train_x[i*batch_size:min((i+1)*batch_size, train_x.shape[0]),:]
            batch_ys = train_y[i*batch_size:min((i+1)*batch_size, train_y.shape[0])]              
            print(i)
            print(sess.run(actv,feed_dict ={x: batch_xs, y: batch_ys} ))
            # Fit training using batch data
            sess.run(optm, feed_dict={x: batch_xs, y: batch_ys})
            # Compute average loss
            avg_cost += sess.run(cost, feed_dict={x: batch_xs, y: batch_ys})/num_batch

        # Display logs per epoch step
        if epoch % display_step == 0:
            print (avg_cost)
            
    print ("Optimization Finished!")
    
print(sess.run(accr.eval, feed_dict={x: test_x, y: test_y}))


'''
#launch graph
with tf.Session() as sess:
    tf.initialize_all_variables().run()
    print 'initialized'
    for step in range(iteration):
        for i in range(total_batch):
            batch_xs = train_x[i*batch_size:min((i+1)*batch_size, train_x.shape[0]),:]
            batch_ys = train_y[i*batch_size:min((i+1)*batch_size, train_y.shape[0])]

            _, l, predictions = sess.run([optimizer, total_loss, train_prediction], feed_dict = {X : batch_xs, y : batch_ys})

    if (step % 250 == 0) :
        print("minibatch loss at step %d: %f" %(step, l))
    
avg    
    tp = test_prediction2.eval()
    embed()
    for i in range(len(tp)):
        if tp[i] >= 0.5 :
            tp[i] = 1
        else :
            tp[i] = 0

    accu = np.sum(tp == test_y)/len(tp)
    print("test accuracy : %.1f%%" %(accu))
    
'''