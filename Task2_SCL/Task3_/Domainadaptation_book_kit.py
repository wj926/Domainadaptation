
import numpy as np
import cPickle as pickle
import tensorflow as tf
from IPython import embed
from sklearn.cross_validation import train_test_split
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn import svm, cross_validation
from sklearn.linear_model import LogisticRegression


#parameter
learning_rate = 0.1
batch_size = 1000 
training_epochs = 4000
display_step    = 10

#autoencoder book load
f = open('encoded_8138.txt', 'r')
con_auto = pickle.load(f)
f.close()
encoded_book = con_auto[0:6238,:]
encoded_kit = con_auto[6238:8318,:]


#SVD load
f = open('reduced_bag.txt', 'r')
con_bag = pickle.load(f)
f.close()
reduced_book = con_bag[0:6238,:]
reduced_kit = con_bag[6238:8318,:]
svd = TruncatedSVD(n_components=128)
svd_book = svd.fit_transform(reduced_book)
svd_kit = svd.fit_transform(reduced_kit)


#concatenate 
con_book = np.concatenate((encoded_book, svd_book), axis = 1)
con_kit = np.concatenate((encoded_kit, svd_kit), axis = 1)


#Feature number 
n_input = con_book.shape[1]

#Feature X,weight define
x = tf.placeholder("float", [None, n_input])
w = tf.Variable(tf.random_normal([n_input,2]))
#bias term
b = tf.Variable(tf.zeros([2]))
y = tf.placeholder("float", [None,2])


#regularization parameter
beta1 = 0.001 
beta2 = 0.001 

#label define
label_book = np.zeros((encoded_book.shape[0],1))
label_book[3119:]=1

label_kit = np.zeros((encoded_kit.shape[0],1))
label_kit[1040:]=1

labels_book = np.zeros((label_book.shape[0],2))

for i in range(label_book.shape[0]):
    if label_book[i]==0:
        labels_book[i,0]=1
        labels_book[i,1]=0
    elif label_book[i]==1:
        labels_book[i,0]=0
        labels_book[i,1]=1

label_kit = np.zeros((encoded_kit.shape[0],1))
label_kit[1040:]=1

labels_kit = np.zeros((label_kit.shape[0],2))

for i in range(label_kit.shape[0]):
    if label_kit[i]==0:
        labels_kit[i,0]=1
        labels_kit[i,1]=0
    elif label_kit[i]==1:
        labels_kit[i,0]=0
        labels_kit[i,1]=1 
        
        
"""session for w1"""        

#data split
train_x_book, test_x_book, train_y_book, test_y_book = train_test_split(con_book, labels_book, test_size=0.3, random_state=3)

#activation, cost
#actv = tf.sigmoid((tf.matmul(x, w) + b))
actv = tf.nn.softmax((tf.matmul(x, w) + b))
#cost1 = -tf.reduce_mean((y*tf.log(actv) + (1-y)*tf.log(1-actv))) 
cost1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(actv, y, name=None)) 
#cost1 = tf.reduce_mean(-tf.reduce_sum(y*tf.log(actv), reduction_indices=1)) 

reg_cost_auto = tf.nn.l2_loss(w[0:128,:])
reg_cost_svd = tf.nn.l2_loss(w[128:256,:])

cost = cost1 + beta1*(reg_cost_auto) + beta2*(reg_cost_svd)

#optimizer
optm = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
total_batch = int(train_x_book.shape[0] / batch_size)

#accuracy
pred = tf.equal(tf.argmax(actv, 1), tf.argmax(y, 1))    
accr = tf.reduce_mean(tf.cast(pred, "float"))

# Launch the graph
init = tf.initialize_all_variables()
print ("Network constructed")

with tf.Session() as sess:
    sess.run(init)
    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        num_batch = int(6238*0.7/batch_size)
        #print("start!")
        # Loop over all batches
        for i in range(num_batch): 
            batch_xs = train_x_book[i*batch_size:min((i+1)*batch_size, train_x_book.shape[0]),:]
            batch_ys = train_y_book[i*batch_size:min((i+1)*batch_size, train_y_book.shape[0])]
                          
            # Fit training using batch data
            sess.run(optm, feed_dict={x: batch_xs, y: batch_ys})
            
            # Compute average loss
        avg_cost = sess.run(cost, feed_dict={x: batch_xs, y: batch_ys})/num_batch
        tp = sess.run(actv, feed_dict={x:test_x_book})    
        #Display logs per epoch step
        if epoch % display_step == 0:
            print(avg_cost)
            print ("Accuracy:", accr.eval({x:test_x_book, y: test_y_book}))        
            w_book = sess.run(w)
            b_book = sess.run(b)
print ("Optimization Finished!")









#regularization parameter
beta1 = 0.1 
beta2 = 0.1 


#data split
train_x_kit, test_x_kit, train_y_kit, test_y_kit = train_test_split(con_kit, labels_kit, test_size=0.3, random_state=3)

#activation, cost
#actv = tf.sigmoid((tf.matmul(x, w) + b))
actv = tf.nn.softmax((tf.matmul(x, w) + b))
#cost1 = -tf.reduce_mean((y*tf.log(actv) + (1-y)*tf.log(1-actv))) 
cost1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(actv, y, name=None)) 
#cost1 = tf.reduce_mean(-tf.reduce_sum(y*tf.log(actv), reduction_indices=1)) 

reg_cost_auto = tf.nn.l2_loss(w[0:128,:])
reg_cost_svd = tf.nn.l2_loss(tf.add(w[128:256,:],-w_book[128:256,:]))

cost = cost1 + beta1*(reg_cost_auto) + beta2*(reg_cost_svd)
cost2 = cost1 + beta1*(reg_cost_auto)



#optimizer
optm = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
total_batch = int(train_x_kit.shape[0] / batch_size)

#accuracy
pred = tf.equal(tf.argmax(actv, 1), tf.argmax(y, 1))    
accr = tf.reduce_mean(tf.cast(pred, "float"))

# Launch the graph
init = tf.initialize_all_variables()
print ("Network constructed")

with tf.Session() as sess:
    sess.run(init)
    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        num_batch = int(2080*0.7/batch_size)
        #print("start!")
        # Loop over all batches
        for i in range(num_batch): 
            batch_xs = train_x_kit[i*batch_size:min((i+1)*batch_size, train_x_kit.shape[0]),:]
            batch_ys = train_y_kit[i*batch_size:min((i+1)*batch_size, train_y_kit.shape[0])]
                          
            #print(i)
            if epoch == 0 and i==0:
                #print('w = ',sess.run(w))
                aa = sess.run(w)
                bb = sess.run(b)
                xxx = train_x_kit[0*batch_size:min((0+1)*batch_size, train_x_kit.shape[0]),:]
                yyy = train_y_kit[0*batch_size:min((0+1)*batch_size, train_y_kit.shape[0])]  
                #print('activ =',sess.run(actv,feed_dict={x: batch_xs, y: batch_ys}))
                
                print('cost1 = ',sess.run(cost1, feed_dict={x: batch_xs, y: batch_ys}))
                print('cost = ',sess.run(cost, feed_dict={x: batch_xs, y: batch_ys}))
                print(batch_xs.shape)
            if epoch == 2 and i==2:
                #print('w = ',sess.run(w))
                aaa = sess.run(w)
                bbb = sess.run(b)
                xxxx = train_x_kit[2*batch_size:min((2+1)*batch_size, train_x_kit.shape[0]),:]
                yyyy = train_y_kit[2*batch_size:min((2+1)*batch_size, train_y_kit.shape[0])]  
                #print('activ =',sess.run(actv,feed_dict={x: batch_xs, y: batch_ys}))
                
                print('cost1 = ',sess.run(cost1, feed_dict={x: batch_xs, y: batch_ys}))
                print('cost = ',sess.run(cost, feed_dict={x: batch_xs, y: batch_ys}))
                print(batch_xs.shape)
                           
            
            # Fit training using batch data
            sess.run(optm, feed_dict={x: batch_xs, y: batch_ys})
            
            # Compute average loss
        avg_cost = sess.run(cost, feed_dict={x: batch_xs, y: batch_ys})/num_batch
        tp = sess.run(actv, feed_dict={x:test_x_kit})    
        #Display logs per epoch step
        if epoch % display_step == 0:
            print(avg_cost)
            print ("Accuracy:", accr.eval({x:test_x_kit, y: test_y_kit}))        

print ("Optimization Finished!")





#auto kit test
train_x, test_x, train_y, test_y = train_test_split(encoded_kit, label_kit, test_size=0.3, random_state=3)
clf = LogisticRegression(C=0.1)
clf.fit(train_x, train_y)
print 'test phase auto kit'
predicts = clf.predict(test_x)
print accuracy_score(test_y, predicts)

#auto book test
train_x, test_x, train_y, test_y = train_test_split(encoded_book, label_book, test_size=0.3, random_state=3)
clf = LogisticRegression(C=1.0)
clf.fit(train_x, train_y)
print 'test phase'
predicts = clf.predict(test_x)
print accuracy_score(test_y, predicts)

#svm test
train_x, test_x, train_y, test_y = train_test_split(svd_kit, label_kit, test_size=0.3, random_state=3)
clf = LogisticRegression(C=1.0)
clf.fit(train_x, train_y)
print 'test phase'
predicts = clf.predict(test_x)
print accuracy_score(test_y, predicts)