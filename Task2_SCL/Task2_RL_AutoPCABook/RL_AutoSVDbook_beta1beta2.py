
import numpy as np
import cPickle as pickle
import tensorflow as tf
from IPython import embed
from sklearn.cross_validation import train_test_split
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn import svm, cross_validation

#parameter
learning_rate = 0.1
batch_size = 1000 
training_epochs = 4000
display_step    = 10

#autoencoder book load
f = open('encoded_8138.txt', 'r')
a = pickle.load(f)
f.close()
encoded_book = a[0:6238,:]

#SVD load
f = open('reduced_bag.txt', 'r')
b = pickle.load(f)
f.close()
reduced_book = b[0:6238,:]
svd = TruncatedSVD(n_components=128)
svd_book = svd.fit_transform(reduced_book)

#concatenate 
con_book = np.concatenate((encoded_book, svd_book), axis = 1)


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
label = np.zeros((encoded_book.shape[0],1))
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
train_x, test_x, train_y, test_y = train_test_split(con_book, labels, test_size=0.3, random_state=3)

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
total_batch = int(train_x.shape[0] / batch_size)

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
            batch_xs = train_x[i*batch_size:min((i+1)*batch_size, train_x.shape[0]),:]
            batch_ys = train_y[i*batch_size:min((i+1)*batch_size, train_y.shape[0])]
                          
            #print(i)
            if epoch == 0 and i==0:
                #print('w = ',sess.run(w))
                aa = sess.run(w)
                bb = sess.run(b)
                xxx = train_x[0*batch_size:min((0+1)*batch_size, train_x.shape[0]),:]
                yyy = train_y[0*batch_size:min((0+1)*batch_size, train_y.shape[0])]  
                #print('activ =',sess.run(actv,feed_dict={x: batch_xs, y: batch_ys}))
                
                print('cost1 = ',sess.run(cost1, feed_dict={x: batch_xs, y: batch_ys}))
                print('cost = ',sess.run(cost, feed_dict={x: batch_xs, y: batch_ys}))
                print(batch_xs.shape)
            if epoch == 2 and i==2:
                #print('w = ',sess.run(w))
                aaa = sess.run(w)
                bbb = sess.run(b)
                xxxx = train_x[2*batch_size:min((2+1)*batch_size, train_x.shape[0]),:]
                yyyy = train_y[2*batch_size:min((2+1)*batch_size, train_y.shape[0])]  
                #print('activ =',sess.run(actv,feed_dict={x: batch_xs, y: batch_ys}))
                
                print('cost1 = ',sess.run(cost1, feed_dict={x: batch_xs, y: batch_ys}))
                print('cost = ',sess.run(cost, feed_dict={x: batch_xs, y: batch_ys}))
                print(batch_xs.shape)
                           
            
            # Fit training using batch data
            sess.run(optm, feed_dict={x: batch_xs, y: batch_ys})
            
            # Compute average loss
        avg_cost = sess.run(cost, feed_dict={x: batch_xs, y: batch_ys})/num_batch
        tp = sess.run(actv, feed_dict={x:test_x})    
        #Display logs per epoch step
        if epoch % display_step == 0:
            print(avg_cost)
            print ("Accuracy:", accr.eval({x:test_x, y: test_y}))        
            print ("Optimization Finished!")



#test
act = 1/(1+ np.exp(-((np.matmul(xxx,aa) + bb))))
ccc = (yyy*np.log(act) + (1-yyy)*np.log(1-act))
cost11 = -np.sum(ccc)/(200*200)
cost22 = np.linalg.norm(aa)
cost22*0.001 + cost11


