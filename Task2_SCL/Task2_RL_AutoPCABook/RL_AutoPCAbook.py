import numpy as np
import cPickle as pickle
import tensorflow as tf
from IPython import embed
from sklearn.cross_validation import train_test_split
from sklearn.decomposition import TruncatedSVD

#parameter
learning_rate = 0.001
batch_size = 1000 
training_epochs = 50
display_step    = 10

#autoencoder book load
f = open('encoded_bag.txt', 'r')
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
w = tf.Variable(tf.random_normal([n_input,1]))
#bias term
b = tf.Variable(tf.zeros([1]))
y = tf.placeholder("float", [None,1])


beta1 = 0.001 #regularization parameter

#label define
label = np.zeros((encoded_book.shape[0],1))
label[3119:]=1


   

#data split
train_x, test_x, train_y, test_y = train_test_split(con_book, label, test_size=0.3, random_state=0)

#activation, cost
actv = tf.sigmoid((tf.matmul(x, w) + b))
#cost1 = -tf.nn.sigmoid_cross_entropy_with_logits(actv, y, name=None) 
cost1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(actv, -y, name=None)) 
reg_cost = tf.nn.l2_loss(w)
cost = cost1 + beta1*(reg_cost)

#optimizer
optm = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
total_batch = int(train_x.shape[0] / batch_size)


# Launch the graph
init = tf.initialize_all_variables()
print ("Network constructed")

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
                
            #print(sess.run(actv,feed_dict ={x: batch_xs, y: batch_ys} ))
            # Fit training using batch data
            sess.run(optm, feed_dict={x: batch_xs, y: batch_ys})
            # Compute average loss
            avg_cost += sess.run(cost, feed_dict={x: batch_xs, y: batch_ys})/num_batch
            
        #Display logs per epoch step
            if epoch % display_step == 0:
                print(avg_cost)
            
    print ("Optimization Finished!")

#accuracy
    tp = sess.run(actv, feed_dict={x:test_x})

for i in range(len(tp)):
        if tp[i] >= 0.5:
            tp[i] = 1
        else :
            tp[i] = 0
    
qwe = np.zeros((len(tp),1))
    
for i in range(len(tp)):
    qwe[i,0] = test_y[i]
        
right = 0.
for i in range(len(tp)):
        if qwe[i,0] == tp[i,0]:
            right = right +1
    
accu = right/len(tp)
accu

#test
act = 1/(1+ np.exp(-((np.matmul(xxx,aa) + bb))))
cost11 = -tf.reduce_mean((yyy*np.log(act) + (1-yyy)*np.log(1-act)))

ccc = (yyy*np.log(act) + (1-yyy)*np.log(1-act))
cost11 = -np.sum(ccc)/(200*200)
cost22 = np.linalg.norm(aa)
cost22*0.001 + cost11
