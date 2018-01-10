import tensorflow as tf
import numpy as np
import pandas as pd
import itertools
import data
import os

def get_data(datafile,train=True):

  def nanfix(x,fill=0):
    return np.asscalar(np.where(np.isnan(float(x)),fill,float(x)))

  feats = data.names()
  cats  = data.cats()
  df = pd.read_csv(datafile,dtype=str)
  df.fillna('nan',inplace=True)
  df = df.apply(lambda x: x.astype(str).str.lower())

  X  = [[nanfix(x,0)
         for x in df[c].values]  for c in feats['cts']]\
     + [[float(cats[c].index(x))
         for x in df[c].values]  for c in feats['ord']]\
     + np.transpose(np.concatenate([np.eye(len(cats[c]))[[cats[c].index(x)
         for x in df[c].values]] for c in feats['cat']],axis=1)).tolist()
  tX = np.transpose(X)
  if train:
    tY = np.transpose([[nanfix(y,0) for y in df[c].values] for c in feats['out']])
  else:
    tY = []
  tid = df[feats['id']]
  return tX,tY,tid

def init_weights(shape):
  return tf.Variable(tf.random_normal(shape, stddev=np.sqrt(2.0/shape[0])))

def model(X, W):
  h = X
  for wi in W[:-1]:
    h = tf.nn.relu(tf.matmul(h, wi))
  return tf.matmul(h,W[-1])

# initialization
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.reset_default_graph()

# load the data
trX,trY,trId = get_data('data/train.csv')
teX,teY,teId = get_data('data/test.csv',train=False)

# define the parameters
size_in  = trX.shape[1]
size_out = trY.shape[1]
size_h   = [size_in,128,32,size_out]
batch_size = 128

# define tf variables
X  = tf.placeholder("float", [None, size_in])
Y  = tf.placeholder("float", [None, size_out])
W  = [init_weights([hi,ho]) for hi,ho in zip(size_h[:-1],size_h[1:])]

# define the model & operations
Yp = model(X, W)
cost       = tf.reduce_mean(tf.pow(Yp-Y, 2.))
optimizer  = tf.train.ProximalAdagradOptimizer(
                learning_rate = 0.05,
                l1_regularization_strength = 0.0001,
                l2_regularization_strength = 0.0001)
grads      = optimizer.compute_gradients(cost,W)
train_op   = optimizer.minimize(cost)
predict_op = Yp

# Launch the graph in a session
with tf.Session() as sess:
  tf.global_variables_initializer().run()

  for i in range(5000):
    #print W[0][0][0].eval(session=sess) # weight
    #print sess.run([grad for grad,_ in grads] ,feed_dict={X: trX, Y: trY}) # gradient
    for start, end in zip(range(0, len(trX), batch_size),
                          range(batch_size, len(trX)+1, batch_size)):
      sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end]})
    print str(i+1).rjust(5)+': '\
         +str(int(np.mean([abs(y-yp) for y,yp in
          zip(trY,sess.run(predict_op, feed_dict={X: trX}))]))).rjust(9)
         # +str(trY[0][0]).rjust(9)+' vs '\
         # +str(sess.run(predict_op, feed_dict={X: trX})[0][0]).rjust(9)

  # predict the test data
  predictions = sess.run(predict_op, feed_dict={X: teX})
  output      = pd.DataFrame(
                  {'Id'       : [v[0] for v in teId.values],
                   'SalePrice': [v[0] for v in predictions]})
  # write submission
  with open('data/tmp.csv','w') as outfile:
    output.to_csv(outfile,index=False)
