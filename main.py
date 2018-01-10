from sklearn.model_selection import train_test_split as tsplit
import tensorflow as tf
import numpy as np
import pandas as pd
import shutil
import data
import os

def get_data(datafile,loady=True):

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
  if loady:
    tY = np.transpose([[nanfix(y,0) for y in df[c].values] for c in feats['out']])
  else:
    tY = []
  tid = df[feats['id']]
  return tX,tY,tid

def init_weights(shape):
  return tf.Variable(tf.random_normal(shape, stddev=np.sqrt(2.0/shape[0])))

def build_model(X, W, dropout=0):
  h = X
  for wi in W[:-1]:
    h = tf.nn.dropout(tf.nn.relu(tf.matmul(h, wi)),1.0-dropout)
  return tf.nn.relu(tf.matmul(h,W[-1]))

# initialization
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
shutil.rmtree('log',ignore_errors=True)
tf.reset_default_graph()

# load the data
trX,trY,trId = get_data('data/train.csv')
teX,teY,teId = get_data('data/test.csv',loady=False)
trX,vaX,trY,vaY,trId,vaId = tsplit(trX,trY,trId,test_size=0.3)

# define the parameters
size_in  = trX.shape[1]
size_out = trY.shape[1]
size_h   = [size_in,512,64,size_out]
batch_size = 128

# define tf symbolic variables
X  = tf.placeholder("float", [None, size_in])
Y  = tf.placeholder("float", [None, size_out])
W  = [init_weights([hi,ho]) for hi,ho in zip(size_h[:-1],size_h[1:])]

# define the model & operations
Yt         = build_model(X, W, dropout=0.3)
Yp         = build_model(X, W, dropout=0.0)
objective  = tf.reduce_mean(tf.pow(Yt-Y, 2.))
Jfun       = tf.reduce_mean(tf.pow(Yp-Y, 2.))
optimizer  = tf.train.AdagradOptimizer(
                  learning_rate             = 0.1,
                  initial_accumulator_value = 0.1)
# optimizer  = tf.train.RMSPropOptimizer(
#                 learning_rate = 0.001,
#                 decay         = 0.9,
#                 momentum      = 0.1)
# optimizer  = tf.train.ProximalAdagradOptimizer(
#                 learning_rate = 0.05,
#                 l1_regularization_strength = 0.05,
#                 l2_regularization_strength = 0.01)
train_op   = optimizer.minimize(objective)
predict_op = Yp

# tensorboard
writer = {}
writer.update({'tr':tf.summary.FileWriter('./log/tr')})
writer.update({'va':tf.summary.FileWriter('./log/va')})
tf.summary.scalar('Objective',  Jfun)
summary = tf.summary.merge_all()

# Launch the graph in a session
with tf.Session() as sess:
  tf.global_variables_initializer().run()

  for e in range(10000):
    for start, end in zip(range(0, len(trX), batch_size),
                          range(batch_size, len(trX)+1, batch_size)):
      sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end]})
    writer['tr'].add_summary(sess.run(summary,feed_dict={X:trX,Y:trY}),e)
    writer['tr'].flush()
    writer['va'].add_summary(sess.run(summary,feed_dict={X:vaX,Y:vaY}),e)
    writer['va'].flush()
    if not e%100:
      print str(e).rjust(5)

  # predict the test data
  predictions = sess.run(predict_op, feed_dict={X: teX})
  output      = pd.DataFrame(
                  {'Id'       : [v[0] for v in teId.values],
                   'SalePrice': [v[0] for v in predictions]})
  # write submission
  with open('data/tmp.csv','w') as outfile:
    output.to_csv(outfile,index=False)
