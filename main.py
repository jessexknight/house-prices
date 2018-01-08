import re
import csv
import copy
import tempfile
import pandas as pd
import numpy  as np
import tensorflow as tf

def merge(d0,*args):
  d = d0.copy()
  for arg in args:
    d.update(arg)
  return d

def nanfix(x,fill=0):
  return np.asscalar(np.where(np.isnan(x),fill,x))

def get_cols():
  refmt = '(?:\A|\n)([^\s]*):.*\[(.*)\].*\n((?:\n.*\t.*)*|)'
  recat = '(?:\n(.*)\t.*)'
  with open('data/data_description.txt','r') as fmtfile:
    colfmts = re.findall(refmt,fmtfile.read())
    cols   = [colfmt[0] for colfmt in colfmts]
    dtypes = {colfmt[0]:colfmt[1] for colfmt in colfmts}
    cats   = {colfmt[0]:[cat.strip()
              for cat in re.findall(recat,colfmt[2])
              if cat.strip() is not 'NA']+['nan']
              for colfmt in colfmts}
  return cols,dtypes,cats

def load_data(datafile):
  cols,dtypes,cats = get_cols()
  df = pd.read_csv(datafile,names=[c for c in cols],skiprows=1,dtype=str)
  df.fillna('nan',inplace=True)
  names = {'id':  [c for c in cols if dtypes[c] in ['id']],
           'out': [c for c in cols if dtypes[c] in ['out']],
           'cts': [c for c in cols if dtypes[c] in ['rv','iv']],
           'ord': [c for c in cols if dtypes[c] in ['ord']],
           'cat': [c for c in cols if dtypes[c] in ['cat']]}
  # some meta-data
  meta  = {'len': df.shape[0],
           'Id':  df['Id'].values}
  # feature columns (symbolic)
  fcols = {'cts': {k: tf.feature_column.numeric_column(k)
                  for k in names['cts']},
           'ord': {k: tf.feature_column.numeric_column(k)
                  for k in names['ord']},
           'cat': {k: []
                  for k in names['cat']}}
  # tensors (data)
  cols = {'id':  {k: tf.constant([nanfix(float(v),0)
             for v in df[k].values])
             for k in names['out']},
          'out': {k: tf.constant([nanfix(float(v),0)
             for v in df[k].values])
             for k in names['out']},
          'cts': {k: tf.constant([nanfix(float(v),0)
             for v in df[k].values])
             for k in names['cts']},
          'ord': {k: tf.constant([float(cats[k].index(v))
             for v in df[k].values])
             for k in [c for c in cols if dtypes[c] in ['ord']]},
          'cat': {k: tf.SparseTensor( # FIX
               indices=[[i,0] for i in [cats[k].index(v) for v in df[k].values]],
               values=[1 for v in df[k].values],
               dense_shape=[df[k].shape[0],len(dtypes[c])])
             for k in [c for c in cols if dtypes[c] in ['cat']]}}
  x  = merge(cols['out'], cols['cts'], cols['ord'])# + cols_cat)
  fx = fcols['cts'].values()+fcols['ord'].values()
  return x,fx,meta,names

def input_fn(x,outname,batchsize=128):
  dataset = tf.data.Dataset.from_tensors(x)
  dataset.shuffle(buffer_size=batch_size)
  dataset.batch(batchsize)
  iterator = dataset.make_one_shot_iterator()
  xi = iterator.get_next()
  yi = xi.pop(outname)
  return xi,yi

if __name__ == '__main__':
  # hyperparameters
  learning_rate = 0.1
  l1_lambda     = 0.001
  l2_lambda     = 0.001
  batch_size    = 128
  # init tf session
  tf.reset_default_graph()
  sess = tf.InteractiveSession()
  # load data
  x,fx,meta,names = load_data('data/train.csv')
  # define training and prediction functions
  input_fn_train = lambda: input_fn(x,names['out'][0])
  input_fn_valid = tf.estimator.inputs.numpy_input_fn(
    {k:v.eval() for k,v in x.iteritems() if k is not names['out'][0]},
    num_epochs=1,shuffle=False)
  # define the optimizer and model
  optimizer = tf.train.ProximalAdagradOptimizer(
    learning_rate = learning_rate,
    l1_regularization_strength=l1_lambda,
    l2_regularization_strength=l2_lambda)
  model = tf.estimator.DNNRegressor(hidden_units=[64,64],
                                    feature_columns=fx,
                                    optimizer=optimizer,
                                    model_dir=tempfile.mkdtemp())
  # train the model
  for e in range(100):
    for i in range(int(meta['len']/batch_size)):
      model.train(input_fn=input_fn_train)
    results = model.evaluate(input_fn=input_fn_train)
    print '\n['+str(e+1).zfill(4)+'] '+\
          'J: '+str(int(np.sqrt(results['average_loss']))).zfill(6)
    predictions = list(model.predict(input_fn=input_fn_valid))
    for i in range(0,10):
      print str(int(x[names['out'][0]][i].eval()))+' -> '\
           +str(int(predictions[i]['predictions'][0]))
  # predict the test data
  xt,_,meta,_ = load_data('data/test.csv')
  input_fn_test = tf.estimator.inputs.numpy_input_fn(
    {k:v.eval() for k,v in xt.iteritems() if k is not names['out'][0]},
    num_epochs=1,shuffle=False)
  predictions = list(model.predict(input_fn=input_fn_test))
  output      = pd.DataFrame(
                  {'Id'       : meta['Id'],
                   'SalePrice': [p['predictions'][0] for p in predictions]})
  with open('data/out.csv','w') as outfile:
    output.to_csv(outfile,index=False)
  sess.close()
