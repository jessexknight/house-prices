import re
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

def predict(model,outfile=None):
  # load the test data
  xt,_,names = load_data('data/test.csv')
  input_fn_test = lambda: input_fn(xt,names['out'][0])
  # compute the test loss
  results = model.evaluate(input_fn_test)
  print '[test]\tJ = '+str(results['average_loss'])

def load_data(datafile):
  cols,dtypes,cats = get_cols()
  df = pd.read_csv(datafile,names=[c for c in cols],skiprows=1,dtype=str)
  df.fillna('nan',inplace=True)
  names = {'id':  [c for c in cols if dtypes[c] in ['id']],
           'out': [c for c in cols if dtypes[c] in ['out']],
           'cts': [c for c in cols if dtypes[c] in ['rv','iv']],
           'ord': [c for c in cols if dtypes[c] in ['ord']],
           'cat': [c for c in cols if dtypes[c] in ['cat']]}
  # feature columns (symbolic)
  fcols = {'cts': {k: tf.feature_column.numeric_column(k)
                  for k in names['out']}, # dummy fit
           'ord': {k: []
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
          'ord': {k: tf.constant([float(cats[k].index(v)) # FIX
             for v in df[k].values])
             for k in [c for c in cols if dtypes[c] in ['ord']]},
          'cat': {k: tf.SparseTensor( # FIX
               indices=[[i,0] for i in [cats[k].index(v) for v in df[k].values]],
               values=[1 for v in df[k].values],
               dense_shape=[df[k].shape[0],len(dtypes[c])])
             for k in [c for c in cols if dtypes[c] in ['cat']]}}
  x  = cols['out']
  # x  = merge(cols['out'], cols['cts'])#, cols_ord)# + cols_cat)
  fx = fcols['cts'].values()
  return x,fx,names

def input_fn(x,outname):
  dataset = tf.data.Dataset.from_tensors(x)
  dataset.batch(128)
  iterator = dataset.make_one_shot_iterator()
  xi = iterator.get_next()
  # yi = xi.pop(outname) # real
  yi = xi[outname] # dummy fit
  return xi,yi

if __name__ == '__main__':
  tf.reset_default_graph()
  sess = tf.InteractiveSession()
  x,fx,names = load_data('data/train.csv')
  input_fn_train = lambda: input_fn(x,names['out'][0])
  optimizer = tf.train.ProximalAdagradOptimizer(
    learning_rate = 0.5,
    l1_regularization_strength=0.0001)
  model = tf.estimator.DNNRegressor(hidden_units=[5],
                                    feature_columns=fx,
                                    # optimizer=optimizer,
                                    model_dir=tempfile.mkdtemp())
  for e in range(10):
    for i in range(1):
      model.train(input_fn=input_fn_train)
    results = model.evaluate(input_fn=input_fn_train)
    # print '['+str(e+1).zfill(4)+']\tJ = '+str(results['average_loss'])
    predict = model.predict(input_fn=input_fn_train)
    for i in range(0,10):
      print str(x['Id'][i].eval())+' -> '+str(predict.next()['predictions'][0])
  sess.close()
