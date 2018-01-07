import re
import tempfile
import pandas as pd
import numpy  as np
import tensorflow as tf

def merge(d1,d2):
  d = d1.copy()
  d.update(d2)
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
  # feature columns
  f_cols_num = {k: tf.feature_column.numeric_column(k)
               for k in [c for c in cols if dtypes[c] in ['rv','iv']]}
  # f_cols_ord = {k: []
  #              for k in [c for c in cols if dtypes[c] in ['ord']]}
  # f_cols_cat = {k: []
  #              for k in [c for c in cols if dtypes[c] in ['cat']]}
  # tensors
  cols_out = {k: tf.constant([nanfix(float(v),0)
             for v in df[k].values])
             for k in [c for c in cols if dtypes[c] in ['out']]}
  cols_num = {k: tf.constant([nanfix(float(v),0)
             for v in df[k].values])
             for k in [c for c in cols if dtypes[c] in ['rv','iv']]}
  # cols_ord = {k: tf.constant([float(cats[k].index(v))
  #            for v in df[k].values])
  #            for k in [c for c in cols if dtypes[c] in ['ord']]}
  # cols_cat = {k: tf.SparseTensor(
  #              indices=[[i,0] for i in [cats[k].index(v) for v in df[k].values]],
  #              values=[1 for v in df[k].values],
  #              dense_shape=[df[k].shape[0],len(dtypes[c])])
  #            for k in [c for c in cols if dtypes[c] in ['cat']]}
  x  = merge(cols_num, cols_out)# + cols_ord + cols_cat}
  fx = f_cols_num.values()
  return x,fx

def input_fn(x):
  dataset = tf.data.Dataset.from_tensors(x)
  dataset.batch(128)
  iterator = dataset.make_one_shot_iterator()
  xi = iterator.get_next()
  yi = xi.pop('SalePrice')
  return xi,yi

if __name__ == '__main__':
  x,fx = load_data('data/train.csv')
  model = tf.estimator.DNNRegressor(hidden_units=[128,128],
                                      feature_columns=fx,
                                      model_dir=tempfile.mkdtemp())
  for e in range(250):
    for i in range(5):
      model.train(input_fn=lambda: input_fn(x))
    results = model.evaluate(input_fn=lambda: input_fn(x))
    print '['+str(e).zfill(4)+']\tJ = '+str(results['average_loss'])

  xt,_ = load_data('data/test.csv')
  results = model.evaluate(input_fn=lambda: input_fn(xt))
  print '[test]\tJ = '+str(results['average_loss'])
