import re
import pandas as pd
import numpy  as np
import tensorflow as tf

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
  cols_num = {k: tf.constant([float(v)
             for v in df[k].values])
             for k in [c for c in cols if dtypes[c] in ['rv','iv']]}
  cols_ord = {k: tf.constant([float(cats[k].index(v))
             for v in df[k].values])
             for k in [c for c in cols if dtypes[c] in ['ord']]}
  cols_cat = {k: tf.SparseTensor(
               indices=[[i,0] for i in [cats[k].index(v) for v in df[k].values]],
               values=[1 for v in df[k].values],
               dense_shape=[df[k].shape[0],len(dtypes[c])])
             for k in [c for c in cols if dtypes[c] in ['cat']]}
  features = dict(cols_num.items() + cols_ord.items() + cols_cat.items())
  output = features.pop('SalePrice')
  return features,output

if __name__ == '__main__':
  x0,y = load_data('data/train.csv')
