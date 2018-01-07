class DataCol:
  def __init__(self,name,dtype,cats):
    self.name  = name
    self.dtype = dtype
    self.cats  = cats

  def __str__(self):
    return '[DataCol ('+self.dtype+') \t'+self.name+']'

  def encode(self,data):
    try:
      if self.dtype == 'rv':     # real valued
        return float(data)
      elif self.dtype == 'iv':   # integer valued
        return float(data)
      elif self.dtype == 'ord':  # ordinate
        return float(self.cats.index(data))
      elif self.dtype == 'cat':  # categorical
        return float(self.cats.index(data))
      else:
        print '(!) dtype "'+self.dtype+'" is not defined.'
        return [None]
    except: # ugly
      return [None]

def parse_data_description():
  with open('data/data_description.txt','r') as fmtfile:
    fmtstr = fmtfile.read()
  refmt = '(?:\A|\n)([^\s]*):.*\[(.*)\].*\n((?:\n.*\t.*)*|)'
  recat = '(?:\n(.*)\t.*)'
  fmts = re.findall(refmt,fmtstr)
  dataCols = {}
  for fmt in fmts:
    cats = [cat.strip() for cat in re.findall(recat,fmt[2])]
    dataCols.update({fmt[0]:DataCol(fmt[0],fmt[1],cats)})
  return dataCols

def load_and_parse_data():
  dataCols = parse_data_description()
  with open('data/train.csv') as csvfile:
    reader = csv.DictReader(csvfile)

    data.append([dataCols[header[i]].encode(c) for i,c in row])
  return data
