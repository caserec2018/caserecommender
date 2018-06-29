"""

    Running item recommendation algorithms

"""
from caserec.recommenders.item_recommendation.bprmf import BprMF

tr = '/home/user/Documentos/dataset/ml-100k/folds/0/train.dat'
te = '/home/user/Documentos/dataset/ml-100k/folds/0/test.dat'


BprMF(tr, te, batch_size=30).compute()
