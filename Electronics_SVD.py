#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!pip3 install --no-cache-dir --upgrade git+https://github.com/evfro/polara.git@develop#egg=polara


# In[2]:


import numpy as np
import pandas as pd
from tqdm import tqdm

import polara
from polara import get_movielens_data
from polara.preprocessing.dataframes import leave_one_out, reindex

from dataprep import transform_indices
from evaluation import topn_recommendations, downvote_seen_items

from polara.lib.tensor import hooi
from polara.lib.sparse import tensor_outer_at
from polara.evaluation.pipelines import random_grid

from sa_hooi import sa_hooi, form_attention_matrix, get_scaling_weights, generate_position_projector

from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import norm, svds
from scipy.linalg import solve_triangular, sqrtm

from IPython.utils import io
import pandas as pd


# # Electronics

# # Data preproccesing

# In[ ]:


import sys
import gzip
import tempfile
from io import BytesIO
from ast import literal_eval
from  urllib import request
import pandas as pd


def amazon_data_reader(path):
    with gzip.open(path, 'rt') as gz:
        for line in gz:
            yield literal_eval(line)

def read_amazon_data(path=None, name=None):
    '''Data is taken from https://jmcauley.ucsd.edu/data/amazon/'''
    if path is None and name is None:
            raise ValueError('Either the name of the dataset to download \
                or a path to a local file must be specified.')
    if path is None:
        file_url = f'http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_{name}_5.json.gz'
        print(f'Downloading data from: {file_url}')
        with request.urlopen(file_url) as response:
            file = response.read()
            with tempfile.NamedTemporaryFile(delete=False) as temp:
                temp.write(file)
                path = temp.name
                print(f'Temporarily saved file at: {path}')
    return pd.DataFrame.from_records(
        amazon_data_reader(path),
        columns=['reviewerID', 'asin', 'overall', 'unixReviewTime']
    )

data = read_amazon_data(name = "Electronics")
data.rename(columns = {'reviewerID' : 'userid', 'asin' : 'movieid', "overall" : "rating", "unixReviewTime" : "timestamp"}, inplace = True)


# In[3]:


def reindex_data_new(data, entity, data_index):
    field = data_index[entity].name
    new_index = data_index[entity].get_indexer(data[field])
    return data.assign(**{f'{field}': new_index})

def full_preproccessing(data = None):
    if (data is None):
        data = get_movielens_data("ml-10m.zip", include_time=True)
    test_timepoint = data['timestamp'].quantile(
    q=0.95, interpolation='nearest'
    )
    
    labels, levels = pd.factorize(data.movieid)
    data.loc[:, 'movieid'] = labels

    labels, levels = pd.factorize(data.userid)
    data.loc[:, 'userid'] = labels
    
    if (data["rating"].nunique() > 5):
        data["rating"] = data["rating"] * 2
        
    data["rating"] = data["rating"].astype(int)

    test_data_ = data.query('timestamp >= @test_timepoint')
    train_data_ = data.query(
    'userid not in @test_data_.userid.unique() and timestamp < @test_timepoint'
    )
    
    training, data_index = transform_indices(train_data_.copy(), 'userid', 'movieid')
    test_data = reindex(test_data_, data_index['items'])

    testset_, holdout_ = leave_one_out(
    test_data, target='timestamp', sample_top=True, random_state=0
    )
    testset_valid_, holdout_valid_ = leave_one_out(
        testset_, target='timestamp', sample_top=True, random_state=0
    )

    test_users_val = np.intersect1d(testset_valid_.userid.unique(), holdout_valid_.userid.unique())
    testset_valid = testset_valid_.query('userid in @test_users_val').sort_values('userid')
    holdout_valid = holdout_valid_.query('userid in @test_users_val').sort_values('userid')

    test_users = np.intersect1d(testset_.userid.unique(), holdout_.userid.unique())
    testset = testset_.query('userid in @test_users').sort_values('userid')
    holdout = holdout_.query('userid in @test_users').sort_values('userid')
    
    assert holdout_valid.set_index('userid')['timestamp'].ge(
        testset_valid
        .groupby('userid')
        ['timestamp'].max()
    ).all()

    data_description = dict(
        users = data_index['users'].name,
        items = data_index['items'].name,
        feedback = 'rating',
        n_users = len(data_index['users']),
        n_items = len(data_index['items']),
        n_ratings = training['rating'].nunique(),
        min_rating = training['rating'].min(),
        test_users = holdout_valid[data_index['users'].name].drop_duplicates().values, # NEW
        n_test_users = holdout_valid[data_index['users'].name].nunique() # NEW
    )

    return training, testset_valid, holdout_valid, testset, holdout, data_description, data_index


# In[4]:


training, testset_valid, holdout_valid, testset, holdout, data_description, data_index = full_preproccessing(data)


# ## Evaluation functions

# In[5]:


def model_evaluate(recommended_items, holdout, holdout_description, alpha=3, topn=10, dcg=False):
    itemid = holdout_description['items']
    rateid = holdout_description['feedback']
    alpha = 3 if holdout_description["n_ratings"] == 5 else 6
    n_test_users = recommended_items.shape[0]
    holdout_items = holdout[itemid].values
    assert recommended_items.shape[0] == len(holdout_items)
    
    hits_mask = recommended_items[:, :topn] == holdout_items.reshape(-1, 1)
    pos_mask = (holdout[rateid] >= alpha).values
    neg_mask = (holdout[rateid] < alpha).values
    
    # HR calculation
    #hr = np.sum(hits_mask.any(axis=1)) / n_test_users
    hr_pos = np.sum(hits_mask[pos_mask].any(axis=1)) / n_test_users
    hr_neg = np.sum(hits_mask[neg_mask].any(axis=1)) / n_test_users
    hr = hr_pos + hr_neg
    
    # MRR calculation
    hit_rank = np.where(hits_mask)[1] + 1.0
    mrr = np.sum(1 / hit_rank) / n_test_users
    pos_hit_rank = np.where(hits_mask[pos_mask])[1] + 1.0
    mrr_pos = np.sum(1 / pos_hit_rank) / n_test_users
    neg_hit_rank = np.where(hits_mask[neg_mask])[1] + 1.0
    mrr_neg = np.sum(1 / neg_hit_rank) / n_test_users
    
    # Matthews correlation
    TP = np.sum(hits_mask[pos_mask]) # + 
    FP = np.sum(hits_mask[neg_mask]) # +
    cond = (hits_mask.sum(axis = 1) == 0)
    FN = np.sum(cond[pos_mask])
    TN = np.sum(cond[neg_mask])
    N = TP+FP+TN+FN
    S = (TP+FN)/N
    P = (TP+FP)/N
    C = (TP/N - S*P) / np.sqrt(P*S*(1-P)*(1-S))
    
    # DCG calculation
    if dcg:
        pos_hit_rank = np.where(hits_mask[pos_mask])[1] + 1.0
        neg_hit_rank = np.where(hits_mask[neg_mask])[1] + 1.0
        ndcg = np.mean(1 / np.log2(pos_hit_rank+1))
        ndcl = np.mean(1 / np.log2(neg_hit_rank+1))
    
    # coverage calculation
    n_items = holdout_description['n_items']
    cov = np.unique(recommended_items).size / n_items
    if dcg:
        return hr, hr_pos, hr_neg, mrr, mrr_pos, mrr_neg, cov, C, ndcg, ndcl
    else:
        return hr, hr_pos, hr_neg, mrr, mrr_pos, mrr_neg, cov, C

def make_prediction(tf_scores, holdout, data_description, mode, context="", print_mode=True):
    if (mode and print_mode):
        print(f"for context {context} evaluation ({mode}): \n")
    tf_recs = topn_recommendations(tf_scores, 20)
    for n in [5, 10, 20]:
        hr, hr_pos, hr_neg, mrr, mrr_pos, mrr_neg, cov, C = model_evaluate(tf_recs[:, :n], holdout, data_description, topn=n)
        if (print_mode):
            print(f"HR@{n} = {hr:.4f}, MRR@{n} = {mrr:.4f}, Coverage@{n} = {cov:.4f}")
            print(f"HR_pos@{n} = {hr_pos:.4f}, HR_neg@{n} = {hr_neg:.4f}")
            print(f"MRR_pos@{n} = {mrr_pos:.4f}, MRR_neg@{n} = {mrr_neg:.4f}")
            print(f"Matthews@{n} = {C:.4f}")
            print("-------------------------------------")
        if (n == 10):
            mrr10 = mrr
            hr10 = hr
            c10 = C
    return mrr10, hr10, c10

def valid_mlrank(mlrank):
    '''
    Only allow ranks that are suitable for truncated SVD computations
    on unfolded compressed tensor (the result of ttm product in HOOI).
    '''
    #s, r1, r2, r3 = mlrank
    s, r1, r3 = mlrank
    r2 = r1
    #print(s, r1, r2, r3)
    return r1*r2 > r3 and r1*r3 > r2 and r2*r3 > r1


# # Random Model

# In[6]:


def build_random_model(trainset, trainset_description):
    itemid = trainset_description['items']
    n_items = trainset[itemid].max() + 1
    random_state = np.random.RandomState(42)
    return n_items, random_state

def random_model_scoring(params, testset, testset_description):
    n_items, random_state = params
    n_users = testset_description['n_test_users']
    scores = random_state.rand(n_users, n_items)
    return scores

def simple_model_recom_func(scores, topn=20):
    recommendations = np.apply_along_axis(topidx, 1, scores, topn)
    return recommendations

def topidx(a, topn):
    parted = np.argpartition(a, -topn)[-topn:]
    return parted[np.argsort(-a[parted])]


# # Popularity-based model

# In[7]:


def build_popularity_model(trainset, trainset_description):
    itemid = trainset_description['items']
    item_popularity = trainset[itemid].value_counts()
    return item_popularity

def popularity_model_scoring(params, testset, testset_description):
    item_popularity = params
    n_items = item_popularity.index.max() + 1
    n_users = testset_description['n_test_users']
    # fill in popularity scores for each item with indices from 0 to n_items-1
    popularity_scores = np.zeros(n_items,)
    popularity_scores[item_popularity.index] = item_popularity.values
    # same scores for each test user
    scores = np.tile(popularity_scores, n_users).reshape(n_users, n_items)
    return scores


# # PureSVD

# In[8]:


from sklearn.utils.extmath import randomized_svd
svds = randomized_svd

def matrix_from_observations(data, data_description):
    useridx = data[data_description['users']]
    itemidx = data[data_description['items']]
    values = data[data_description['feedback']]
    return csr_matrix((values, (useridx, itemidx)), shape=(useridx.values.max() + 1, data_description["n_items"]), dtype='f8')

def build_svd_model(config, data, data_description):
    source_matrix = matrix_from_observations(data, data_description)
    #print(source_matrix.shape)
    D = norm(source_matrix, axis=0)
    A = source_matrix.dot(diags(D**(config['f']-1)))

    _, _, vt = svds(A, n_components=config['rank'], random_state=42)
#     singular_values = s[::-1]
    item_factors = np.ascontiguousarray(vt[::-1, :].T)
    return item_factors

def svd_model_scoring(params, data, data_description):
    item_factors = params
    test_data = data.assign(
        userid = pd.factorize(data['userid'])[0]
    )
    test_matrix = matrix_from_observations(test_data, data_description)
    #print(test_matrix.shape, item_factors.shape)
    scores = test_matrix.dot(item_factors) @ item_factors.T
    return scores


# ## Tuning

# In[9]:


rank_grid = []
for i in range(5, 9):
    rank_grid.append(2 * 2 ** i)
    rank_grid.append(3 * 2 ** i)
    
rank_grid = np.array(rank_grid)

f_grid = np.linspace(0, 2, 21)


# In[10]:


hr_tf = {}
mrr_tf = {}
C_tf = {}
#grid = list(zip(np.meshgrid(rank_grid, f_grid)[0].flatten(), np.meshgrid(rank_grid, f_grid)[1].flatten()))
for f in tqdm(f_grid):
    svd_config = {'rank': rank_grid[-1], 'f': f}
    svd_params = build_svd_model(svd_config, training, data_description)
    for r in rank_grid:
        svd_scores = svd_model_scoring(svd_params[:, :r], testset_valid, data_description)
        downvote_seen_items(svd_scores, testset_valid, data_description)
        svd_recs = topn_recommendations(svd_scores, topn=10)
        hr, hr_pos, hr_neg, mrr, mrr_pos, mrr_neg, cov, C = model_evaluate(svd_recs, holdout_valid, data_description, alpha=3, topn=10, dcg=False)
        hr_tf[f'r={r}, f={f:.2f}'] = hr
        mrr_tf[f'r={r}, f={f:.2f}'] = mrr
        C_tf[f'r={r}, f={f:.2f}'] = C


# In[12]:


hr_sorted = sorted(hr_tf, key=hr_tf.get, reverse=True)
for i in range(1):
    print(hr_sorted[i], hr_tf[hr_sorted[i]])


# In[13]:


mrr_sorted = sorted(mrr_tf, key=mrr_tf.get, reverse=True)
for i in range(1):
    print(mrr_sorted[i], mrr_tf[mrr_sorted[i]])


# In[14]:


C_sorted = sorted(C_tf, key=C_tf.get, reverse=True)
for i in range(1):
    print(C_sorted[i], C_tf[C_sorted[i]])


# # Test metrics

# In[15]:


data_description = dict(
    users = data_index['users'].name,
    items = data_index['items'].name,
    feedback = 'rating',
    n_users = len(data_index['users']),
    n_items = len(data_index['items']),
    n_ratings = training['rating'].nunique(),
    min_rating = training['rating'].min(),
    test_users = holdout[data_index['users'].name].drop_duplicates().values,
    n_test_users = holdout[data_index['users'].name].nunique()
)


# ## Random model

# In[16]:


rnd_params = build_random_model(training, data_description)
rnd_scores = random_model_scoring(rnd_params, None, data_description)
downvote_seen_items(rnd_scores, testset, data_description)

_ = make_prediction(rnd_scores, holdout, data_description, mode="Test")


# ## Popularity-based model

# In[17]:


pop_params = build_popularity_model(training, data_description)
pop_scores = popularity_model_scoring(pop_params, None, data_description)
downvote_seen_items(pop_scores, testset, data_description)


# In[18]:


_ = make_prediction(pop_scores, holdout, data_description, mode="Test")


# ## PureSVD

# In[19]:


for_hr = sorted(hr_tf, key=hr_tf.get, reverse=True)[0]
for_mrr = sorted(mrr_tf, key=mrr_tf.get, reverse=True)[0]
for_mc = sorted(C_tf, key=C_tf.get, reverse=True)[0]

svd_config_hr = {'rank': int(for_hr.split(",")[0][2:]), 'f': float(for_hr.split(",")[1][3:])}
svd_config_mrr = {'rank': int(for_mrr.split(",")[0][2:]), 'f': float(for_mrr.split(",")[1][3:])}
svd_config_mc = {'rank': int(for_mc.split(",")[0][2:]), 'f': float(for_mc.split(",")[1][3:])}

svd_configs = [(svd_config_hr, "HR"), (svd_config_mrr, "MRR"), (svd_config_mc, "MC")]

for svd_config in svd_configs:
    print(svd_config)
    svd_params = build_svd_model(svd_config[0], training, data_description)
    svd_scores = svd_model_scoring(svd_params, testset, data_description)
    downvote_seen_items(svd_scores, testset, data_description)

    _ = make_prediction(svd_scores, holdout, data_description, mode="Test")


# # CoFFee

# In[20]:
