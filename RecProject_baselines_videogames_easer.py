#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip3 install --no-cache-dir --upgrade git+https://github.com/evfro/polara.git@develop#egg=polara


# In[2]:

import os
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
os.environ["NUMBA_NUM_THREADS"] = "8"


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


# # Data preprocessing

# In[4]:


col_names = ['userid', 'movieid', 'rating', 'timestamp']
data = pd.read_csv("ratings_Video_Games.csv", names=col_names)


# In[5]:


def full_preproccessing(data = None):
    if (data is None):
        data = get_movielens_data("ml-10m.zip", include_time=True)
    test_timepoint = data['timestamp'].quantile(
    q=0.8, interpolation='nearest'
    )
    
    labels, levels = pd.factorize(data.movieid)
    data.movieid = labels

    labels, levels = pd.factorize(data.userid)
    data.userid = labels
    
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


# In[6]:


training, testset_valid, holdout_valid, testset, holdout, data_description, data_index = full_preproccessing(data)


# ## Utils

# In[7]:


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
    for n in [5, 10, 20]:
        tf_recs = topn_recommendations(tf_scores, n)
        hr, hr_pos, hr_neg, mrr, mrr_pos, mrr_neg, cov, C = model_evaluate(tf_recs, holdout, data_description, topn=n)
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


# # EASEr

# In[13]:


def matrix_from_observations(data, data_description):
    useridx = data[data_description['users']]
    itemidx = data[data_description['items']]
    values = data[data_description['feedback']]
    return csr_matrix((values, (useridx, itemidx)), dtype='f8')


def easer(data, data_description, lmbda=500):
    X = matrix_from_observations(data, data_description)
    G = X.T.dot(X)
    diag_indices = np.diag_indices(G.shape[0])
    G[diag_indices] += lmbda
    P = np.linalg.inv(G.A)
    B = P / (-np.diag(P))
    B[diag_indices] = 0
    
    return B

def easer_scoring(params, data, data_description):
    item_factors = params
    test_data = data.assign(
        userid = pd.factorize(data['userid'])[0]
    )
    test_matrix = matrix_from_observations(test_data, data_description)
    scores = test_matrix.dot(item_factors)
    return scores


# ## Tuning

# In[46]:


lambda_grid = np.arange(50, 1000, 50)
# lambda_grid = np.arange(5, 55, 5)


# In[47]:


hr_tf = {}
mrr_tf = {}
C_tf = {}
for lmbda in tqdm(lambda_grid):
    easer_params = easer(training, data_description, lmbda=lmbda)
    easer_scores = easer_scoring(easer_params, testset_valid, data_description)
    downvote_seen_items(easer_scores, testset_valid, data_description)
    easer_recs = topn_recommendations(easer_scores, topn=10)
    hr, hr_pos, hr_neg, mrr, mrr_pos, mrr_neg, cov, C = model_evaluate(easer_recs, holdout_valid, data_description, alpha=3, topn=10, dcg=False)
    hr_tf[lmbda] = hr
    mrr_tf[lmbda] = mrr
    C_tf[lmbda] = C


# In[48]:


hr_sorted = sorted(hr_tf, key=hr_tf.get, reverse=True)
for i in range(5):
    print(hr_sorted[i], hr_tf[hr_sorted[i]])


# In[49]:


mrr_sorted = sorted(mrr_tf, key=mrr_tf.get, reverse=True)
for i in range(5):
    print(mrr_sorted[i], mrr_tf[mrr_sorted[i]])


# In[50]:


C_sorted = sorted(C_tf, key=C_tf.get, reverse=True)
for i in range(5):
    print(C_sorted[i], C_tf[C_sorted[i]])


# # Test metrics

# In[14]:


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


# ## EASEr

# In[16]:


easer_params = easer(training, data_description, lmbda=C_sorted[i])
easer_scores = easer_scoring(easer_params, testset, data_description)
downvote_seen_items(easer_scores, testset, data_description)

make_prediction(easer_scores, holdout, data_description, mode='Test')

