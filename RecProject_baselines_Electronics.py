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
    q=0.8, interpolation='nearest'
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
    _, _, vt = svds(A, k=config['rank'], return_singular_vectors='vh')
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
grid = list(zip(np.meshgrid(rank_grid, f_grid)[0].flatten(), np.meshgrid(rank_grid, f_grid)[1].flatten()))
for params in grid:
    r, f = params
    svd_config = {'rank': int(r), 'f': f}
    svd_params = build_svd_model(svd_config, training, data_description)
    svd_scores = svd_model_scoring(svd_params, testset_valid, data_description)
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


from IPython.utils import io
from scipy.special import softmax

def tf_model_build(config, data, data_description, testset, holdout, attention_matrix=np.array([])):
    userid = data_description["users"]
    itemid = data_description["items"]
    feedback = data_description["feedback"]

    idx = data[[userid, itemid, feedback]].values
    idx[:, -1] = idx[:, -1] - data_description['min_rating'] # works only for integer ratings!
    val = np.ones(idx.shape[0], dtype='f8')
    
    n_users = data_description["n_users"]
    n_items = data_description["n_items"]
    n_ratings = data_description["n_ratings"]
    shape = (n_users, n_items, n_ratings)
    core_shape = config['mlrank']
    num_iters = config["num_iters"]
    
    if (attention_matrix.shape[0] == 0):
        attention_matrix = form_attention_matrix(
            data_description['n_ratings'],
            **config['params'],
            format = 'csr'
        )
        
    attention_matrix = np.array(attention_matrix)

    item_popularity = (
        data[itemid]
        .value_counts(sort=False)
        .reindex(range(n_items))
        .fillna(1)
        .values
    )
    scaling_weights = get_scaling_weights(item_popularity, scaling=config["scaling"])

    with io.capture_output() as captured:
        u0, u1, u2 = sa_hooi(
            idx, val, shape, config["mlrank"],
            attention_matrix = attention_matrix,
            scaling_weights = scaling_weights,
            testset = testset,
            holdout = holdout,
            data_description = data_description,
            max_iters = config["num_iters"],
            parallel_ttm = True,
            randomized = config["randomized"],
            growth_tol = config["growth_tol"],
            seed = config["seed"],
            iter_callback = None,
        )
    
    return u0, u1, u2, attention_matrix    
    
config = {
    "scaling": 1,
    "mlrank": (30, 30, data_description['n_ratings']),
    "n_ratings": data_description['n_ratings'],
    "num_iters": 5,
    "params": None,
    "randomized": True,
    "growth_tol": 1e-4,
    "seed": 42
}

def tf_scoring(params, data, data_description, context=["3+4+5"]):
    user_factors, item_factors, feedback_factors, attention_matrix = params
    userid = data_description["users"]
    itemid = data_description["items"]
    feedback = data_description["feedback"]

    data = data.sort_values(userid) 
    data_new = data.assign(
        userid = pd.factorize(data['userid'])[0]
    ) # NEW
    useridx = data_new[userid]
    itemidx = data_new[itemid].values
    ratings = data_new[feedback].values
    ratings = ratings - data_description['min_rating'] # NEW
    
    n_users = useridx.nunique()
    n_items = data_description['n_items']
    n_ratings = data_description['n_ratings']
    
    inv_attention = solve_triangular(attention_matrix, np.eye(n_ratings), lower=True)
    
    tensor_outer = tensor_outer_at('cpu')
    #matrix_softmax = softmax(inv_attention.T @ feedback_factors)
    matrix_softmax = inv_attention.T @ feedback_factors
    #
    if (n_ratings == 10):
        coef = 2
    else:
        coef = 1
        
    if (context == "5"): # make softmax 
        inv_aT_feedback = matrix_softmax[(-1 * coef) , :]
    elif (context == "4+5"):
        inv_aT_feedback = np.sum(matrix_softmax[(-2 * coef):, :], axis=0)
    elif (context == "3+4+5"):
        inv_aT_feedback = np.sum(matrix_softmax[(-3 * coef):, :], axis=0)
    #elif (context == "2+3+4+5"):
    #    inv_aT_feedback = np.sum(matrix_softmax[-4:, :], axis=0)
    elif (context == "3+4+5-2-1"):
        inv_aT_feedback = np.sum(matrix_softmax[(-3 * coef):, :], axis=0) - np.sum(matrix_softmax[:(2 * coef), :], axis=0)
        
    scores = tensor_outer(
        1.0,
        item_factors,
        attention_matrix @ feedback_factors,
        itemidx,
        ratings
    )
    scores = np.add.reduceat(scores, np.r_[0, np.where(np.diff(useridx))[0]+1]) # sort by users
    scores = np.tensordot(
        scores,
        inv_aT_feedback,
        axes=(2, 0)
    ).dot(item_factors.T)

    return scores


# In[21]:


from tqdm import tqdm 
from polara.evaluation.pipelines import random_grid

def full_pipeline(config, training, data_description, testset_valid, holdout_valid, testset, holdout, attention_matrix, factor=None):

    config["mlrank"] = (64, 64, data_description["n_ratings"])
    print("Starting pipeline...")
    print("Training with different context in progress...")
    print("------------------------------------------------------")

    for context in ["5", "4+5", "3+4+5", "3+4+5-2-1"]:
        tf_params = tf_model_build(config, training, data_description, testset_valid, holdout_valid, attention_matrix=attention_matrix)
        seen_data = testset_valid
        tf_scores = tf_scoring(tf_params, seen_data, data_description, context)
        downvote_seen_items(tf_scores, seen_data, data_description)
        cur_mrr, cur_hr, cur_C = make_prediction(tf_scores, holdout_valid, data_description, "Validation", context)
        print("------------------------------------------------------")

    print(f"Tuning model for all contexts...\n")

    rank_grid = []
    for i in range(5, 8):
        rank_grid.append(2 * 2 ** i)
        rank_grid.append(3 * 2 ** i)
    
    rank_grid = np.array(rank_grid)
    tf_hyper = {
    'scaling': [factor] if factor else [0.6], #np.linspace(0, 2, 21),
    'r1': rank_grid, #np.arange(100, 220, 25),
    #'r2': np.arange(50, 801, 25),
    'r3': range(2, 6, 1)#range(2, 11, 2), # change
    }

    grid, param_names = random_grid(tf_hyper, n=0)
    tf_grid = [tuple(mlrank) for mlrank in grid if valid_mlrank(mlrank)]

    hr_tf = {}
    hr_pos_tf = {}
    hr_neg_tf = {}
    mrr_tf = {}
    mrr_pos_tf = {}
    mrr_neg_tf = {}
    cov_tf = {}
    C_tf = {}
    
    seen_data = testset_valid
    
    for mlrank in tf_grid:
        with io.capture_output() as captured:
            r1, r3 = mlrank[1:]
            cur_mlrank = tuple((r1, r1, r3))
            config['mlrank'] = cur_mlrank
            config['scaling'] = mlrank[0]
            tf_params = tf_model_build(config, training, data_description, testset_valid, holdout_valid, attention_matrix=attention_matrix)
            for context in ["5", "4+5", "3+4+5", "3+4+5-2-1"]:
                tf_scores = tf_scoring(tf_params, seen_data, data_description, context)
                downvote_seen_items(tf_scores, seen_data, data_description)
                tf_recs = topn_recommendations(tf_scores, topn=10)
                
                hr, hr_pos, hr_neg, mrr, mrr_pos, mrr_neg, cov, C = model_evaluate(tf_recs, holdout_valid, data_description, topn=10)
                hr_tf[(context, cur_mlrank, mlrank[0])] = hr
                hr_pos_tf[(context, cur_mlrank, mlrank[0])] = hr_pos
                hr_neg_tf[(context, cur_mlrank, mlrank[0])] = hr_neg
                mrr_tf[(context, cur_mlrank, mlrank[0])] = mrr
                mrr_pos_tf[(context, cur_mlrank, mlrank[0])] = mrr_pos
                mrr_neg_tf[(context, cur_mlrank, mlrank[0])] = mrr_neg
                cov_tf[(context, cur_mlrank, mlrank[0])] = cov
                C_tf[(context, cur_mlrank, mlrank[0])] = C
                

    print(f'Best HR={pd.Series(hr_tf).max():.4f} achieved with context {pd.Series(hr_tf).idxmax()[0]} and mlrank = {pd.Series(hr_tf).idxmax()[1]} and scale factor = {pd.Series(hr_tf).idxmax()[2]}')
    print(f'Best HR_pos={pd.Series(hr_pos_tf).max():.4f} achieved with context {pd.Series(hr_pos_tf).idxmax()[0]} and mlrank = {pd.Series(hr_pos_tf).idxmax()[1]} and scale factor = {pd.Series(hr_pos_tf).idxmax()[2]}')
    print(f'Best HR_neg={pd.Series(hr_neg_tf).min():.4f} achieved with context {pd.Series(hr_neg_tf).idxmin()[0]} and mlrank = {pd.Series(hr_neg_tf).idxmin()[1]} and scale factor = {pd.Series(hr_neg_tf).idxmin()[2]}')
    
    print(f'Best MRR={pd.Series(mrr_tf).max():.4f} achieved with context {pd.Series(mrr_tf).idxmax()[0]} and mlrank = {pd.Series(mrr_tf).idxmax()[1]} and scale factor = {pd.Series(mrr_tf).idxmax()[2]}')
    print(f'Best MRR_pos={pd.Series(mrr_pos_tf).max():.4f} achieved with context {pd.Series(mrr_pos_tf).idxmax()[0]} and mlrank = {pd.Series(mrr_pos_tf).idxmax()[1]} and scale factor = {pd.Series(mrr_pos_tf).idxmax()[2]}')
    print(f'Best MRR_neg={pd.Series(mrr_neg_tf).min():.4f} achieved with context {pd.Series(mrr_neg_tf).idxmin()[0]} and mlrank = {pd.Series(mrr_neg_tf).idxmin()[1]} and scale factor = {pd.Series(mrr_neg_tf).idxmin()[2]}')
    
    print(f'Best Matthews={pd.Series(C_tf).max():.4f} achieved with context {pd.Series(C_tf).idxmax()[0]} and mlrank = {pd.Series(C_tf).idxmax()[1]} and scale factor = {pd.Series(C_tf).idxmax()[2]}')
                          
    print(f'COV={pd.Series(cov_tf)[pd.Series(C_tf).idxmax()]:.4f} (based on best Matthews value)')
    print("---------------------------------------------------------")
    print("Evaluation of the best model on test holdout in progress...\n")
    
    print("Best by MRR@10:\n")
    config["mlrank"] = pd.Series(mrr_pos_tf).idxmax()[1]
    tf_params = tf_model_build(config, training, data_description, testset, holdout, attention_matrix=attention_matrix)

    seen_data = testset
    tf_scores = tf_scoring(tf_params, seen_data, data_description, pd.Series(mrr_pos_tf).idxmax()[0])
    downvote_seen_items(tf_scores, seen_data, data_description)
    cur_mrr, cur_hr, cur_C = make_prediction(tf_scores, holdout, data_description, "Test", pd.Series(mrr_pos_tf).idxmax()[0])
    
    print("---------------------------------------------------------")
    
    print("Best by HR@10:\n")
    config["mlrank"] = pd.Series(hr_pos_tf).idxmax()[1]
    tf_params = tf_model_build(config, training, data_description, testset, holdout, attention_matrix=attention_matrix)

    seen_data = testset
    tf_scores = tf_scoring(tf_params, seen_data, data_description, pd.Series(hr_pos_tf).idxmax()[0])
    downvote_seen_items(tf_scores, seen_data, data_description)
    cur_mrr, cur_hr, cur_C = make_prediction(tf_scores, holdout, data_description, "Test", pd.Series(hr_pos_tf).idxmax()[0])
    
    print("---------------------------------------------------------")
    
    print("Best by Matthews@10:\n")
    config["mlrank"] = pd.Series(C_tf).idxmax()[1]
    tf_params = tf_model_build(config, training, data_description, testset, holdout, attention_matrix=attention_matrix)

    seen_data = testset
    tf_scores = tf_scoring(tf_params, seen_data, data_description, pd.Series(C_tf).idxmax()[0])
    downvote_seen_items(tf_scores, seen_data, data_description)
    cur_mrr, cur_hr, cur_C = make_prediction(tf_scores, holdout, data_description, "Test", pd.Series(C_tf).idxmax()[0])
    print("Pipeline ended.")


# In[ ]:


attention_matrix = np.eye(5)
full_pipeline(config, training, data_description, testset_valid, holdout_valid, testset, holdout, attention_matrix=attention_matrix, factor = float(for_mc.split(",")[1][3:]))


# ## Attention matrix

# In[ ]:


def sigmoid_func(x):
    return 1.0 / (1 + np.exp(-x))

def arctan(x):
    return 0.5 * np.arctan(x) + 0.5

def sq3(x):
    return 0.5 * np.cbrt(x) + 0.5

def get_similarity_matrix(mode, n_ratings = 10):
    matrix = np.zeros((n_ratings, n_ratings))
    if (mode == "sigmoid"):
        x_space = np.linspace(-6, 6, n_ratings)
        for i in range(n_ratings):
            for j in range(i, n_ratings, 1):
                matrix[i, j] = 1.0 - np.abs(sigmoid_func(x_space[i]) - sigmoid_func(x_space[j]))
                matrix[j, i] = matrix[i, j]
                
    elif (mode == "linear"):
        x_space = np.linspace(0, 1, n_ratings)
        for i in range(n_ratings):
            for j in range(i, n_ratings, 1):
                matrix[i, j] = 1.0 - np.abs(x_space[i] - x_space[j])
                matrix[j, i] = matrix[i, j]
                
    elif (mode == "arctan"):
        x_space = np.linspace(-np.pi / 2.0, np.pi / 2.0, n_ratings)
        for i in range(n_ratings):
            for j in range(i, n_ratings, 1):
                matrix[i, j] = 1.0 - np.abs(arctan(x_space[i]) - arctan(x_space[j]))
                matrix[j, i] = matrix[i, j]
                
    elif (mode == "sq3"):
        x_space = np.linspace(-1, 1, n_ratings)
        for i in range(n_ratings):
            for j in range(i, n_ratings, 1):
                matrix[i, j] = 1.0 - np.abs(sq3(x_space[i]) - sq3(x_space[j]))
                matrix[j, i] = matrix[i, j]
                
    return matrix


# In[ ]:


matrices = []
config["params"] = {}
modes = [ "linear", "sq3", "sigmoid", "arctan"]

for mode in modes:
    print(f"FOR matrix - {mode}")
    similarity_matrix = get_similarity_matrix(mode, data_description["n_ratings"])
    attention_matrix = sqrtm(similarity_matrix).real
    full_pipeline(config, training, data_description, testset_valid, holdout_valid, testset, holdout, attention_matrix=attention_matrix, factor = float(for_mc.split(",")[1][3:]))
    print("_____________________________________________________")


# In[ ]:




