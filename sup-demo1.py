#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 15:03:08 2017

@author: frank
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

def loadMovieLens(path='/home/frank/Documents/machinlearning/recommendation'):
    movies = {}
    for line in open(path+'/u.item'):
        (id, title)=line.split('|')[0:2]
        movies[id] = title
    prefs = {}
    for line in open(path+'/u.data'):
        (user, movieid, rating, ts) = line.split('\t')
        prefs.setdefault(user, {})
        prefs[user][movies[movieid]] = float(rating)
    return prefs

def getReco_user_base(sim, user, prefs, n=5):
    totals = {}
    simSums = {}
    user_list= list(prefs.keys())
    user_index = user_list.index(user)
    for other in prefs:
        if other == user: continue
        other_index=user_list.index(other)        
        sim_ = sim[user_index][other_index]
        if sim_ <=0: continue
        for item in prefs[other]:
            if item not in prefs[user] or prefs[user][item]==0:
                totals.setdefault(item,0)
                totals[item] += prefs[other][item] * sim_
                simSums.setdefault(item, 0)
                simSums[item] += sim_
    rankings = [(round(total/simSums[item],1), item) for item, total \
                in totals.items()]
    rankings.sort()
    rankings.reverse()
    return rankings[0:n]

def Recommender_by_user(user, trainset, sim, prefs, n=5):
    iid = trainset.to_inner_iid(user)
    if trainset.knows_user(iid)==False:
        print('user:'+ user + ' is not in the system')
        return 1
    else:
        reco = getReco_user_base(sim, user, prefs, n)
        print(reco)
        return 0
    
import os
from surprise import Dataset
from surprise import Reader
#load file from csv, using movielens data
file_path = os.path.expanduser(
        '~/Documents/machinlearning/recommendation/u.data')
#define reader, with line_format='user item rating timestamp'
reader = Reader(line_format='user item rating timestamp', sep='\t',
                rating_scale=(1,5), skip_lines=1)
data = Dataset.load_from_file(file_path, reader=reader)
trainset = data.build_full_trainset()

#choose similarity measure and prediction algorithms, then train the data
from surprise import KNNBasic
sim_options = {'name': 'pearson'}
algo = KNNBasic(sim_options=sim_options)
algo.train(trainset)

#try recommender
sim = algo.sim
prefs = loadMovieLens()
Recommender_by_user('87', trainset, sim, prefs, 5)



