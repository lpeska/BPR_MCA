"""
Bayesian Personalized Ranking with Multiple Content Alignments

Original Implementation of BPR obtained from https://github.com/gamboviol/bpr
Some IO operations and CV brought from PyDTI component, https://github.com/stephenliu0423/PyDTI

Extended with Multiple Content Alignments by Ladislav Peska, peska@ksi.mff.cuni.cz
"""
from __future__ import division
import numpy as np
import time
import scipy.sparse as sp
from math import exp    

from functions import *

class BPR_MCA_sq(object):

    def __init__(self,args):
        
        self.D = args["D"]
        self.orig_learning_rate = args["learning_rate"]
        self.learning_rate = self.orig_learning_rate

        
        self.max_iters = args["max_iters"]
        self.global_regularization = args["global_regularization"]        
        self.bias_regularization = args["global_regularization"] * args["bias_regularization"]
        self.ca_regularization = args["ca_regularization"]
        self.ca_lambda = args["ca_lambda"]
        self.shape = args["shape"]
        self.fraction = args["fraction"]     
        self.learn_sim_weights = args["learn_sim_weights"]  
        
        self.simple_predict = args["simple_predict"]
        
        self.sim_matrix_names = args["sim_names"]      
   
        self.neg_item_learning_rate = 0.1
        self.k_size = 10
        self.sim_mat = {}
        self.sim_user = {}
        self.sim_lambda = {}
        j=0
        for i in self.sim_matrix_names:
            self.sim_mat[i] = load_matrix(i, "data",self.shape,args["user_indicator"][j]) 
            self.sim_user[i] = args["user_indicator"][j]
            self.sim_lambda[i] = args["init_lambda"][j]

            j = j+1
        if(len(self.sim_lambda)>0):            
            self.sim_lambda_zero = 1/len(self.sim_lambda) 
        else:
            self.sim_lambda_zero = 1
            
        #regularize matrix similarities, so the average sum of similarity vector is 1
        for i in self.sim_matrix_names:            
            self.sim_mat[i] = np.asmatrix(self.get_nearest_neighbors(self.sim_mat[i], self.k_size))                
        for i in self.sim_matrix_names:  
            self.sim_mat[i] -= np.eye(self.sim_mat[i].shape[0]) 
            uSum = (self.sim_mat[i].sum() / self.sim_mat[i].shape[0])   
            self.sim_mat[i] = (1/uSum) * self.sim_mat[i] 
        if self.learn_sim_weights == True:    
            self.filename = "bpr_mca_sq %.3f %s %s %s Dtxt" % (self.fraction, self.simple_predict,self.k_size, len(self.sim_lambda))
        else:
            self.filename = "bpr_mca_sqnsw %.3f %s %s %s Dtxt" % (self.fraction, self.simple_predict,self.k_size, len(self.sim_lambda))
        self.filename = self.filename.replace(" ", "_")
        self.filename = self.filename.replace(".", "")
        self.filename = self.filename.replace("D", ".")    
  
            
    def get_nearest_neighbors(self, S, size=5):
        m, n = S.shape
        X = np.zeros((m, n))
        for i in xrange(m):
            ii = np.argsort(np.asarray(S[i, :]).reshape(-1))[::-1][:min(size, n)]
            ii = ii[0:size] 
            if ii[0]>0:
                X[i, ii] = S[i, ii]
            
        return X   
    
    def learn_hyperparameters(self):    
        self.best_hyperpar_metric = 0
        self.best_params = {
            "max_iters":15,
            "global_regularization": 0.05,
            "ca_lambda": 0.05,
            "ca_regularization": 0.05,
        }
        cv_data = cross_validation(self.dt, [8845], 1, 0, 0.1)
        for seed in cv_data.keys():            
            for W, test_data, test_label in cv_data[seed]:
                for glR in [0.01, 0.05]:
                    if(len(self.sim_lambda)>0) & (self.learn_sim_weights==True): 
                        for caL in [0.001, 0.01, 0.05]:
                            for caR in [0.01, 0.05, 0.2]:
                                self.global_regularization = glR
                                self.ca_lambda = caL
                                self.ca_regularization = caR
                                self.learning_rate = self.orig_learning_rate
                                for i in self.sim_matrix_names:
                                    self.sim_lambda[i] = self.sim_lambda_zero

                                self.fix_model(W, self.dt, seed)
                                self.train(seed, test_data, test_label, True)
                    elif(len(self.sim_lambda)>0):
                        for caL in [0.001, 0.01, 0.05]:
                            self.global_regularization = glR
                            self.ca_lambda = caL
                            self.learning_rate = self.orig_learning_rate
                            for i in self.sim_matrix_names:
                                self.sim_lambda[i] = self.sim_lambda_zero

                            self.fix_model(W, self.dt, seed)
                            self.train(seed, test_data, test_label, True)
                    else:
                        self.global_regularization = glR
                        self.learning_rate = self.orig_learning_rate
                        self.fix_model(W, self.dt, seed)
                        self.train(seed, test_data, test_label, True)
                        
        self.max_iters = self.best_params["max_iters"]
        self.global_regularization = self.best_params["global_regularization"]
        self.ca_lambda = self.best_params["ca_lambda"] 
        self.ca_regularization = self.best_params["ca_regularization"]
        for i in self.sim_matrix_names:
            self.sim_lambda[i] = self.sim_lambda_zero
        self.learning_rate = self.orig_learning_rate
        
        print("finished learning with best hyperparameters iters: %s glR: %.3f caL: %.3f caR: %.3f"% (self.max_iters, self.global_regularization, self.ca_lambda,self.ca_regularization))

        
        
    def evaluate_hyperpar_results(self, aupr_val, auc_val, ndcg_val, iteration):
        hyperpar_metric =  ndcg_val;
        if hyperpar_metric > self.best_hyperpar_metric:
            self.best_hyperpar_metric = hyperpar_metric
            self.best_params["max_iters"] = iteration
            self.best_params["global_regularization"] = self.global_regularization
            self.best_params["ca_lambda"] = self.ca_lambda
            self.best_params["ca_regularization"] = self.ca_regularization
            
        
    def fix_model(self, W, intMat, seed):
        self.learning_rate = self.orig_learning_rate
        self.num_users, self.num_objects = intMat.shape
        
        dt = np.multiply(W,intMat)  
        self.dt = dt
        #print(self.dt.shape)
        data = sp.csr_matrix(dt)
        self.data = data
        x, y = np.where(dt > 0)
        self.train_users, self.train_objects = set(x.tolist()), set(y.tolist())
        self.dinx = np.array(list(self.train_users))
        self.tinx = np.array(list(self.train_objects)) 
        
        self.omega_fraction = round((self.num_users + self.num_objects)/2)
        self.omega_learning_rate = self.learning_rate/self.omega_fraction
        #print(self.dinx.shape, self.tinx.shape)

        np.random.seed(seed)
                                                

    def train(self, seed, test_data, test_label, hyperpar_search = False):
        """train model
        data: user-item matrix as a scipy sparse matrix
              users and items are zero-indexed
        userSim: matrix of user similarities
        itemSim: matrix of item similarities
        """

        self.init( seed)      
        
        act_loss = self.loss()
        n_samples = self.data.nnz
        #print 'initial loss {0}'.format(act_loss)
        t = time.clock()
        for it in xrange(self.max_iters):
            #print 'starting iteration {0}'.format(it)
            users, pos_items, neg_items = self._uniform_user_sampling( n_samples)
            #print("sampling_time: {0}".format(time.clock()-t))
                       
            for u,i,j in zip(users, pos_items, neg_items):
                self.update_factors(u,i,j)

            #execute bold driver learning  after each epoch  
            new_loss =  self.loss()
            if new_loss < act_loss:
                self.learning_rate = self.learning_rate * 1.1
            else:
                self.learning_rate = self.learning_rate * 0.5
            self.omega_learning_rate = self.learning_rate/self.omega_fraction     
            act_loss = new_loss
            if(hyperpar_search != True):
                print(act_loss, self.learning_rate)
                print(self.sim_lambda)
            
            #print(time.clock()-t)
            t = time.clock()
            #print(self.sim_lambda)
            if self.learn_sim_weights == True:    
                f = "train_bpr_mca_sq %.3f %s %s %s Dtxt" % (self.fraction, self.simple_predict,self.k_size, len(self.sim_lambda))
            else:
                f = "train_bpr_mca_sqnsw %.3f %s %s %s Dtxt" % (self.fraction, self.simple_predict,self.k_size, len(self.sim_lambda))
            
            f = f.replace(" ", "_")
            f = f.replace(".", "")
            f = f.replace("D", ".") 
            if(hyperpar_search == True):
                if (it % 5 == 0) & (it != 0):
                    results = self.evaluation(test_data, test_label, True)

                    self.evaluate_hyperpar_results(results[0], results[1], results[2], it)
                    with open(f, "a") as procFile:
                        procFile.write("iter:%.6f, glR: %.3f, caL: %.3f, caR: %.3f, aupr: %.5f,auc: %.5f,ndcg: %.5f\n" % (it,self.global_regularization, self.ca_lambda, self.ca_regularization,results[0], results[1], results[2]))

        
    def init(self, seed):
        
        self.num_users,self.num_items = self.data.shape
        self.user_bias = np.zeros(self.num_users)
        self.item_bias = np.zeros(self.num_items)
        
        if seed is None:
            self.user_factors = np.sqrt(1/float(self.D*self.num_users)) * np.random.normal(size=(self.num_users,self.D))
            self.item_factors = np.sqrt(1/float(self.D*self.num_items)) * np.random.normal(size=(self.num_items,self.D))
        else:
            prng = np.random.RandomState(seed)
            self.user_factors = np.sqrt(1/float(self.D*self.num_users)) * prng.normal(size=(self.num_users,self.D))
            self.item_factors = np.sqrt(1/float(self.D*self.num_items)) * prng.normal(size=(self.num_items,self.D))
        
     
        self.create_loss_samples()

    def create_loss_samples(self):
        # apply rule of thumb to decide num samples over which to compute loss
        num_loss_samples = int(10*self.num_users**0.5)
        users, pos_items, neg_items = self._uniform_user_sampling( num_loss_samples)
        self.loss_samples = zip(users, pos_items, neg_items)

    def update_factors(self,u,i,j,update_u=True,update_i=True):
        """apply SGD update"""
        update_j = True

        x = self.item_bias[i] - self.item_bias[j] \
            + np.dot(self.user_factors[u,:],self.item_factors[i,:]-self.item_factors[j,:])
            
        
        if x > 200:
            z = 0
        if x < -200:
            z = 1
        else:    
            ex = exp(-x)
            z = ex/(1.0 + ex)
         
        d_omega = {}
           

        d_u = (self.item_factors[i,:]-self.item_factors[j,:])*z - self.global_regularization*self.user_factors[u,:] 

        for k in self.sim_matrix_names:
            if self.sim_user[k] == True:
                sm = self.sim_mat[k]
                users = np.nonzero(sm[u,:])[1]
                factors = self.user_factors[users,:]                     
                sum_sim = np.sum(sm[u,users])
                dot_product = np.dot(sm[u,users], factors)#.todense()                                        
                ssu = np.asarray((sum_sim * self.user_factors[u,:])).reshape(-1)                    
                d_u = d_u + self.ca_lambda * self.sim_lambda[k] *  (np.asarray(dot_product).reshape(-1) - ssu ).T   

                if self.learn_sim_weights == True:
                    factor_diff = factors - self.user_factors[u,:]
                    user_sim_error = np.sum(np.multiply(np.square(factor_diff), sm[u,users].T))
                    d_omega[k] = - self.ca_lambda * user_sim_error                    


        d_i = self.user_factors[u,:]*z - self.global_regularization*self.item_factors[i,:] 

        for k in self.sim_matrix_names:
            if self.sim_user[k] == False:
                sm = self.sim_mat[k]
                items = np.nonzero(sm[i,:])[1]
                factors = self.item_factors[items,:] 
                dot_product = np.dot(sm[i,items], factors)#.todense()
                sum_sim = np.sum(sm[i,items])

                ssu = np.asarray((sum_sim * self.item_factors[i,:])).reshape(-1)                    
                d_i = d_i + self.ca_lambda * self.sim_lambda[k] *  (np.asarray(dot_product).reshape(-1) - ssu ).T                                          

                if self.learn_sim_weights == True:
                    factor_diff = factors - self.item_factors[i,:]
                    item_sim_error = np.sum(np.multiply(np.square(factor_diff), sm[i,items].T))
                    d_omega[k] = - self.ca_lambda * item_sim_error    

        d_j = -self.user_factors[u,:]*z - self.global_regularization*self.item_factors[j,:]                        

        for k in self.sim_matrix_names:
            if self.sim_user[k] == False:
                sm = self.sim_mat[k]
                items = np.nonzero(sm[j,:])[1]
                factors = self.item_factors[items,:] 
                dot_product = np.dot(sm[j,items], factors)#.todense()
                sum_sim = np.sum(sm[j,items])

                ssu = np.asarray((sum_sim * self.item_factors[j,:])).reshape(-1)                    
                d_j = d_j + self.ca_lambda * self.sim_lambda[k] *  (np.asarray(dot_product).reshape(-1) - ssu ).T                                          

                if self.learn_sim_weights == True:
                    factor_diff = factors - self.item_factors[j,:]
                    item_sim_error = np.sum(np.multiply(np.square(factor_diff), sm[j,items].T))
                    d_omega[k] += - self.ca_lambda * item_sim_error   

        self.user_factors[u,:] += self.learning_rate * np.asarray(d_u).reshape(-1)
        self.item_factors[i,:] += self.learning_rate * np.asarray(d_i).reshape(-1)
        self.item_factors[j,:] += self.neg_item_learning_rate * self.learning_rate * np.asarray(d_j).reshape(-1)

        #update similarity lambdas 
        if self.learn_sim_weights == True:
            sum_lambda = 0  
            for k in self.sim_matrix_names:
                if self.sim_user[k] == True:
                    d_omega[k] = d_omega[k] - (self.ca_regularization *  self.sim_lambda[k]) +  (self.ca_regularization * self.sim_lambda_zero)
                else:
                    d_omega[k] = d_omega[k] - (2* self.ca_regularization *  self.sim_lambda[k]) + (2* self.ca_regularization * self.sim_lambda_zero)

                self.sim_lambda[k] +=  self.omega_learning_rate * d_omega[k]
                sum_lambda += abs(self.sim_lambda[k])

            #print(sum_lambda,self.sim_lambda) 

            for k in self.sim_matrix_names:
                self.sim_lambda[k] = self.sim_lambda[k]/sum_lambda

            
    def _uniform_user_sampling(self, n_samples):
        """
          Creates `n_samples` random samples from training data for performing Stochastic
          Gradient Descent. We start by uniformly sampling users, 
          and then sample a positive and a negative item for each 
          user sample.
        """
        #print("Generating %s random training samples\n" % str(n_samples))
        
        sgd_users = np.random.choice(list(self.train_users),size=n_samples)
        sgd_ni = np.random.choice(list(self.train_objects),size=(n_samples*2)) 
        i = 0
        sgd_pos_items, sgd_neg_items = [], []
        for sgd_user in sgd_users:
            pos_item = np.random.choice(self.data[sgd_user].indices)
            
            neg_item = sgd_ni[i]
            while neg_item in self.data[sgd_user].indices:
                i = i+1
                neg_item = sgd_ni[i]
                
            sgd_pos_items.append(pos_item)
            sgd_neg_items.append(neg_item)
            i = i+1

        return sgd_users, sgd_pos_items, sgd_neg_items        

    def loss(self):
        ranking_loss = 0;
        for u,i,j in self.loss_samples:
            x = self.predict(u,j) - self.predict(u,i)
            
            if x > 200:
                rl = 0
            if x < -200:
                rl = 1
            else:    
                ex = exp(-x)
                rl = 1.0/(1.0+ex)
            ranking_loss += rl
                
        complexity = self.complexity()
        #print 'complexity = {0}'.format(complexity)   
        return ranking_loss + complexity
    
    
    def complexity(self):
        complexity = 0
        for u,i,j in self.loss_samples:
            complexity += self.global_regularization * np.dot(self.user_factors[u],self.user_factors[u])
            complexity += self.global_regularization * np.dot(self.item_factors[i],self.item_factors[i])
            complexity += self.global_regularization * np.dot(self.item_factors[j],self.item_factors[j])
            d=0
            for k in self.sim_matrix_names:
                if self.sim_user[k] == True:
                    sm = self.sim_mat[k]
                    users = np.nonzero(sm[u,:])[1]
                    factors = self.user_factors[users,:] 

                    
                    factor_diff = factors - self.user_factors[u,:]      
                    user_sim_error = np.sum(np.multiply(np.square(factor_diff), sm[u,users].T))

                    d = d + abs(self.sim_lambda[k]) * user_sim_error

                else:
                    sm = self.sim_mat[k]
                    items_i = np.nonzero(sm[i,:])[1]
                    items_j = np.nonzero(sm[j,:])[1]
                    factors_i = self.item_factors[items_i,:] 
                    factors_j = self.item_factors[items_j,:] 
                    
                    factor_diff_i = factors_i - self.item_factors[i,:]
                    factor_diff_j = factors_j - self.item_factors[j,:]
                    item_sim_error_i = np.sum(np.multiply(np.square(factor_diff_i), sm[i,items_i].T))
                    item_sim_error_j = np.sum(np.multiply(np.square(factor_diff_j), sm[j,items_j].T))

                    d = d + abs(self.sim_lambda[k]) * (item_sim_error_i + item_sim_error_j )
  
                complexity += self.ca_regularization * self.sim_lambda[k]**2    
            complexity += self.ca_lambda * d
           
            complexity += self.bias_regularization * self.user_bias[u]**2
            complexity += self.bias_regularization * self.item_bias[i]**2
            complexity += self.bias_regularization * self.item_bias[j]**2
            
  
        return complexity

    def predict(self,u,i):
        
        #predict only from learned factors, no content smoothing
        if self.simple_predict:
            return self.item_bias[i] + self.user_bias[u] + np.dot(self.user_factors[u],self.item_factors[i])
        #perform content smoothing for new users
        elif (u not in self.train_users):   
            sum_sim = 0   
            vector = np.zeros((1,self.D))
            for k in self.sim_matrix_names:
                if self.sim_user[k] == True:
                    sm = self.sim_mat[k]
                    users = np.nonzero(sm[u,:])[1]
                    factors = self.user_factors[users,:] 
                    dot_product = np.dot(sm[u,users], factors)
                    sum_sim += self.sim_lambda[k] * np.sum(sm[u,users])                
                    vector += self.sim_lambda[k] * dot_product 
                         
            alignmentVectorU = vector/sum_sim 
            return self.item_bias[i] + np.mean(self.user_bias) + np.sum(np.array(alignmentVectorU[:]).flatten() * np.array(self.item_factors[i,:]).flatten()) 

        else:
            return self.item_bias[i] + self.user_bias[u] + np.dot(self.user_factors[u],self.item_factors[i])
    

    def evaluation(self, test_data, test_label, hyperpar_search = False):
        scores = []

        if self.D > 0:
            for d, t in test_data:
                score = self.predict(d,t)          
                if score > 200:
                    scores.append(1)
                elif score < -200:
                    scores.append(0)
                else:  
                    sc = np.exp(score)
                    scores.append(sc/(1+sc))
                        
        
        x, y = test_data[:, 0], test_data[:, 1]

        test_data = np.column_stack((x,y))
        test_label = np.array(test_label).T
        #vals = [ndcg, aupr, auc]
        vals = per_user_rankings(test_data, test_label, np.array(scores))
        if hyperpar_search == False:
            with open(self.filename, "a") as procFile:
                procFile.writelines(["%s;%s;%s \n" % (item[0], item[1], item[2])  for item in vals.T])

        return np.mean(vals[1,:]), np.mean(vals[2,:]), np.mean(vals[0,:])
    
    

        
    
    def __str__(self):
        return "Model: BPR_MCA, factors:%s, learningRate:%s,  max_iters:%s, global_reg:%s, bias_reg:%s, ca_reg:%s, simple_predict:%s" % (self.D, self.learning_rate, self.max_iters,  self.global_regularization, self.bias_regularization, self.ca_regularization,  self.simple_predict)
    


if __name__ == "__main__":  

    import os
    import sys
    import time
    import getopt
    import cv_eval
    import numpy as np
    import pandas as pd
    
    #run BPR_MCA5
    """
    args = {
        "D": 20,
        "learning_rate": 0.1,
        "max_iters":31,
        "global_regularization": 0.05,
        "bias_regularization": 1,
        "ca_lambda": 0.05,
        "ca_regularization": 0.05,
        "simple_predict": False,
        "learn_sim_weights": True,
        "sim_names": ["sim_userML1M", "sim_userUSPost", "sim_itemML1M", "sim_itemIMDB", "sim_itemDBT"],
        "user_indicator": [True, True, False, False, False],
        "init_lambda": [0.2, 0.2, 0.2, 0.2, 0.2]

    }
    """
    #run BPR_MCA5uniform
    args = {
        "D": 20,
        "learning_rate": 0.1,
        "max_iters":31,
        "global_regularization": 0.05,
        "bias_regularization": 1,
        "ca_lambda": 0.05,
        "ca_regularization": 0.05,
        "simple_predict": False,
        "learn_sim_weights": False,
        "sim_names": ["sim_userML1M", "sim_userUSPost", "sim_itemML1M", "sim_itemIMDB", "sim_itemDBT"],
        "user_indicator": [True, True, False, False, False],
        "init_lambda": [0.2, 0.2, 0.2, 0.2, 0.2]

    }
    
    #run standard BPR
    """
    args = {
        "D": 20,
        "learning_rate": 0.1,
        "max_iters":31,
        "global_regularization": 0.05,
        "bias_regularization": 1,
        "ca_lambda": 0,
        "ca_regularization": 0,
        "simple_predict": True,
        "learn_sim_weights": False,
        "sim_names": [ ],#["sim_userML1M", "sim_userUSPost", "sim_itemML1M", "sim_itemIMDB", "sim_itemDBT"],#
        "user_indicator": [True],#[ False],#
        "init_lambda": [1]

    }    
    """
    #run BPR_MCA2
    """
    args = {
        "D": 20,
        "learning_rate": 0.1,
        "max_iters":31,
        "global_regularization": 0.05,
        "bias_regularization": 1,
        "ca_lambda": 0.05,
        "ca_regularization": 0.05,
        "simple_predict": False,
        "learn_sim_weights": True,
        "sim_names": ["sim_userML1M",  "sim_itemML1M"],#[ ],#
        "user_indicator": [True, False],#[ False],#
        "init_lambda": [0.5, 0.5]

    }
    """

    #load ML1M interaction data
    dt = pd.read_csv("data/ratings.dat", sep=';', header=None)
    data = np.asmatrix(dt)
    
    dt_id = pd.read_csv("data/id_map.csv", sep=';')

    id_map = dict(zip(dt_id.mid.tolist(), dt_id.sid.tolist()))
    rows = [i[0,0]-1 for i in data[:,0]]

    cols = [id_map[i[0,0]]-1 for i in data[:,1]]
    vals = [i[0,0] for i in data[:,2]]    
    flat_vals = [(0 if i < 3 else 1) for i in vals]

    max_oid = 3882
    max_uid = 6039
    nr = []
    nc = []
    nd = []
    for m in range(0,len(rows)):        
        if (rows[m] <= max_uid) & (cols[m] <= max_oid) & (flat_vals[m] > 0):  
            nr.append(rows[m])
            nc.append(cols[m])
            nd.append(flat_vals[m])

    coo_mat = sp.coo_matrix((nd,(nr,nc)))

    sm = coo_mat.tocsr()
    sma = sm.toarray()

    #define p75, p90, p95, p98 evaluation scenarios
    #fraction = 0.75
    #fraction = 0.9
    #fraction = 0.95
    fraction = 0.98
    

    #perform 5x Monte Carlo cross-validation
    cv_data = cross_validation(sma, [35, 2085, 1737, 8854, 124], 1, 0, fraction)

    args["shape"] = sma.shape
    args["fraction"] = fraction

    for seed in cv_data.keys():
        for W, test_data, test_label in cv_data[seed]:
            model = BPR_MCA_sq(args)
                        
            model.fix_model(W, sma, 50)
            
            model.learn_hyperparameters()
            
            model.train(50, test_data, test_label)
            aupr_val, auc_val, ndcg_val = model.evaluation(test_data, test_label)

            print(aupr_val, auc_val, ndcg_val)


