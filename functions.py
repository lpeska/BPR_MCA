import os
import time
import numpy as np
import rank_metrics as rank
import scipy.sparse as sp
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.metrics import auc
from collections import defaultdict

def load_matrix(dataset, folder, shape, first):
    with open(os.path.join(folder, dataset+".csv"), "r") as inf:
        #inf.next()
        int_array = [line.strip("\n").split(";")[0:] for line in inf]
    intMat = np.array(int_array, dtype=np.float64)  
    return intMat
    

def cross_validation(intMat, seeds, cv=1, invert=0, fract=0.75):
   
    cv_data = defaultdict(list)
    for seed in seeds:
        num_drugs, num_targets = intMat.shape
        prng = np.random.RandomState(seed)
        if cv == 0:
            index = prng.permutation(num_drugs)
        if cv == 1:
            index = prng.permutation(intMat.size)
        step = round(index.size*fract)
        
        ii = index[0:int(step)]
        
        if cv == 0:
            test_data = np.array([[k, j] for k in ii for j in xrange(num_targets)], dtype=np.int32)
        elif cv == 1:
            test_data = np.array([[k/num_targets, k % num_targets] for k in ii], dtype=np.int32)
        x, y = test_data[:, 0], test_data[:, 1]
        test_label = intMat[x, y]
        W = np.ones(intMat.shape)
        W[x, y] = 0

        #print test_data
        #print  np.column_stack((y,x))
        #print type(test_data), type(np.column_stack((y,x)))

        if invert:
            W_T = W.T
            test_data_T = np.column_stack((y,x))
            cv_data[seed].append((W_T, test_data_T, test_label))               
        else:    
            cv_data[seed].append((W, test_data, test_label))
    return cv_data


def train(model, cv_data, intMat, drugMat, targetMat):
    aupr, auc, ndcg, ndcg_inv, results = [], [], [], [], []
    for seed in cv_data.keys():
        for W, test_data, test_label in cv_data[seed]:
            t = time.clock()
            model.fix_model(W, intMat, drugMat, targetMat, seed)
            aupr_val, auc_val, ndcg_val, ndcg_inv_val = model.evaluation(test_data, test_label)
            results = results + [("","","","")] + zip(test_data[:,0],test_data[:,1],test_label,model.scores)
            
            print(aupr_val, auc_val, ndcg_val, ndcg_inv_val , time.clock()-t)
            aupr.append(aupr_val)
            auc.append(auc_val)
            ndcg.append(ndcg_val)
            ndcg_inv.append(ndcg_inv_val)
    return np.array(aupr, dtype=np.float64), np.array(auc, dtype=np.float64), np.array(ndcg, dtype=np.float64), np.array(ndcg_inv, dtype=np.float64), results


def svd_init(M, num_factors):
    from scipy.linalg import svd
    U, s, V = svd(M, full_matrices=False)
    ii = np.argsort(s)[::-1][:num_factors]
    s1 = np.sqrt(np.diag(s[ii]))
    U0, V0 = U[:, ii].dot(s1), s1.dot(V[ii, :])
    return U0, V0.T


def mean_confidence_interval(data, confidence=0.95):
    import scipy as sp
    import scipy.stats
    a = 1.0*np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * sp.stats.t._ppf((1+confidence)/2., n-1)
    return m, h


def write_metric_vector_to_file(auc_vec, file_name):
    np.savetxt(file_name, auc_vec, fmt='%.6f')


def load_metric_vector(file_name):
    return np.loadtxt(file_name, dtype=np.float64)


       
    
def normalized_discounted_cummulative_gain(test_data,test_label, scores):
    unique_users = np.unique(test_data[:,0])
    user_array = test_data[:,0]
    ndcg = []         
    for u in unique_users:
        indices_u =  np.in1d(user_array, [u])
        labels_u = test_label[indices_u].astype(float)
        scores_u = scores[indices_u].astype(float)
        #ndcg is calculated only for the users with some positive examples
        if not all(i <= 0.001 for i in labels_u):                        
            tmp = np.c_[labels_u,scores_u]
            tmp = tmp[tmp[:,1].argsort()[::-1],:]
            ordered_labels = tmp[:,0]
            ndcg_u = rank.ndcg_at_k(ordered_labels,ordered_labels.shape[0],1)
            ndcg.append(ndcg_u)            
    return np.mean(ndcg)
        
        
def per_user_rankings(test_data,test_label, scores):
    unique_users = np.unique(test_data[:,0])
    user_array = test_data[:,0]
    ndcg = []   
    aupr_list = [] 
    auc_list = [] 
    for u in unique_users:
        indices_u =  np.in1d(user_array, [u])
        labels_u = test_label[indices_u].astype(float)
        scores_u = scores[indices_u].astype(float)
        #ndcg is calculated only for the users with some positive examples
        if not all(i <= 0.001 for i in labels_u):                        
            tmp = np.c_[labels_u,scores_u]
            tmp = tmp[tmp[:,1].argsort()[::-1],:]
            ordered_labels = tmp[:,0]
            ndcg_u = rank.ndcg_at_k(ordered_labels,ordered_labels.shape[0],1)
            ndcg.append(ndcg_u)  
            
            prec, rec, thr = precision_recall_curve(labels_u, scores_u)
            aupr_val = auc(rec, prec)
            aupr_list.append(aupr_val)
            
            fpr, tpr, thr = roc_curve(labels_u, scores_u)
            auc_val = auc(fpr, tpr)
            auc_list.append(auc_val)     
    return np.array([ndcg, aupr_list, auc_list])
        
        
                
        
        

        
        