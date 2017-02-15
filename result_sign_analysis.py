
from functions import *
import scipy.stats as st
import numpy as np
with open("results_sign.csv", "w") as resFile:
    for cv in ["1"]: #"1", "2", "3"
        resFile.write("\n")
        resFile.write("method;dataset;nDCG;AUPR;t_nDCG;p_nDCG;t_AUPR;p_AUPR\n" )
        dt = ["0750","0900","0950","0980"]#"gpcr","ic", "nr", "e" , "e"

        met1 = ["bpr_mca_sqnsw_", "bpr_mca_sq_", "bpr_mca_sq_", "bpr_mca_sqnsw_"] 
        met2 = ["True_10_0_", "False_10_2_", "False_10_5_", "False_10_5_"] 
        
        for dataset in dt: #"gpcr","ic", "nr", "e" , "e"
            resFile.write("\n")

            max_aupr = 0
            max_ndcg = 0

            v_max_aupr = np.ones(50)
            v_max_ndcg = np.ones(50)

            for idRes in range(0,4): #get maximal values for each evaluation metric throughout the evaluated methods
                resVector = np.loadtxt(met1[idRes]+dataset+"_"+met2[idRes]+".txt", delimiter=";")
                v_ndcg = resVector[:,0]
                v_aupr = resVector[:,1]
            
                avg_aupr = np.mean(v_aupr)
                if avg_aupr > max_aupr:
                    max_aupr = avg_aupr
                    v_max_aupr = v_aupr[:]
                avg_ndcg = np.mean(v_ndcg)
                if avg_ndcg > max_ndcg:
                    max_ndcg = avg_ndcg
                    v_max_ndcg = v_ndcg[:]
               
                    
            for idRes in range(0,4):  #calculate stat. sign. of other methods vs. the best one
                v2 = resVector = np.loadtxt(met1[idRes]+dataset+"_"+met2[idRes]+".txt", delimiter=";")
                cp_aupr = resVector[:,1]
                cp_ndcg = resVector[:,0]
                
                x2, y2 = st.ttest_rel(v_max_aupr, cp_aupr)
                x3, y3 = st.ttest_rel(v_max_ndcg, cp_ndcg)
                resFile.write(""+met1[idRes]+met2[idRes]+";"+dataset+";%.6f;%.6f;%.9f;%.9f;%.9f;%.9f\n" % ( np.mean(cp_ndcg), np.mean(cp_aupr), x3, y3/2.0,  x2, y2/2.0) )
                print dataset, met1[idRes],met2[idRes], np.mean(cp_aupr), np.mean(cp_ndcg),x2, y2, x3, y3
            print ""
