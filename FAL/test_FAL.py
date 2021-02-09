import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from time import time
from time import time
import FAL_sklearn as flsk #FAL
import FAK_COVXY ad fcov #FBC
import FAL_sklearn_Nested as flsktopk #FAL-Nested
import FAL_COVXY_Nested as fcovk #FBC-Nested
import FAL_sklearn_Nested_Append as flskdis #FAL Nested-Append
import FAL_COVXY_Nested_Append as fcovdis #FBC Nested-Append
import prep as pr
import pickle
import AL
import RL

##Compas
path='RecidivismData_Normalized.csv'
response='two_year_recid'
sensitive='race'
atr=['MarriageStatus','age','juv_fel_count', 'juv_misd_count', 'juv_other_count','priors_count', 'days_b_screening_arrest', 'c_days_from_compas','c_charge_degree']

# ##Adult
# path=['IBM_adult_A.txt','IBM_adult_X.txt','IBM_adult_Y.txt']
# sensitive= 'Gender'#male is 1
# response = 'Income'
# atr=[]


rnd=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]
alp=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9] #accuracy weight
# alp=[1] #for adaptive alpha
# k=[2,4,8,16,32,64,128,256,512] #for Nested and Nested-Append


demo_option=0#0 is mutual information, 1 is for absolute difference, 2 is for proportional ratio
data_option=3#0 for adult, 1 for credit and 3 for compas
#b is the sampling budget
b=200
flag_demo=False#True for Equalized Odds

#loop over different alpha values
for t in range(len(alp)):
    #set the alpha value for accuracy fairness trade-off
    alpha=np.linspace(1,1,11)*alp[t]#for user-defined alpha
    #alpha=np.linspace(1,0,11)*alp[t]#for adaptive alpha; the range can be changed as needed the default is (1,0)
    filename='filename'
    #loop over different random splits
    for r in range(0, len(rnd)):
        #choose one of the following methods
        #demo,_Xl,_Xl_s,_yl,clf,overall_score,f1score = flsk.FAL(path,response,sensitive,atr,demo_option,r,b,alpha,rnd,data_option,flag_demo)
        #demo,_Xl,_Xl_s,_yl,theta,clf,overall_score = fcov.FAL(path,response,sensitive,atr,demo_option,r,b,alpha,rnd,data_option,flag_demo)
        demo_test,_Xl,_Xl_s,_yl,clf,overall_score,f1score = flsktopk.FAL(path,response,sensitive,atr,demo_option,r,b,alpha,rnd,data_option,flag_demo,k[t])
        #demo,_Xl,_Xl_s,_yl,theta,clf,overall_score,time = fcovk.FAL(path,response,sensitive,atr,demo_option,r,b,alpha,rnd,data_option,flag_demo,k[t])
        #demo_test,_Xl,_Xl_s,_yl,clf,overall_score,f1score = flskdis.FAL(path,response,sensitive,atr,demo_option,r,b,alpha,rnd,data_option,flag_demo,k[t])
        #demo,_Xl,_Xl_s,_yl,theta,clf,overall_score = fcovdis.FAL(path,response,sensitive,atr,demo_option,r,b,alpha,rnd,data_option,flag_demo,k[t])
        #demo,_Xl,_Xl_s,_yl,clf,overall_score = AL.AL(path,response,sensitive,atr,demo_option,r,b,alpha,rnd,data_option)    
        #demo,_Xl,clf,overall_score = RL.RL(path,response,sensitive,atr,demo_option,r,b,alpha,rnd,data_option)    
        
        ##save the output
        with open(filename+str(rnd[r])+'.pkl', 'wb') as f:
            pickle.dump([demo,_Xl,_Xl_s,_yl,clf,overall_score], f)