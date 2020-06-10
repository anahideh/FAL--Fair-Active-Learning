# Hadis Anahideh, Abolfazl Asudeh 2020
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from time import time
from time import time
import FAL_sklearn as flsk #FAL
import FAL_COVXY as fcov #efficient FAL
import prep as pr
import pickle
import AL
import RL

##Compas
path='RecidivismData_Normalized.csv'
response='two_year_recid'
sensitive='race'
atr=['MarriageStatus','age','juv_fel_count', 'juv_misd_count', 'juv_other_count','priors_count', 'days_b_screening_arrest', 'c_days_from_compas','c_charge_degree']

##Adult
path=['IBM_adult_A.txt','IBM_adult_X.txt','IBM_adult_Y.txt']
sensitive= 'Gender'#male is 1
response = 'Income'
atr=[]


##Credit
path='german_credit_dataset.csv'
response='credit.rating'
sensitive='marital.status'
atr=['account.balance', 'credit.duration.months',
       'previous.credit.payment.status', 'credit.purpose', 'credit.amount',
       'savings', 'employment.duration', 'installment.rate', 
       'guarantor', 'residence.duration', 'current.assets', 'age',
       'other.credits', 'apartment.type', 'bank.credits', 'occupation',
       'dependents', 'telephone', 'foreign.worker']

rnd=[109212566,223920065,90219211,18592594,133306463,47827655,80816501,9527535,46882677,136690857]
alp=[0.6] #accuracy weight
alp=[1] #for adaptive alpha

demo_option=0#0 is mutual information, 1 is for absolute difference, 2 is for proportional ratio
data_option=3#0 for adult, 1 for credit and 3 for compas
#b is the sampling budget
b=200
#loop over different alpha values
for t in range(len(alp)):
    #set the alpha value for accuracy fairness trade-off
    alpha=np.linspace(1,1,11)*alp[t]#for user-defined alpha
    #alpha=np.linspace(1,0,11)*alp[t]#for adaptive alpha; the range can be changed as needed the default is (1,0)
    filename='filename'
    #loop over different random splits
    for r in range(0, len(rnd)):
        #choose one of the following methods
        demo,_Xl,_Xl_s,_yl,clf,overall_score = flsk.FAL(path,response,sensitive,atr,demo_option,r,b,alpha,rnd,data_option)
        #demo,_Xl,_Xl_s,_yl,clf,overall_score = fcov.FAL(path,response,sensitive,atr,demo_option,r,b,alpha,rnd,data_option)
        #demo,_Xl,_Xl_s,_yl,clf,overall_score = AL.AL(path,response,sensitive,atr,demo_option,r,b,alpha,rnd,data_option)    
        #demo,_Xl,clf,overall_score = RL.RL(path,response,sensitive,atr,demo_option,r,b,alpha,rnd,data_option)    
        
        ##save the output
        with open(filename+str(rnd[r])+'.pkl', 'wb') as f:
            pickle.dump([demo,_Xl,_Xl_s,_yl,clf,overall_score], f)