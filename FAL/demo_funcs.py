# Hadis Anahideh, Abolfazl Asudeh 2020
import pandas as pd
import numpy as np
from sklearn.metrics import mutual_info_score,confusion_matrix
import lr_inc as lr

def Demo(_Cset,_Cset_s,_Cset_y,clf=None,option=0):
    if option==0: return Demo_mutinfo(_Cset,_Cset_s,_Cset_y,clf)
    elif option==1: return Demo_err(_Cset,_Cset_s,_Cset_y,clf)
    else: return Demo_ratio(_Cset,_Cset_s,_Cset_y,clf)
        
    
def Demo_mutinfo(_Cset,_Cset_s,_Cset_y,clf):
    _Cset_yhat= clf.predict(_Cset) 
    demoErr=mutual_info_score(_Cset_s,_Cset_yhat)
    return abs(demoErr)

def Demo_err(_Cset,_Cset_s,_Cset_y,clf):
    _Cset_yhat= clf.predict(_Cset) 
    a=_Cset[(_Cset_s==0)&(_Cset_yhat==0)].shape[0]
    b=_Cset[(_Cset_s==0)&(_Cset_yhat==1)].shape[0]
    c=_Cset[(_Cset_s==1)&(_Cset_yhat==0)].shape[0]
    d=_Cset[(_Cset_s==1)&(_Cset_yhat==1)].shape[0]
    demoErr=(b/(a+b))-(d/(c+d))
    return abs(demoErr)

def Demo_ratio(_Cset,_Cset_s,_Cset_y,clf):
    _Cset_yhat_1= clf.predict(_Cset[np.where(_Cset_s==1)]) 
    tn1, fp1, fn1, tp1 = confusion_matrix(_Cset_y[np.where(_Cset_s==1)], _Cset_yhat_1, labels=[0, 1], sample_weight=None).ravel()
    _Cset_yhat_0= clf.predict(_Cset[np.where(_Cset_s==0)]) 
    tn0, fp0, fn0, tp0 = confusion_matrix(_Cset_y[np.where(_Cset_s==0)], _Cset_yhat_0, labels=[0, 1], sample_weight=None).ravel()
    G1_demo_p=(fp1+tp1)/(fp1+tp1+fp0+tp0)
    G0_test_count=len(_Cset[np.where(_Cset_s==0)])/len(_Cset_s)
    G1_test_count=len(_Cset[np.where(_Cset_s==1)])/len(_Cset_s)
    demoErr=1- min(G1_demo_p/G1_test_count,G1_test_count/G1_demo_p)
    return abs(demoErr)