import pandas as pd
import numpy as np
from sklearn.metrics import mutual_info_score,confusion_matrix
import lr_inc as lr

def Demo(_Cset,_Cset_s,_Cset_y,clf=None,option=0):
    if option==0: return Demo_mutinfo(_Cset,_Cset_s,_Cset_y,clf)
    elif option==1: return Demo_err(_Cset,_Cset_s,_Cset_y,clf)
    elif option==2: return Eqq_odds(_Cset,_Cset_s,_Cset_y,clf)
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

def Eqq_odds(_Cset,_Cset_s,_Cset_y,clf):
    _Cset_yhat= clf.predict(_Cset) 
    a=_Cset[(_Cset_s==0)&(_Cset_yhat==1)&(_Cset_y==1)].shape[0]
    b=_Cset[(_Cset_s==0)&(_Cset_y==1)].shape[0]
    c=_Cset[(_Cset_s==1)&(_Cset_yhat==1)&(_Cset_y==1)].shape[0]
    d=_Cset[(_Cset_s==1)&(_Cset_y==1)].shape[0]
    eqqOdds=(a/b)-(c/d)
    return abs(eqqOdds)

