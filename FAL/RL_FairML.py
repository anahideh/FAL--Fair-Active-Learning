import numpy as np
import math
from time import time
from sklearn.linear_model import LogisticRegression
import prep as pr
import demo_funcs as dm
import lr_inc as lr
from sklego.linear_model import DemographicParityClassifier

def RL(path,response,sensitive,atr,demo_option,r,b,alpha,rnd,data_option,flag_demo):
    demo_test=[] 
    f1score=[] 
    _Xl,_Xl_s,n,m,_yl,_Xu,_Xu_s,_yu,_Xt,_Xt_s,_yt,_Cset,_Cset_s,_Cset_y=pr.data_prep(path,response,sensitive,atr,r,rnd,data_option,flag_demo)
    index = np.arange(len(_Xu))
    rnd_id = np.random.choice(index,b) 
    _Xl = np.append(_Xl,_Xu[rnd_id],axis=0)
    _Xl_s = np.append(_Xl_s,_Xu_s[rnd_id],axis=0)
    _yl=np.append(_yl,_yu[rnd_id],axis=0)
    clf = DemographicParityClassifier(sensitive_cols=-1, covariance_threshold=0.5)
    _Xl_with_s = np.append(_Xl, _Xl_s[:, None], axis=1)
    clf.fit(_Xl_with_s, _yl)
    _Xt_with_s = np.append(_Xt, _Xt_s[:, None], axis=1)
    _Cset_with_s = np.append(_Cset, _Cset_s[:, None], axis=1)
    score=clf.score(_Xt_with_s, _yt)
    demo_test=demo_test=dm.Demo(_Xt_with_s,_Xt_s,_yt,clf=clf,option=demo_option)
    return demo_test,_Xl,_Xl_s,_yl,clf,score,f1score 


