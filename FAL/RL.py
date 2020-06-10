# Hadis Anahideh,2020
import numpy as np
import math
from time import time
from sklearn.linear_model import LogisticRegression
import prep as pr
import demo_funcs as dm
import lr_inc as lr

def RL(path,response,sensitive,atr,demo_option,r,b,alpha,rnd,data_option):
    demo_test=[]
    _Xl,_Xl_s,n,m,_yl,_Xu,_Xu_s,_yu,_Xt,_Xt_s,_yt,_Cset,_Cset_s,_Cset_y=pr.data_prep(path,response,sensitive,atr,r,rnd,data_option)
    #Randomly sample b data points from U
    index = np.arange(len(_Xu))
    rnd_id = np.random.choice(index,b) 
    _Xl = np.append(_Xl,_Xu[rnd_id],axis=0)
    _Xl_s = np.append(_Xl_s,_Xu_s[rnd_id],axis=0)
    _yl=np.append(_yl,_yu[rnd_id],axis=0)
    
    # train the model on the sampled dataset
    clf = LogisticRegression(solver= 'liblinear').fit(_Xl, _yl)
    theta = clf.coef_.T                
    score=clf.score(_Xt,_yt)
    demo_test=np.append(demo_test,dm.Demo(_Xt,_Xt_s,_yt,clf=clf,option=demo_option))
    return demo_test,_Xl,clf,score


