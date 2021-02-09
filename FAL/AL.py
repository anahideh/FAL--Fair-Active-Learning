import numpy as np
import math
from time import time
from sklearn.linear_model import LogisticRegression
import prep as pr
import demo_funcs as dm
import lr_inc as lr


def AL(path,response,sensitive,atr,demo_option,r,b,alpha,rnd,data_option,flag_demo): 
    demo_test=[]
    demo_cset=[]
    _Xl,_Xl_s,n,m,_yl,_Xu,_Xu_s,_yu,_Xt,_Xt_s,_yt,_Cset,_Cset_s,_Cset_y =pr.data_prep(path,response,sensitive,atr,r,rnd,data_option,flag_demo)
    overall_score=[]
    # train the model for the first time
    clf=LogisticRegression(solver= 'liblinear').fit(_Xl, _yl)
    theta= clf.coef_.T
    for Iter in range(b):
        u=len(_Xu)
        E_corr=np.zeros(u)
        # print("Iteration:", Iter)
        a=alpha[min(10,math.floor(Iter/int(b/11)))]
        # print(a)
        score=clf.score(_Xt, _yt)
        overall_score=np.append(overall_score,score)
        # print("test score is:", score)
        demo_test=np.append(demo_test,dm.Demo(_Xt,_Xt_s,_yt,clf=clf,option=demo_option))
        probas_val=clf.predict_proba(_Xu)
        e = (-probas_val * np.log2(probas_val)).sum(axis=1)
        e_all=(e)
        selection=np.argsort(e_all)[::-1][0]
        _Xl=np.append(_Xl,[_Xu[selection]],axis=0)
        _Xl_s=np.append(_Xl_s,[_Xu_s[selection]],axis=0)
        _yl=np.append(_yl,[_yu[selection]],axis=0)
        clf = LogisticRegression(solver= 'liblinear').fit(_Xl, _yl)
        _Xu = np.delete(_Xu, selection, 0)
        _yu = np.delete(_yu, selection, 0)
    return demo_test,_Xl,_Xl_s,_yl,clf,overall_score