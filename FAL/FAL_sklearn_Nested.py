import numpy as np
import math
from time import time
from sklearn.linear_model import LogisticRegression
import prep as pr
import demo_funcs as dm
import lr_inc as lr
from sklearn.metrics import f1_score


def FAL(path,response,sensitive,atr,demo_option,r,b,alpha,rnd,data_option,flag_demo,kk): 
    demo_test=[]
    demo_cset=[]
    Time = np.zeros(b)
    _Xl,_Xl_s,n,m,_yl,_Xu,_Xu_s,_yu,_Xt,_Xt_s,_yt,_Cset,_Cset_s,_Cset_y =pr.data_prep(path,response,sensitive,atr,r,rnd,data_option,flag_demo)
    overall_score=[]
    f1score=[]
    # train the model for the first time
    clf=LogisticRegression(solver= 'liblinear').fit(_Xl, _yl)
    t1 = time()
    for Iter in range(b):
        u=len(_Xu)
#         print("Iteration:", Iter)
        a=alpha[min(10,math.floor(Iter/int(b/11)))]
        score=clf.score(_Xt, _yt)
        overall_score=np.append(overall_score,score)
        y_pred=clf.predict(_Xt)
        demo_test=np.append(demo_test,dm.Demo(_Xt,_Xt_s,_yt,clf=clf,option=demo_option))
        demo_cset=dm.Demo(_Cset,_Cset_s,_Cset_y,clf=clf,option=demo_option)
        probas_val=clf.predict_proba(_Xu)
        e = (-probas_val * np.log2(probas_val)).sum(axis=1)
        idx=np.argsort(e)[::-1][0:kk]
        E_corr=np.zeros(kk)
        for j in range(0,len(idx)):
            f_tmp=[]
            for k in range(0,2):
                _Xl_tmp=np.append(_Xl,[_Xu[idx[j]]],axis=0)
                _yl_tmp=np.append(_yl,[k],axis=0)
                clf_tmp=LogisticRegression(solver='liblinear').fit(_Xl_tmp, _yl_tmp)
                f_tmp=np.append(f_tmp,dm.Demo(_Cset,_Cset_s,_Cset_y,clf=clf_tmp,option=demo_option))
                f_tmp[np.isnan(f_tmp)] = 0
            p = clf.predict_proba(_Xu)[idx[j]][0]
            E_corr[j]=(f_tmp).dot([p,1-p])
        # find the argmax and add label it
        selection=idx[np.argsort(E_corr)[::1][0]]
        _Xl=np.append(_Xl,[_Xu[selection]],axis=0)
        _Xl_s=np.append(_Xl_s,[_Xu_s[selection]],axis=0)
        _yl=np.append(_yl,[_yu[selection]],axis=0)
        clf = LogisticRegression(solver= 'liblinear').fit(_Xl, _yl)
        _Xu = np.delete(_Xu, selection, 0)
        _yu = np.delete(_yu, selection, 0)
        _Xu_s= np.delete(_Xu_s, selection, 0)
        Time[Iter] = time()-t1
    return demo_test,_Xl,_Xl_s,_yl,clf,overall_score,f1score