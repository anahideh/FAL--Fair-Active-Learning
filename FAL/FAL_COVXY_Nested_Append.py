import numpy as np
from time import time
from sklearn.linear_model import LogisticRegression
import FairnessByCov as fbc
import prep as pr
import demo_funcs as dm
import math

def FAL(path,response,sensitive,atr,demo_option,r,b,alpha,rnd,data_option,flag_demo,k):
    _Xl,_Xl_s,n,m,_yl,_Xu,_Xu_s,_yu,_Xt,_Xt_s,_yt,_Cset,_Cset_s,_Cset_y =pr.data_prep(path,response,sensitive,atr,r,rnd,data_option,flag_demo)
    overall_score=np.zeros(b)
    Time = np.zeros(b)
    demo = np.zeros(b)
    # train the model for the first time
    clf=LogisticRegression(solver= 'liblinear').fit(_Xl, _yl)
    theta= clf.coef_.T
    covXS = np.cov(np.concatenate((_Xu, _Xu_s.reshape(_Xu_s.shape[0],1)),1).T)[0:m,-1].reshape(m,1)
    fbc.init(_Xl, _yl,covXS,theta)
    t1 = time()
    for Iter in range(b):
        u=len(_Xu)
        ECov = np.zeros(u)
        # record stats to be reported -- these lines are added for the purpose of experiments and are not part of the algorithm
        a=alpha[min(10,math.floor(Iter/int(b/11)))]
        overall_score[Iter] = clf.score(_Xt, _yt)
        demo[Iter] = dm.Demo(_Xt,_Xt_s,_yt,clf=clf,option=demo_option)

        #compute the entropy and expected covariance improvement 
        probas_val=clf.predict_proba(_Xu)
        e = (-probas_val * np.log2(probas_val)).sum(axis=1)
        covYS1=np.dot(covXS.transpose(),theta)[0,0]
        idx=np.argsort(e)[::-1][0:k]
        ECov = np.zeros(k)
        for j in range(0,len(idx)): 
            tmp = clf.predict_proba(_Xu[idx[j]].reshape(1, -1)).reshape(-1,1)
            ECov[j] = fbc.efi(_Xu[idx[j]],tmp)
        # find the argmax and add label it
        selection=idx[np.argsort(ECov)[::-1][0]]
        _Xl_tmp=np.append(_Xl,[_Xu[selection]],axis=0)
        _yl_tmp=np.append(_yl,[_yu[selection]],axis=0)
        clf_tmp = LogisticRegression(solver= 'liblinear').fit(_Xl_tmp, _yl_tmp)
        theta_tmp = clf_tmp.coef_.T
        covYS2=np.dot(covXS.transpose(),theta_tmp)[0,0]
        #replicate the points that improves unfairness reduction after labeling them
        if abs(covYS2)-abs(covYS1)<0:
            _Xl=np.append(_Xl,[_Xu[selection]],axis=0)
            _yl=np.append(_yl,[_yu[selection]],axis=0)
            _Xl_s=np.append(_Xl_s,[_Xu_s[selection]],axis=0)
        _Xl=np.append(_Xl,[_Xu[selection]],axis=0)
        _yl=np.append(_yl,[_yu[selection]],axis=0)
        _Xl_s=np.append(_Xl_s,[_Xu_s[selection]],axis=0)
        # update the model
        clf = LogisticRegression(solver= 'liblinear').fit(_Xl, _yl)
        theta = clf.coef_.T
        fbc.updateAggs(_Xu[selection],_yu[selection],theta)
        _Xu = np.delete(_Xu, selection, 0)
        _yu = np.delete(_yu, selection, 0)
        _Xu_s= np.delete(_Xu_s, selection, 0)
        ECov = ECov[:-1]
        Time[Iter] = time()-t1
    return demo,_Xl,_Xl_s,_yl,theta,clf,overall_score

