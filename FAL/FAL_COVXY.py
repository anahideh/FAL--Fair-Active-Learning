import numpy as np
from time import time
from sklearn.linear_model import LogisticRegression
import FairnessByCov as fbc
import prep as pr
import demo_funcs as dm
import math

def FAL(path,response,sensitive,atr,demo_option,r,b,alpha,rnd,data_option,flag_demo):
    _Xl,_Xl_s,n,m,_yl,_Xu,_Xu_s,_yu,_Xt,_Xt_s,_yt,_Cset,_Cset_s,_Cset_y =pr.data_prep(path,response,sensitive,atr,r,rnd,data_option,flag_demo)
    overall_score=np.zeros(b)
    Time = np.zeros(b)
    demo = np.zeros(b)
    # train the model for the first time
    clf=LogisticRegression(solver= 'liblinear').fit(_Xl, _yl)
    theta= clf.coef_.T
    covXS = np.cov(np.concatenate((_Xu, _Xu_s.reshape(_Xu_s.shape[0],1)),1).T)[0:m,-1].reshape(m,1)
#     print('covXS_train:', covXS_tmp.T)
    # covXS = np.cov(np.concatenate((_Xt, _Xt_s.reshape(_Xt_s.shape[0],1)),1).T)[0:m,-1].reshape(m,1) # this is not correct
#     print('covXS_test:', covXS.T)
    fbc.init(_Xl, _yl,covXS,theta)
    t1 = time()
    for Iter in range(b):
        u=len(_Xu)
        ECov = np.zeros(u)
        # record stats to be reported -- these lines are added for the purpose of experiments and are not part of the algorithm
        a=alpha[min(10,math.floor(Iter/int(b/11)))]
        overall_score[Iter] = clf.score(_Xt, _yt)
        demo[Iter] = dm.Demo(_Xt,_Xt_s,_yt,clf=clf,option=demo_option)

        #compute the entropy and expected covariance improvement for all instances in U
        probas_val=clf.predict_proba(_Xu)
        e = (-probas_val * np.log2(probas_val)).sum(axis=1)
        covYS=np.dot(covXS.transpose(),theta)[0,0]
        #t2 = time()
        for j in range(0,u): 
            tmp = clf.predict_proba(_Xu[j].reshape(1, -1)).reshape(-1,1)
            ECov[j] = fbc.efi(_Xu[j],tmp)
        #print("Cov time = ",time()-t2)
        Emax = ECov.max(); Emin=ECov.min()
        
        # normalize the values
#         if Emax>Emin: ECov=(Emax-ECov)/(Emax-Emin)
        if Emax>Emin: ECov=(ECov-Emin)/(Emax-Emin)
        emin = e[0:u].min(); emax = e[0:u].max()
        if emax>emin: e = (e[0:u]-emin)/(emax-emin)
        e_all=a*e + (1-a)*ECov
        
#         e_all=a*e + (1-a)*(1-ECov)/covYS
        # find the argmax and add label it
        selection=np.argsort(e_all)[::-1][0]
        _Xl=np.append(_Xl,[_Xu[selection]],axis=0)
        _yl=np.append(_yl,[_yu[selection]],axis=0)

        # update the model
        clf = LogisticRegression(solver= 'liblinear').fit(_Xl, _yl)
        theta = clf.coef_.T
        fbc.updateAggs(_Xu[selection],_yu[selection],theta)
        _Xu = np.delete(_Xu, selection, 0)
        _yu = np.delete(_yu, selection, 0)
#         _Xu_s= np.delete(_Xu_s, selection, 0)
        ECov = ECov[:-1]
        #u-=1
        Time[Iter] = time()-t1
    return demo,_Xl,_Xl_s,_yl,theta,clf,overall_score,Time

