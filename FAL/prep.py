# Hadis Anahideh,2020
import pandas as pd
import numpy as np
from sklearn import preprocessing
from numpy.random import RandomState
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def data_prep(path,response,sensitive,atr,r,rnd,data_option):
    if data_option==0: return data_prep_adult(path,response,sensitive,atr,r,rnd)
    else: return data_prep_compas(path,response,sensitive,atr,r,rnd)


def data_prep_adult(path,response,sensitive,atr,r,rnd):
    #read the dataset
    pathA=path[0]
    pathX=path[1]
    pathY=path[2]
    dfA=pd.read_csv(pathA,sep='\t',index_col = 0,header=None)
    dfX=pd.read_csv(pathX,sep='\t',index_col = 0)
    dfY=pd.read_csv(pathY,sep='\t',index_col = 0,header=None)
    atr=dfX.columns
    
    #normalize the data
    min_max_scaler = preprocessing.MinMaxScaler()
    x=dfX.values
    x_scaled = min_max_scaler.fit_transform(x)
    dfX = pd.DataFrame(x_scaled)
    dfX.columns=atr
    df=dfX
    df[sensitive]=dfA
    df[response]=dfY
    np.random.seed(rnd[r])
    train, test = np.split(df.sample(frac=1), [int(.6*len(df))])
    
    #sample the initial labeled set
    L = train.groupby(response, group_keys=False).apply(lambda x: x.sample(n= 3,random_state=rnd[r]))
    X_L = L
    y_L = L[response].values
    y_L = pd.DataFrame(y_L, index=X_L.index.values)
    
    #define the set of unlabeled sample
    U=train.drop(X_L.index)
    X_U=U
    y_U=U[response].values
    y_U = pd.DataFrame(y_U, index=X_U.index.values)
    _Xl = X_L[atr].values
    (n,m) = _Xl.shape
    
    #define the response, attribute, and sensitive arrays of the initial labeled and unlabled datasets
    _yl = y_L[0].values
    _Xu = X_U[atr].values
    _Xu_s = X_U[sensitive].values
    _yu = y_U[0].values
    _Xt = test[atr].values
    _Xt_s = test[sensitive].values
    _yt = test[response].values
    _Cset = np.copy(_Xu)
    _Cset_s =np.copy(_Xu_s)
    _Cset_y = np.copy(_yu)
    _Xl_s = X_L[sensitive].values
    return _Xl,_Xl_s,n,m,_yl,_Xu,_Xu_s,_yu,_Xt,_Xt_s,_yt,_Cset,_Cset_s,_Cset_y

def data_prep_compas(path,response,sensitive,atr,r,rnd):
    #read the dataset
    df = pd.read_csv(path)
    df = df[(df[sensitive]==2) | (df[sensitive]==3)]#african american 2
    df[sensitive] = df[sensitive]-2
    df['response_binary'] = np.where(df[response] > 0, 1, 0)
    y_all = df["response_binary"].values
    response="response_binary"
    np.random.seed(rnd[r])
    train, test = np.split(df.sample(frac=1), [int(.6*len(df))])
    
    #sample the initial labeled set
    L = train.groupby(response, group_keys=False).apply(lambda x: x.sample(n= 3,random_state=rnd[r]))
    X_L = L
    y_L = L[response].values
    y_L = pd.DataFrame(y_L, index=X_L.index.values)
    
    #define the set of unlabeled sample
    U=train.drop(X_L.index)
    X_U=U
    y_U=U[response].values
    y_U = pd.DataFrame(y_U, index=X_U.index.values)
    _Xl = X_L[atr].values
    (n,m) = _Xl.shape
    
    #define the response, attribute, and sensitive arrays of the initial labeled and unlabled datasets
    _yl = y_L[0].values
    _Xu = X_U[atr].values
    _Xu_s = X_U[sensitive].values
    _yu = y_U[0].values
    _Xt = test[atr].values
    _Xt_s = test[sensitive].values
    _yt = test[response].values
    _Cset = np.copy(_Xu)
    _Cset_s =np.copy(_Xu_s)
    _Cset_y = np.copy(_yu)
    _Xl_s = X_L[sensitive].values
    return _Xl,_Xl_s,n,m,_yl,_Xu,_Xu_s,_yu,_Xt,_Xt_s,_yt,_Cset,_Cset_s,_Cset_y

