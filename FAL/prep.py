import pandas as pd
import numpy as np
from sklearn import preprocessing
from numpy.random import RandomState
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def data_prep(path,response,sensitive,atr,r,rnd,data_option,flag_demo):
    if data_option==0: return data_prep_adult(path,response,sensitive,atr,r,rnd,flag_demo)
    else: return data_prep_compas(path,response,sensitive,atr,r,rnd,flag_demo)


def data_prep_adult(path,response,sensitive,atr,r,rnd,flag_demo):
    pathA=path[0]
    pathX=path[1]
    pathY=path[2]
    dfA=pd.read_csv(pathA,sep='\t',index_col = 0,header=None)
    dfX=pd.read_csv(pathX,sep='\t',index_col = 0)
    dfY=pd.read_csv(pathY,sep='\t',index_col = 0,header=None)
    atr=dfX.columns
    min_max_scaler = preprocessing.MinMaxScaler()
    x=dfX.values
    x_scaled = min_max_scaler.fit_transform(x)
    dfX = pd.DataFrame(x_scaled)
    dfX.columns=atr
    df=dfX
    df[sensitive]=dfA
    df[response]=dfY
    demo=[]
    np.random.seed(rnd[r])
    if flag_demo==True:
        valset0=df[df[sensitive]==0].sample(n=int(len(df)*0.05/2))
        valset1=df[df[sensitive]==1].sample(n=int(len(df)*0.05/2))
        valset=valset0.append(valset1)
        df.drop(valset.index, axis=0,inplace=True)
        _Cset = valset[atr].values
        _Cset_s = valset[sensitive].values
        _Cset_y = valset[response].values

    train, test = np.split(df.sample(frac=1), [int(.6*len(df))])
    L = train.groupby(response, group_keys=False).apply(lambda x: x.sample(n= 3,random_state=rnd[r]))
    X_L = L
    y_L = L[response].values
    y_L = pd.DataFrame(y_L, index=X_L.index.values)
    U=train.drop(X_L.index)[0:10000]
    X_U=U
    y_U=U[response].values
    y_U = pd.DataFrame(y_U, index=X_U.index.values)
    _Xl = X_L[atr].values
    (n,m) = _Xl.shape
    _yl = y_L[0].values
    _Xu = X_U[atr].values
    _Xu_s = X_U[sensitive].values
    _yu = y_U[0].values
    _Xt = test[atr].values
    _Xt_s = test[sensitive].values
    _yt = test[response].values
    if flag_demo==False:
        _Cset = np.copy(_Xu)
        _Cset_s =np.copy(_Xu_s)
        _Cset_y = np.copy(_yu)
    _Xl_s = X_L[sensitive].values
    return _Xl,_Xl_s,n,m,_yl,_Xu,_Xu_s,_yu,_Xt,_Xt_s,_yt,_Cset,_Cset_s,_Cset_y

def data_prep_compas(path,response,sensitive,atr,r,rnd,flag_demo):
    df = pd.read_csv(path)
    df = df[(df[sensitive]==2) | (df[sensitive]==3)]#african american 2
    df[sensitive] = df[sensitive]-2
    df['response_binary'] = np.where(df[response] > 0, 1, 0)

    y_all = df["response_binary"].values
    response="response_binary"
    demo=[]
    np.random.seed(rnd[r])

    if flag_demo==True:
        valset0=df[df[sensitive]==0].sample(n=int(len(df)*0.2/2))
        valset1=df[df[sensitive]==1].sample(n=int(len(df)*0.2/2))
        valset=valset0.append(valset1)
        df.drop(valset.index, axis=0,inplace=True)
        _Cset = valset[atr].values
        _Cset_s = valset[sensitive].values
        _Cset_y = valset[response].values

    train, test = np.split(df.sample(frac=1), [int(.6*len(df))])
    L = train.groupby(response, group_keys=False).apply(lambda x: x.sample(n= 3,random_state=rnd[r]))
    X_L = L
    y_L = L[response].values
    y_L = pd.DataFrame(y_L, index=X_L.index.values)
    U=train.drop(X_L.index)#[0:2000]
    X_U=U
    y_U=U[response].values
    y_U = pd.DataFrame(y_U, index=X_U.index.values)
    _Xl = X_L[atr].values
    (n,m) = _Xl.shape
    _yl = y_L[0].values
    _Xu = X_U[atr].values
    _Xu_s = X_U[sensitive].values
    _yu = y_U[0].values
    _Xt = test[atr].values
    _Xt_s = test[sensitive].values
    _yt = test[response].values
    if flag_demo==False:
        _Cset = np.copy(_Xu)
        _Cset_s =np.copy(_Xu_s)
        _Cset_y = np.copy(_yu)
    _Xl_s = X_L[sensitive].values
    return _Xl,_Xl_s,n,m,_yl,_Xu,_Xu_s,_yu,_Xt,_Xt_s,_yt,_Cset,_Cset_s,_Cset_y

