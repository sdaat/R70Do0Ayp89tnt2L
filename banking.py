import pandas as pd
import numpy as np 
from scipy.stats import pearsonr
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split



def object_or_integer(data):
    column_name=data.columns
    object_type=[]
    numeric_type=[]
    for i in column_name:
        if data[i].dtype==object:
            object_type.append(i)
        else:
            numeric_type.append(i)
    return object_type,numeric_type

def distirbution_with_target_mean(data,obj_column,y):
    distirbution=[]
    a=data.copy()
    a["y"]=y
    for i in obj_column:
        dist=a.groupby(i).y.describe()
        dist=dist.reset_index()
        distirbution.append(dist)
    return distirbution

def correlation_with_target(int_column,data,y):
    corre={}
    p_value={}
    dropcolumn=[]
    for i in int_column:
        corr = pearsonr(data[i],y)
        if (abs(corr[0]) <0.05) or abs(corr[1])>0.05 :
            dropcolumn.append(i)
        corre[i]=corr[0]
        p_value[i]=corr[1]
    # plt.scatter(corre.keys(),corre.values())
    # plt.xticks(rotation=90)
    return corre,dropcolumn,p_value

def category_onehot_other(colums,finalset):
    setfinal=finalset
    i=0
    for field in colums:
        df1=pd.get_dummies(finalset[field],drop_first=True,prefix=field)
        finalset.drop([field],axis=1,inplace=True)
        if i ==0:
            setfinal=df1.copy()
        else:
            setfinal=pd.concat([setfinal,df1],axis=1)
        i=i+1
    setfinal=pd.concat([finalset,setfinal],axis=1)
    return setfinal

data=pd.read_csv("term-deposit-marketing-2020.csv",na_values="unknown")
data["y"]=(data["y"]=="yes").astype(float)
data["loan_no"]=(data["loan"]=="no").astype(float)
data["housing_no"]=(data["housing"]=="no").astype(float)
data["default_no"]=(data["default"]=="no").astype(float)
dataa=data.copy()
data["contact"]=data["contact"].fillna("Unknown")
# data["duration_long"]=(data["duration"]>400).astype(float)


data_dist=pd.DataFrame(data.groupby(["month","day"]).y.agg([np.mean])) 
data_dist=data_dist.reset_index()
data_dist["day_mean_factor"]=round(data_dist["mean"]*10)
# data_dist["day_volume_fac"]=round(data_dist["len"]/100)
data_dist=data_dist.drop(["mean"],axis=1)
data=data.merge(data_dist,how="left",on=["day","month"])
# data["age"]=data["age"]*((data["age"] > 27) | (data["age"] < 60)).astype(float)

# data.loc[data.balance<0,"balance"]=(-1)*round(np.log10(-1*data.loc[data.balance<0,"balance"]))
# data.loc[data.balance>0,"balance"]=round(np.log10(data.loc[data.balance>0,"balance"]))
data=data.dropna()
# data["campaign"]=data["campaign"]*(data["campaign"]<18).astype(float)

data=data.drop(["loan","housing","default","day","month"],axis=1)
obj_column,int_column=object_or_integer(data.drop("y",axis=1))

dist=distirbution_with_target_mean(data.drop("y",axis=1), data.drop("y",axis=1).columns, data["y"])
# corr,dropcolumn,pvalue=correlation_with_target(int_column, data, data["y"])

y=data["y"]
data=data.drop("y",axis=1)
data_df=category_onehot_other(obj_column, data)
# corr1,dropcolumn1,pvalue1=correlation_with_target(data_df.columns, data_df, y)

X_train,X_test,y_train,y_test=train_test_split(data_df,y,test_size=0.2,random_state=42,stratify=y)

# rf=RandomForestClassifier()
# scores_rf=cross_val_score(rf,data_df,y,scoring="accuracy",cv=5)
# print(np.mean(scores_rf))
# gb=GradientBoostingClassifier()
# scores_gb=cross_val_score(gb,data_df,y,scoring="accuracy",cv=5)
# print(np.mean(scores_gb))
# svc=SVC()
# scores_svc=cross_val_score(svc,data_df,y,scoring="accuracy",cv=5)
# print(np.mean(scores_svc))


# svc.fit(X_train,y_train)
# y_pred_svc=svc.predict(X_test)

