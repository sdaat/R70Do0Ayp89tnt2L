#Firstly, imported necessery modules for the model performance.
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

#To load datasets:
data=pd.read_csv("term-deposit-marketing-2020.csv",na_values="unknown")

#To convert target variable to numeric type:
data["y"]=(data["y"]=="yes").astype(float)

#Columns with only have two class are converted as numeric type:
data["loan_no"]=(data["loan"]=="no").astype(float)
data["housing_no"]=(data["housing"]=="no").astype(float)
data["default_no"]=(data["default"]=="no").astype(float)

#Contact column have null values as %25 of entire values. Also, if take null values as a new class in the column, it treat significantly differ 
#from the other classes respectively target variable. So, null values in this column take as new class.
data["contact"]=data["contact"].fillna("Unknown")

#Row with null values are dropped the dataframe.
data=data.dropna()

#To handle with the month and day column, the target variable investigate its mean with grouping by day and month variables.
data_dist=pd.DataFrame(data.groupby(["month","day"]).y.agg([np.mean])) 
data_dist=data_dist.reset_index()

#To avoid overfitting, mean must be generalize over the day and month grouping.So firtly multiply 10 and rounding the nearest integer.
data_dist["day_mean_factor"]=round(data_dist["mean"]*10)
data_dist=data_dist.drop(["mean"],axis=1)

#To merge day_mean_factor column the original datasets:
data=data.merge(data_dist,how="left",on=["day","month"])

#Converting column eliminate the dataframe:
data=data.drop(["loan","housing","default","day","month"],axis=1)

#To invastigate target summary statistics each column class, distirbution_with_target_mean function are been writing:
def distirbution_with_target_mean(data,obj_column,y):
    distirbution=[]
    a=data.copy()
    a["y"]=y
    for i in obj_column:
        dist=a.groupby(i).y.describe()
        distirbution.append(dist)
    return distirbution

#To important results will share in documentation. 
dist=distirbution_with_target_mean(data.drop("y",axis=1), data.drop("y",axis=1).columns, data["y"])

#To separate target variables with the dataframe:
y=data["y"]
data=data.drop("y",axis=1)

#To get dummy variables, firstly, numeric and object column must be seperated.Also, I used this functions with multiple predictive model problem.
#So, it have a general format.
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
    
obj_column,int_column=object_or_integer(data.drop("y",axis=1))

#To get dummy variables:
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
    
data_df=category_onehot_other(obj_column, data)
 
#I choose RandomForestClassifier, GradientBoostingClassifier and SVC without tuning hyperparameters because of competionally long run-time.
#Mean of accuracy score each classifier; 0.911, 0.916 and 0.926 respectively counting the first sentences. However, in spite of the accuracy score, 
#model can improved with new feature. 

rf=RandomForestClassifier()
scores_rf=cross_val_score(rf,data_df,y,scoring="accuracy",cv=5)
print(np.mean(scores_rf))
gb=GradientBoostingClassifier()
scores_gb=cross_val_score(gb,data_df,y,scoring="accuracy",cv=5)
print(np.mean(scores_gb))
svc=SVC()
scores_svc=cross_val_score(svc,data_df,y,scoring="accuracy",cv=5)
print(np.mean(scores_svc))



