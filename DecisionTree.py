

import pandas as pd
import numpy as np
import os


data_dir = "/Users/akshay/Desktop/A4/DT_data/"
os.listdir(data_dir)


# In[3]:


train_data = pd.read_csv(data_dir+"train.csv")
val_data = pd.read_csv(data_dir+"valid.csv")
test_data = pd.read_csv(data_dir+"test_public.csv")


# In[4]:


train_size = train_data.shape[0]
val_size = val_data.shape[0]
test_size = test_data.shape[0]


# In[5]:


print(train_size,val_size,test_size)


# In[6]:


X = pd.concat([train_data,val_data,test_data],axis=0,ignore_index=True)


# In[7]:


for i in list(X.columns):
    print(i,len(X[i].unique().tolist()))


# In[8]:


x1 = pd.get_dummies(X[" Work Class"],prefix_sep="_")
x2 = pd.get_dummies(X[" Marital Status"],prefix_sep="_")
x3 = pd.get_dummies(X[" Occupation"],prefix_sep="_")
x4 = pd.get_dummies(X[" Relationship"],prefix_sep="_")
x5 = pd.get_dummies(X[" Race"],prefix_sep="_")
x6 = pd.get_dummies(X[" Native Country"],prefix_sep="_")
# x7 = pd.get_dummies(train_data[" Education"],prefix_sep="_")


# In[9]:


data = pd.concat([x1,
               x2,
               x3,
               x4,
               x5,
               x6,
               X["Age"],
#                train_data[" Fnlwgt"],
               X[" Education Number"],
               X[" Capital Gain"],
               X[" Capital Gain"],
               X[" Capital Loss"],
               X[" Hour per Week"],
               X[" Rich?"]],axis=1)


# In[ ]:


X.info()


# In[10]:


train = data.iloc[:train_size,:].values 
val = data.iloc[train_size:train_size+val_size,:].values
test = data.iloc[train_size+val_size:,:].values


# In[11]:


print(train.shape,val.shape,test.shape)


# In[12]:


def create_group_continuous(val,index,data):
    rich_0, rich_1 = [] , [] 
    for row in data:
        if row[index]<val:
            rich_0.append(row)
        else:
            rich_1.append(row)
    return np.array([np.array(rich_0),np.array(rich_1)])       


# In[13]:


def gini_calculate(groups):
    gini=0
    total_groups=0
    for grp in groups:
        if(len(grp)==0):
            continue
        else:            
            group_0 = np.sum(grp[:,-1]==0)/grp[:,-1].shape[0]
            group_1 = np.sum(grp[:,-1]==1)/grp[:,-1].shape[0]
            gini+= (1-(group_0*group_0+group_1*group_1))*(grp[:,-1].shape[0])
            total_groups+=grp[:,-1].shape[0]
    return gini/total_groups        


# In[14]:


def split_groups(data):
    gini_score, split_value, index, best_groups = 1,None,None,None
    for i in range(data.shape[1]-1):
        for value in np.unique(data[:,i]).tolist():
            groups = create_group_continuous(value,i,data)
            cal_gini = gini_calculate(groups)
#             print((i,value,index,gini_score),end="\r",flush=True)
            if(cal_gini<gini_score):
                gini_score = cal_gini
                split_value = value
                index = i
                best_groups = groups
    return {"score":gini_score, "value": split_value, "index":index, "groups":best_groups}


# In[15]:


def leaf_node(grp):
    if(np.sum(grp[:,-1]==1)>np.sum(grp[:,-1]==0)):
        return 1
    else:
        return 0


# In[111]:


def child_split(node,depth,max_depth,min_size):    
#     print((depth,node["score"]),end="\r",flush=True)
    l , r = node["groups"][0], node["groups"][1]
    del node["groups"]
    if depth>=max_depth:
        node["left"] , node["right"] = leaf_node(l), leaf_node(r)
        return 
    if(len(l)==0 or len(r)==0):
        num_row = r.shape[1] if len(l)==0 else l.shape[1]
        node["left"] = node["right"] = leaf_node(np.concatenate([l.reshape(l.shape[0],num_row),r.reshape(r.shape[0],num_row)]))
#         node["left"] = node["right"]
        return
    
    if(len(l)<=min_size):
        node["left"] = leaf_node(l)
    else:
        node["left"] = split_groups(l)
        child_split(node["left"],depth+1,max_depth,min_size)
        
    if(len(r)<=min_size):
        node["right"] = leaf_node(r)
    else:
        node["right"] = split_groups(r)
        child_split(node["right"],depth+1,max_depth,min_size)


def decision_tree(data,max_depth,min_size):
    root = split_groups(data)
    child_split(root,0,max_depth,min_size)
    return root


def predict_classes(node,row):
    if(row[node["index"]]<node["value"]):
        if(isinstance(node["left"],dict)):
            return predict_classes(node["left"],row)
        else:
            return node["left"]
    else:
        if(isinstance(node["right"],dict)):
            return predict_classes(node["right"],row)
        else:
            return node["right"]


tree = decision_tree(train,4,20)

results = []
for row in val:
    results.append(predict_classes(tree,row)) 
c = 0 
for i,r in enumerate(results):
    if(val[i,-1]==r):
        c+=1
print(c/val.shape[0],c)

results = []
for row in train:
    results.append(predict_classes(tree,row)) 
c = 0 
for i,r in enumerate(results):
    if(train[i,-1]==r):
        c+=1
print(c/train.shape[0],c)


