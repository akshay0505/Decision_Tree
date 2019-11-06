#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os
import copy
import sys
import matplotlib.pyplot as plt


# In[2]:


train_data = pd.read_csv(sys.argv[1])
val_data = pd.read_csv(sys.argv[2])
test_data = pd.read_csv(sys.argv[3])

# In[4]:


train_size = train_data.shape[0]
val_size = val_data.shape[0]
test_size = test_data.shape[0]



# In[5]:


print(train_size,val_size,test_size)

X = pd.concat([train_data,val_data,test_data],axis=0,ignore_index=True)

x1 = pd.get_dummies(X[" Work Class"],prefix_sep="_")
x2 = pd.get_dummies(X[" Marital Status"],prefix_sep="_")
x3 = pd.get_dummies(X[" Occupation"],prefix_sep="_")
x4 = pd.get_dummies(X[" Relationship"],prefix_sep="_")
x5 = pd.get_dummies(X[" Race"],prefix_sep="_")
x6 = pd.get_dummies(X[" Native Country"],prefix_sep="_")
# x7 = pd.get_dummies(train_data[" Education"],prefix_sep="_")

data = pd.concat([x1,
               x2,
               x3,
               x4,
               x5,
               x6,
            #   x7,
               X["Age"],
            #   X[" Fnlwgt"],
               X[" Education Number"],
               X[" Capital Gain"],
               X[" Capital Gain"],
               X[" Capital Loss"],
               X[" Hour per Week"],
               X[" Rich?"]],axis=1)

train = data.iloc[:train_size,:].values 
val = data.iloc[train_size:train_size+val_size,:].values
test = data.iloc[train_size+val_size:,:].values

print(train.shape,val.shape,test.shape)

def create_group_continuous(val,index,data):
    rich_0, rich_1 = [] , [] 
    for row in data:
        if row[index]<val:
            rich_0.append(row)
        else:
            rich_1.append(row)
    return np.array([np.array(rich_0),np.array(rich_1)])       

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

def split_groups(data):
    gini_score, split_value, index, best_groups = 1,None,None,None
    for i in range(data.shape[1]-1):
        # if(i==97):
        #     values = np.sort(np.unique(data[:,i])).tolist()
        #     inc = len(values)//100
        #     if(len(values)<200):
        #         inc=1
        #     for j in range(0,len(values),inc):
        #         groups = create_group_continuous(values[j],i,data)
        #         cal_gini = gini_calculate(groups)
        #         # print((i,cal_gini,index,gini_score),end="\r",flush=True)
        #         if(cal_gini<gini_score):
        #             gini_score = cal_gini
        #             split_value = values[j]
        #             index = i
        #             best_groups = groups
        # else:            
        for value in np.unique(data[:,i]).tolist():
            groups = create_group_continuous(value,i,data)
            cal_gini = gini_calculate(groups)
            if(cal_gini<gini_score):
                gini_score = cal_gini
                split_value = value
                index = i
                best_groups = groups
    return {"score":gini_score, "value": split_value, "index":index, "groups":best_groups}

def leaf_node(grp):
    if(np.sum(grp[:,-1]==1)>np.sum(grp[:,-1]==0)):
        return 1
    else:
        return 0

def get_class_number(grp):
    total = grp.shape[0]
    count_one = np.sum(grp[:,-1]==1)
    return [total-count_one,count_one]

def decision_tree(data,max_depth,min_size):
    root = split_groups(data)
    child_split(root,0,max_depth,min_size)
    return root

def child_split(node,depth,max_depth,min_size):    
    l , r = node["groups"][0], node["groups"][1]
    num_row = r.shape[1] if len(l)==0 else l.shape[1]
    node["true"] = 0
    node["false"] = 0
    node["attribute"] = leaf_node(np.concatenate([l.reshape(l.shape[0],num_row),r.reshape(r.shape[0],num_row)]))
    node["depth"] = depth
    del node["groups"]
    print(l.shape,r.shape)
    if(len(l)==0 or len(r)==0):
        num_row = r.shape[1] if len(l)==0 else l.shape[1]
        node["left"] = {}
        node["right"] = {}
        node["left"]["attribute"] = node["right"]["attribute"] = leaf_node(np.concatenate([l.reshape(l.shape[0],num_row),r.reshape(r.shape[0],num_row)]))
        node["left"]["true"] = node["left"]["false"] = node["right"]["true"] = node["right"]["false"] = 0
        node["left"]["depth"]=node["depth"]+1
        node["right"]["depth"]=node["depth"]+1
        return
    if depth>=max_depth:
        node["left"] = {}
        node["right"] = {}
        node["left"]["attribute"] , node["right"]["attribute"] = leaf_node(l), leaf_node(r)
        node["left"]["true"] = node["left"]["false"] = node["right"]["true"] = node["right"]["false"] = 0
        node["left"]["depth"]=node["depth"]+1
        node["right"]["depth"]=node["depth"]+1
        return 
    
    if(len(l)<=min_size):
        node["left"] = {}
        node["left"]["attribute"] = leaf_node(l)
        node["left"]["true"] = node["left"]["false"] = 0 
        node["left"]["depth"]=node["depth"]+1
    else:
        node["left"] = split_groups(l)
        child_split(node["left"],depth+1,max_depth,min_size)
        
    if(len(r)<=min_size):
        node["right"] = {}
        node["right"]["attribute"] = leaf_node(r)
        node["right"]["true"] = node["right"]["false"] = 0 
        node["right"]["depth"]=node["depth"]+1
    else:
        node["right"] = split_groups(r)
        child_split(node["right"],depth+1,max_depth,min_size)
def predict_classes(node,row):
    if((not "left" in node.keys()) or (not "right" in node.keys())): 
        return node["attribute"]
    if(row[node["index"]]<node["value"]):
        return predict_classes(node["left"],row)
    else:
        return predict_classes(node["right"],row)
def get_prun_acc(node):
    prun_acc = 0
    without_prun_acc = 0     
    if((not "left" in node["right"].keys()) and (not "left" in node["left"].keys())):    
        if(node["true"]+node["false"]!=0):
            without_prun_acc = node["true"]/(node["true"]+node["false"])
            if(node["right"]["true"]+node["right"]["false"]!=0 and node["left"]["true"]+node["left"]["false"]!=0):
                right_node_acc = node["right"]["true"]/(node["right"]["true"]+node["right"]["false"])
                left_node_acc = node["left"]["true"]/(node["left"]["true"]+node["left"]["false"])
                prun_acc = ((node["right"]["total"]*right_node_acc)+(node["left"]["total"]*left_node_acc))/(node["left"]["total"]+node["right"]["total"])
    elif(not "right" in node["right"].keys()):
        if(node["true"]+node["false"]!=0):
            without_prun_acc = node["true"]/(node["true"]+node["false"])
            if(node["right"]["true"]+node["right"]["false"]!=0 and node["left"]["true"]+node["left"]["false"]!=0):        
                right_node_acc = node["right"]["true"]/(node["right"]["true"]+node["right"]["false"])
                if(not "pruned_acc" in node["left"].keys()):
                    left_node_acc = node["left"]["true"]/(node["left"]["true"]+node["left"]["false"])
                else:
                    left_node_acc = node["left"]["pruned_acc"]
                prun_acc = ((node["right"]["total"]*right_node_acc)+(node["left"]["total"]*left_node_acc))/(node["left"]["total"]+node["right"]["total"])    
    elif(not "left" in node["left"].keys()):
        if(node["true"]+node["false"]!=0):
            without_prun_acc = node["true"]/(node["true"]+node["false"])
            if(node["right"]["true"]+node["right"]["false"]!=0 and node["left"]["true"]+node["left"]["false"]!=0):
                left_node_acc = node["left"]["true"]/(node["left"]["true"]+node["left"]["false"])
                if(not "pruned_acc" in node["right"].keys()):
                    right_node_acc = node["right"]["true"]/(node["right"]["true"]+node["right"]["false"])
                else:
                    right_node_acc = node["right"]["pruned_acc"]
                prun_acc = ((node["right"]["total"]*right_node_acc)+(node["left"]["total"]*left_node_acc))/(node["left"]["total"]+node["right"]["total"])       
    return prun_acc , without_prun_acc
def predict_classes_val(node,row,tree):
    if(node["attribute"]==row[-1]):
            node["true"]+=1
    else:
            node["false"]+=1
    node["total"] = node["true"]+node["false"]        
    if((not "left" in node.keys()) or (not "right" in node.keys())): 
        return node["attribute"]
    if(row[node["index"]]<node["value"]):
        return predict_classes_val(node["left"],row,tree)
    else:
        return predict_classes_val(node["right"],row,tree)
      
t = decision_tree(train,10,20)
# results = []
# for row in val:
#     results.append(predict_classes_val(t,row,t)) 
# c = 0 
# for i,r in enumerate(results):
#     if(val[i,-1]==r):
#         c+=1
# print(c/val.shape[0])
# acc_list = []
# def post_order(node):
#     global max_acc
#     global tree
#     global tree_opt
#     if((not "left" in node.keys()) or (not "right" in node.keys())): 
#         return 
#     post_order(node["left"])
#     post_order(node["right"])
#     prun , without_prun = get_prun_acc(node)
#     acc_list.append([prun , without_prun])
#     if(prun<=without_prun):
#         print(prun,without_prun)
#         node["pruned_acc"] = prun
#         del node["left"]
#         del node["right"]
# post_order(t)

results = []
for row in val:
    results.append(predict_classes(t,row)) 
np.savetxt(sys.argv[4],np.array(results))
c = 0 
for i,r in enumerate(results):
    if(val[i,-1]==r):
        c+=1
print(c/val_size,c)
results = []
for row in test:
    results.append(predict_classes(t,row)) 
np.savetxt(sys.argv[5],np.array(results))
