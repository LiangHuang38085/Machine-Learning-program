#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#基于鸢尾花（iris）数据集的逻辑回归分类实践
#函数包的常见参数需要多熟悉
#数据分析流程仍然需要整理


# In[5]:


#sigmoid函数
import numpy as np
import matplotlib.pyplot as plt
x=np.arange(-5,5,0.01)
y=1/(1+np.exp(-x))
plt.plot(x,y)
plt.xlabel('z')
plt.ylabel('y')
plt.grid()
plt.show()


# In[6]:


#一：库函数导入
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[8]:


#二：数据读取载入(Iris数据为sklearn自带)
from sklearn.datasets import load_iris
#获取数据特征
data=load_iris()
#获取数据对应的标签（三种类别，0,1,2）
iris_target=data.target
#使用pandas将数据转化为Dataframe格式
#pandas.DataFrame( data, index, columns, dtype, copy)
iris_features=pd.DataFrame(data=data.data,columns=data.feature_names)


# In[9]:


#三：数据信息简单查看
#使用.info()查看数据整体信息，使用.head()和.tail()简单查看数据信息
iris_features.info()


# In[11]:


iris_features.head()


# In[12]:





# In[13]:


#查看对应的类别标签
iris_target


# In[15]:


#使用value_counts函数查看每个类别的数量
pd.Series(iris_target).value_counts()


# In[16]:


#对特征进行统计描述
iris_features.describe()


# In[18]:


#四：可视化描述
#合并标签和特征信息
#进行浅拷贝，保护原始数据完整性
iris_all=iris_features.copy()
iris_all['target']=iris_target


# In[19]:


#特征与标签组合的散点可视化
'''
seaborn.pairplot(data, hue=None, hue_order=None, 
                 palette=None, vars=None, x_vars=None,
                 y_vars=None, kind='scatter', diag_kind='auto', 
                 markers=None, height=2.5, aspect=1,
                 dropna=True,plot_kws=None, diag_kws=None,
                 grid_kws=None, size=None)
data: DataFrame

hue:变量名称
作用：用颜色将数据进行第二次分组

hue_order:字符串列表
作用：指定调色板中颜色变量的顺序

palette:调色板

vars:变量名列表

{x,y}_vars:变量名列表
作用：指定数据中变量分别用于图的行和列，

kind：{"scatter","reg"}
作用：指定数据之间的关系eg. kind="reg":指定数据的线性回归

diag_kind:{"auto","hist","kde"}
作用：指定对角线处子图的类型，默认值取决与是否使用hue。参考案例9和案例11

markers:标记

height:标量
作用：指定图的大小(图都是正方形的，所以只要指定height就行)

{plot，diag，grid} _kws：dicts字典
作用：指定关键字参数的字典
'''
sns.pairplot(data=iris_all,diag_kind='hist',hue='target')
plt.show()


# In[23]:


#从上图大概确定不同特征组合下的区分能力
'''
seaborn.boxplot(x=None, y=None, hue=None, data=None, order=None, hue_order=None, orient=None, color=None, 
palette=None, saturation=0.75, 
width=0.8, dodge=True, fliersize=5, linewidth=None, whis=1.5, notch=False, ax=None, **kwargs)
'''
#饱和度、颜色
for col in iris_features.columns:
    sns.boxplot(x='target',y=col,saturation=0.75,palette='pastel',data=iris_all)
    plt.title(col)
    plt.show()


# In[28]:


#利用箱型图得到不同类别在不同特征上的分布差异
#选取前三个特征绘制三维散点图
from mpl_toolkits.mplot3d import Axes3D
fig=plt.figure(figsize=(10,8))
ax=fig.add_subplot(111,projection='3d')

iris_all_class0=iris_all[iris_all['target']==0].values
iris_all_class1=iris_all[iris_all['target']==1].values
iris_all_class2=iris_all[iris_all['target']==2].values

ax.scatter(iris_all_class0[:,0],iris_all_class1[:,1],iris_all_class0[:,2],label='setosa')
ax.scatter(iris_all_class1[:,0],iris_all_class1[:,1],iris_all_class1[:,2],label='versicolor')
ax.scatter(iris_all_class2[:,0],iris_all_class2[:,1],iris_all_class2[:,2],label='virginica')
plt.legend()#加图例
plt.show()


# In[30]:


#利用逻辑回归模型
#进行二分类训练与预测
from sklearn.model_selection import train_test_split
#选择0和1的类别
iris_features_part=iris_features.iloc[:100]
iris_target_part=iris_target[:100]

x_train,x_test,y_train,y_test=train_test_split(iris_features_part,iris_target_part,test_size=0.2,random_state=2020)
from sklearn.linear_model import LogisticRegression
clf=LogisticRegression(random_state=0,solver='lbfgs')
clf.fit(x_train,y_train)


# In[32]:


print('The weight of Logistic Regression:',clf.coef_)
print('The intercept(w0) of Logistic Regression:',clf.intercept_)


# In[34]:


#在训练集和测试集上进行模型预测
train_predict=clf.predict(x_train)
test_predict=clf.predict(x_test)
from sklearn import metrics

print('The accuracy of the Logistic Regression is:',metrics.accuracy_score(y_train,train_predict))
print('The accuracy of the Logistic Regression is:',metrics.accuracy_score(y_test,test_predict))

#查看混淆矩阵
confusion_matrix_result=metrics.confusion_matrix(test_predict,y_test)
print('The confusion matrix result:\n',confusion_matrix_result)

#利用热力图对结果进行可视化
plt.figure(figsize=(8,6))
sns.heatmap(confusion_matrix_result,annot=True,cmap='Blues')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.show()


# In[35]:


#利用逻辑回归模型进行多分类训练与预测
x_train,x_test,y_train,y_test=train_test_split(iris_features,iris_target,test_size=0.2,random_state=2020)
clf=LogisticRegression(random_state=0,solver='lbfgs')
clf.fit(x_train,y_train)


# In[36]:


print('The weight of Logistic Regression:',clf.coef_)
print('The intercept(w0) of Logistic Regression:',clf.intercept_)


# In[37]:


train_predict=clf.predict(x_train)
test_predict=clf.predict(x_test)
#利用predict_proba函数预测概率
train_predict_proba=clf.predict_proba(x_train)
test_predict_proba=clf.predict_proba(x_test)

print('The test predict Probability of each class:\n',test_predict_proba)

print('The accuracy of the Logistic Regression is:',metrics.accuracy_score(y_train,train_predict))
print('The accuracy of the Logistic Regression is:',metrics.accuracy_score(y_test,test_predict))


# In[40]:


confusion_matrix_result=metrics.confusion_matrix(test_predict,y_test)
print('The confusion matrix result:\n',confusion_matrix_result)

#利用热力图对结果进行可视化
#annot为True意思是在方格内写入数据
plt.figure(figsize=(8,6))
sns.heatmap(confusion_matrix_result,annot=True,cmap='Blues')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.show()


# In[ ]:


#1，2类特征不够明显，导致分类错误的发生


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




