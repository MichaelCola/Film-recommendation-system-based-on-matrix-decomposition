#!/usr/bin/env python
# coding: utf-8

# # 第一步：收集数据

# In[1]:


import pandas as pd


# In[2]:


import numpy as np


# In[3]:


import tensorflow as tf


# In[4]:


# 读取用户评分文件
ratings_df = pd.read_csv('H:/Desktop/ml-latest-small/ratings.csv')


# In[87]:


# 默认是取dataframe中的最后五行
# 此函数根据位置从对象返回最后n行。例如，在对行进行排序或追加之后，它对于快速验证数据很有用
ratings_df.tail()


# In[6]:


# 读取电影种类表格
movies_df = pd.read_csv('H:/Desktop/ml-latest-small/movies.csv')


# In[7]:


movies_df.tail()


# #####  此时的movieId太大，如果以movieId最大值构建评分矩阵，评分表将会过大，而且十分稀疏，十分浪费内存。所以在后面使用行号。

# In[8]:


# 将电影的行号信息添加进去。
movies_df['movieRow'] = movies_df.index


# In[9]:


movies_df.tail()


# ### 筛选movies_df中的特征

# In[10]:


movies_df = movies_df[['movieRow', 'movieId', 'title']]


# In[11]:


movies_df.to_csv('moviesProcessed.csv', index = False, header = True, encoding='utf-8')


# In[12]:


movies_df.tail()


# ### 将ratings_df中的movieId替换为行号

# In[13]:


ratings_df = pd.merge(ratings_df, movies_df, on='movieId')


# In[59]:


# 查看前5行
ratings_df.tail()


# In[15]:


ratings_df = ratings_df[['userId', 'movieRow', 'rating']]


# In[16]:


ratings_df.to_csv('ratingProcessed.csv', index = False, header = True, encoding = 'utf-8')


# In[58]:


np.shape(ratings_df)


# ### 创建电影评分矩阵rating和评分记录矩阵record

# In[18]:


userNo = ratings_df['userId'].max()+1


# In[19]:


movieNo = ratings_df['movieRow'].max() + 1


# In[20]:


userNo


# In[21]:


movieNo


# In[22]:


#新建一个movieNo行，userNo列的0矩阵
rating = np.zeros((movieNo, userNo))


# In[65]:


#记录处理的进度
flag = 0 
# 总共有多少条用户对电影的评分
ratings_df_length = np.shape(ratings_df)[0]

#遍历行数据
for index, row in ratings_df.iterrows():
    # 将ratings_df中的数据，整理到rating数组中，行为movieRow，列为userId。
    # 第movieRow行，第userId的数据为评分rating。
    rating[int(row['movieRow']), int(row['userId'])] = row['rating']
    flag += 1
    print('Processed %d, %d left' %(flag, ratings_df_length - flag))


# In[64]:


rating[0,:]


# In[24]:


# 记录某人是否对某电影进行评分
record = rating > 0


# In[25]:


record


# In[26]:


record = np.array(record, dtype=int)


# In[27]:


record


# # 第三步：构建模型

# In[88]:


def normalizeRatings(rating, record):
    m, n = rating.shape
    rating_mean = np.zeros((m, 1))
    
    # rating_norm用于保存处理之后的数据
    rating_norm = np.zeros((m, n))
    
    for i in range(10):
        
        # 获取每一行中用户评过分的下标
        idx = record[i, :] != 0 

        # 计算每一行的均值，ndarray可以用布尔值索引，返回数据为true的值
        rating_mean[i] = np.mean(rating[i, idx])
        
        # 每一行的评分减去均值
        rating_norm[i, idx] -= rating_mean[i]
    return rating_norm, rating_mean


# In[89]:


rating_norm, rating_mean = normalizeRatings(rating, record)


# In[91]:


rating_norm = np.nan_to_num(rating_norm)  #将nan转换为0
rating_mean = np.nan_to_num(rating_mean)


# In[92]:


rating_norm, rating_mean


# In[93]:


num_features = 10


# ##### 初始化电影内容矩阵X

# In[33]:


X_parameters = tf.Variable(tf.random_normal([movieNo, num_features], stddev=0.35))


# ##### 初始化用户喜好矩阵 Theta

# In[34]:


Theta_parameters = tf.Variable(tf.random_normal([userNo, num_features], stddev=0.35))


# In[35]:


# reduce_sum表示求和，matmul表示矩阵相乘，transpose_b表示对第二个参数进行转置
# -rating_norm表示与用户真实的评分值相减
# *record是因为要拟合的是评分的电影，对于没有评分的电影，计算结果用0代替
# 1/2 * (tf.reduce_sum(X_parameters**2) + tf.reduce_sum(Theta_parameters**2))是正则化项，其中λ设为1
loss = 1/2*tf.reduce_sum(((tf.matmul(X_parameters, Theta_parameters, transpose_b=True)-rating_norm)*record)**2)    + 1/2 * (tf.reduce_sum(X_parameters**2) + tf.reduce_sum(Theta_parameters**2))


# In[36]:


# 设置优化器
optimizer = tf.train.AdamOptimizer(1e-4)
train = optimizer.minimize(loss)


# # 第四步：训练模型

# In[37]:


tf.summary.scalar('loss', loss)


# In[38]:


# 将所有的summary信息汇总
summaryMerged = tf.summary.merge_all()


# In[39]:


filename = 'H:/Desktop/movie_tensorboard'


# In[40]:


#把信息保存到文件中
writer = tf.summary.FileWriter(filename)  


# In[41]:


sess = tf.Session()   #创建一个tf会话


# In[42]:


init = tf.global_variables_initializer()


# In[43]:


sess.run(init)


# In[44]:


for i in range(5000):
    # 将不重要的变量，即train的训练结果，保存在_中。
    # summaryMerged的训练结果保存在movies_summary中
    _, movies_summary = sess.run([train, summaryMerged])
    
    #将loss值随着迭代次数i的变化保存下来
    writer.add_summary(movies_summary, i)


#  # 第五步：评估模型

# In[45]:


Current_X_parameters, Current_Theta_parameters = sess.run([X_parameters, Theta_parameters])


# In[46]:


# dot用于矩阵之间的乘法操作
# 这一步得到完整的用户评分表， 保存在predicts中
predicts = np.dot(Current_X_parameters, Current_Theta_parameters.T) + rating_mean


# In[47]:


errors = np.sqrt(np.sum((predicts - rating)**2))


# In[48]:


errors


# # 第六步：构建完整的电影推荐系统

# In[49]:


user_id = input('您要向哪位用户进行推荐？请输入用户编号：')

# predicts[:, int(user_id)]表示推荐系统预测的该用户对所有电影的评分，上面已经求出
# argsort()表示将数据按从小到大排序，返回的是下标
# argsort()[::-1]表示从大到小排序
sortedResult = predicts[:, int(user_id)].argsort()[::-1]

idx = 0

# .center(80, '=')是让输出更加好看
# Python center() 返回一个原字符串居中,并使用填充字符填充至长度 width 的新字符串。默认填充字符为空格，此处指定为=
print('为该用户推荐的评分最高的20部电影是：'.center(80, '='))

for i in sortedResult:
    #  movies_df.iloc[i]['title']取第i行的title列的值，即电影名称
    print('评分：%.2f， 电影名：%s' %(predicts[i, int(user_id)], movies_df.iloc[i]['title']))
    idx +=1
    if idx == 20: break


# In[ ]:




