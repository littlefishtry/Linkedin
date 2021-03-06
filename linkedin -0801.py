#!/usr/bin/env python
# coding: utf-8

# """
# 1.较上一版本更新的代码格式如下，重点关注update部分即可。
# #----------------------------update---------------------------------------------
#     ......
# #----------------------------update---------------------------------------------
# 
# 2.遗留的两个问题已解决：
# ①新建连接数量与网络密度变化趋势变化一致的问题：经验证，是误用nx.single_source_shortest_path_length()函数导致的，使得统计的建立联系数量
# 为累计值而非增量，已更正为增量，具体见relationship函数中的update部分。
# ②新增联系的数量变化趋势为持续下降：修改代码后，当前变化趋势为先增加，后减少至趋于0，减少的原因在于PageRank算法的应用使得网络中存在较多
# 孤立的节点集，“圈子”与“圈子”之间没有桥接关系，因此每个节点只能与有限个节点建立连边。
# 
# 3.新增逻辑：
# ①pagerank的随机跳转概率值γ=0.85
# ②企业招聘门槛：生成5个服从正态分布且落在（0，1）区间内的概率值，对其由小到大排序后赋值给规模由小到大的五个企业
# ③新增联系的节点范围，由原来的只允许【求职者-意向企业在职者】之间建立联系，变更为允许所有满足三度连接机制的节点对建立联系，不过其他的节点对建立联系的概率为【求职者-意向企业在职者】建立联系概率的20%（随机取的一个较小的值）。
# 
# 4.待定更新（下次研讨）：
# 将一次实验的演化轮次由50次缩短到20次，有以下两点原因：
# ①20轮左右，新增连边的数量减少至0；
# ②20轮左右，各企业空岗的数量达到最低值，人才市场的人岗匹配率达到最高。
# 演化周期缩短，是否需要假设市场稳定：①人才市场无新进入者；②离职后人力资本保持不变。
# """

# In[1]:


import networkx as nx
import pandas as pd
import numpy as np
import random 
import matplotlib.pyplot as plt


# ## 构建初始网络

# ### 抽取子图

# In[2]:


#基于数据集生成子图（点+边）
def sub_graph():
    data=pd.read_csv("D:/pythonwork/edge.csv")   #从csv读取连边
    res=[[] for i in range(6726290)] #创建二维列表，6726290为节点总数
    for line in data.values:
        res[line[0]].append(line[1])  #根据data中的数据，res中存储结构为：res[节点编号]=节点邻居列表
        res[line[1]].append(line[0])
    dct={i:res[i] for i in range(len(res))}  #数据类型转换，定义邻居字典，格式为 节点：邻居节点列表
    allnodes=[i for i in range(6726290)]  #定义节点列表
    allnodes_set=set(allnodes)  #将列表转换为集合，以提高运行效率
    
    nodes_selected_set=set()  #用于存储被抽取的节点的集合
    count=0   #计数，判断抽取节点个数是否达到上限
    α=0.5 #调节回跳概率
    β=0.5 #调节DFS的概率
    #----------------------------update---------新增 设置随机跳转概率------------------------------------
    γ=0.85 #pagerank,即为γ的概率按照链接跳转，1-γ的概率随机跳转
    #----------------------------update---------新增 设置随机跳转概率------------------------------------
    node_father=random.choice(allnodes) #随机选择开始的节点node_father，作为开始的节点
    nodes_selected_set.add(node_father)  
    count=count+1
    fa_neib_nodes_set=set(dct[node_father]) #父节点的邻域节点，即从dct中读出node_father的邻居列表
    print("【计数，节点number，节点邻居列表】")
    print(count,node_father,fa_neib_nodes_set)

    node_cur=random.choice(dct[node_father]) #从父节点的邻居列表中随机抽取下一步节点，作为当前节点node_cur
    nodes_selected_set.add(node_cur)  
    count=count+1
    cur_neib_nodes_set=set(dct[node_cur])
    print(count,node_cur,cur_neib_nodes_set)

    while count<2000:
        rate_rankpage=np.random.random()
        if rate_rankpage<=γ and cur_neib_nodes_set:
            pro_lst=[None]*len(dct[node_cur])
            for i in cur_neib_nodes_set:
                index1=dct[node_cur].index(i)
                if i in fa_neib_nodes_set: #当前节点和上级节点的共同节点，概率为1
                    pro_lst[index1]=1
                elif i==node_father: #回溯上级节点，概率为α
                    pro_lst[index1]=α
                else:                #仅为当前节点的邻居节点，即距离上级节点的最短路径为2，概率为β
                    pro_lst[index1]=β
            total_pro=sum(pro_lst)
            for j in range(len(pro_lst)-1):#概率列表归一化
                pro_lst[j]=pro_lst[j]/total_pro
            pro_lst[len(pro_lst)-1]=1-sum(pro_lst[:len(pro_lst)-1])

            node_cand=np.random.choice(list(dct[node_cur]),p=pro_lst)  #选择下一个节点
        else:
            node_cand=np.random.choice(allnodes)

        if node_cand not in nodes_selected_set: #如果节点未重复取到，添加到抽取节点列表
            nodes_selected_set.add(node_cand)
            count=count+1
            print(count,node_cand,dct[node_cand])
        #更新迭代变量
        node_father=node_cur 
        fa_neib_nodes_set=set(dct[node_father]) 
        node_cur=node_cand
        cur_neib_nodes_set=set(dct[node_cur])

    tmp=np.array(list(nodes_selected_set))
    np.save('selected.npy',tmp) 
    a=np.load('selected.npy')
    a=a.tolist()
    G=nx.Graph()
    data=pd.read_csv("d:/pythonwork/edge.csv")   #读取边集合，用于后续遍历
    eset=set()
    data1=np.array(data)
    for i in data1:
        eset.add(tuple(i))  #处理边的数据结构，将其转化为（i,j）元素组成的集合        
    enodes=set(a)  #转化为集合，遍历速度比list快得多
    edge_part=set() #存储点集中两点之间的联系
    for i in enodes:
        for j in enodes:
            if tuple((i,j)) in eset or tuple((i,j)) in eset:
                edge_part.add(tuple((min(i,j),max(i,j))))
    print("【节点总数，连边总数】")
    print(len(enodes),len(edge_part))
    G.add_nodes_from(list(a)) #将点和边用于创建图G
    G.add_edges_from(list(edge_part))
    #nx.draw(G,node_size=10)  
    return G


# ### 赋予属性 

# In[3]:


#定义节点类，即通过多智能体为每个个体赋予属性和改变属性的方法
class ini_nodes: 
    def __init__(self,name):  #构造函数
        self.school=0  #学校属性，固定不变
        self.enterprise=0  #企业属性，求职成功、离职时改变
        self.hr_ability=np.clip(np.random.normal(0.5,0.167),0,1)  #人力资本，求职的能力，创建对象时随机生成，整体呈正态分布,截断到（0，1）
        self.apply_rate=[0.1,0.3,0.5,0.7,0.9]  #初始投递意向（未添加社会资本的影响）
        self.apply=[False,False,False,False,False]  #是否发生投递行为
        self.relations=[0,0,0,0,0]  #分别与五家企业内在职者的联系条数
        self.name=name
    def schooldef(self,school):   #修改学校属性
        self.school=school
    def enterprisedef(self,enterprise):  #修改企业属性 
        self.enterprise=enterprise
    def apply_change(self,i,bool):  #修改单个投递行为
        self.apply[i]=bool
    def relations_change(self,i,num):  #修改单个联系条数
        self.relations[i]=num   
    def apply_rate_change(self):  #基于联系条数重新计算单个投递意向
        for i in range(5):
            self.apply_rate[i]=round(min(1,float(self.apply_rate[i])* (2-((1-0.1)**int(self.relations[i])))),2)  #会员闭包的影响暂取0.1
    #定义初始化函数
    def apply_to_ini(self):  #投递行为初始化
        self.apply=[False,False,False,False,False]
    def relations_to_zero(self):  #联系条数初始化
        self.relations=[0,0,0,0,0]
    def apply_rate_to_ini(self):   #投递意向初始化
        self.apply_rate=[0.1,0.3,0.5,0.7,0.9]


# In[4]:


#创建2000个同名节点对象，其中求职者占23%（jobseeker）和在职者占77%（employee），并分别为其赋予学校、企业属性
def nodes_classify(G1):
    G=nx.Graph()
    G=G1
    nodes_list=G.nodes()#获取图的节点列表
    jobseeker=random.sample(nodes_list,int(0.23 * 2000)) #随机抽取23%作为求职者
    employee=[]
    for i in nodes_list:  #其他的加入在职者列表
        if i not in jobseeker:
            employee.append(i)
    #创建初始节点对象，将创建好的对象按照节点编号归到两个list中
    obj_list_jobseeker=[]  #求职者对象列表
    obj_list_employee=[]   #在职者对象列表
    for i in jobseeker:
        i=ini_nodes(i)
        obj_list_jobseeker.append(i)
    for j in employee:
        j=ini_nodes(j)
        obj_list_employee.append(j)

    #分配学校、企业属性：
    #1.学校：把包含所有节点的列表打乱，然后取切片，切片比例为1:1:1:1:1
    obj_list=obj_list_employee+obj_list_jobseeker
    random.shuffle(obj_list)
    school_N=[]
    a=int(len(obj_list)/5)
    school_N.append(obj_list[:a])
    school_N.append(obj_list[a:2*a])
    school_N.append(obj_list[2*a:3*a])
    school_N.append(obj_list[3*a:4*a])
    school_N.append(obj_list[4*a:])
    for item in school_N:
        for obj in item:
            obj.schooldef(school_N.index(item)+1) #将对象所处列表的索引值+1作为学校属性赋给对象
    #2.企业：将在职者的节点打乱，然后取切片,切片比例为1:3:5:7:9
    random.shuffle(obj_list_employee)
    employee_N=[]
    b=int(len(obj_list_employee)/25)
    employee_N.append(obj_list_employee[:b])
    employee_N.append(obj_list_employee[b:4*b])
    employee_N.append(obj_list_employee[4*b:9*b])
    employee_N.append(obj_list_employee[9*b:16*b])
    employee_N.append(obj_list_employee[16*b:])
    for item in employee_N:
        for obj in item:
            obj.enterprisedef(employee_N.index(item)+1) #将在职者对象所处列表的索引值+1作为企业属性赋给对象
    return obj_list_jobseeker,obj_list_employee,employee_N


# In[5]:


#构建初始网络（调用函数）
#np.random.seed(1)
G=sub_graph() #创建子图
#np.random.seed(2)
obj_list_jobseeker,obj_list_employee,employee_N=nodes_classify(G) #创建节点同名对象，为求职者、在职者赋予属性
#定义全局变量
job_num=len(obj_list_jobseeker) #定义待招岗位的数量，若接收则该岗位数量-1，若离职则该岗位数量+1
job_num_list=[int(job_num/25),int(3*job_num/25),int(5*job_num/25),int(7*job_num/25),int(9*job_num/25)]  #按照切片比例1:3:5:7:9分配给五家公司
exit_rate=0.188  #离职率
#记录新增联系、新增各企业在职者数量、新增离职者数量
new_relation=[]
new_employed=[]
new_exited=[]


# ## 人员流动

# ### 投递

# In[6]:


#函数：更新求职对象的投递概率，将基于每个公司在职的好友联系数量更新
def cal_apply_rate(): 
    #print("【节点number，节点与五个公司在职者连边的数量，节点对五家企业的投递概率】")
    ini=[0.1,0.3,0.5,0.7,0.9]
    indicator1=[]
    for obj in obj_list_jobseeker:
        obj.relations_to_zero() #先对每个求职者的联系数量进行初始化
        friends=list(G.neighbors(obj.name))
        for item in obj_list_employee:  #对每个求职者进行遍历，如果邻居为求职者，则该企业的relations＋1
            if item.name in friends:
                obj.relations_change(int(item.enterprise)-1,obj.relations[int(item.enterprise)-1]+1)
        obj.apply_rate_change()   #根据relations重新计算apply_rate
        lst1=[]
        for i in range(5):
            res=round((obj.apply_rate[i]-ini[i])/ini[i],2)
            lst1.append(res)
        indicator1.append(lst1)
        #print(obj.name,obj.relations,obj.apply_rate,lst1)
    return indicator1
    #print(indicator1)


# In[7]:


#函数：基于apply_rate判断投递与否
def apply():
    ini=[0.1,0.3,0.5,0.7,0.9]
    count_apply=0
    indicator2_df=pd.DataFrame()
   # print("【节点number，节点是否投递五家企业】")
    for obj in obj_list_jobseeker:
        obj.apply_to_ini()  #初始化apply列表，将值均变为false
        lst2=[0,0,0,0,0]
        for i in range(5):   #求职者针对每家企业随机产生概率p1，p1<apply_rate则投递
            p1=np.random.random()
            if p1<= obj.apply_rate[i]:
                obj.apply_change(i,True)
                count_apply=count_apply+1
            if ini[i]< p1 <= obj.apply_rate[i]:
                lst2[i]=1
        indicator2_df[obj.name]=lst2
       # print(obj.name,obj.apply)
    return indicator2_df,count_apply


# ### 接收

# In[8]:


#模拟企业接收流程
def accept(obj_list_jobseeker,obj_list_employee,job_num_list,employee_N):
    successed=[[],[],[],[],[]] #用于记录各公司本轮录用的求职者
    obj_list_jobseeker_set=set(obj_list_jobseeker) 
    enter_list=[0,1,2,3,4]
     #----------------------------update---------新增 生成企业招聘门槛的策略更新------------------------------------
    accept_rate_lst=np.random.normal(0.4,0.167,5) #企业招聘门槛：生成5个服从正态分布且落在（0，1）区间内的概率值，对其由小到大排序后赋值给规模由小到大的五个企业
    accept_rate_lst=np.clip(accept_rate_lst,0,1)
    accept_rate_lst=sorted(accept_rate_lst)  
    #----------------------------update---------新增 生成企业招聘门槛的策略更新------------------------------------
    random.shuffle(enter_list)  #将公司的顺序打乱
    for i in enter_list:  #计数，计算申请该企业的总人数，用于计算企业的招聘门槛，后续需优化算法
        accept_rate=accept_rate_lst[i]
        print("【企业编号：",i+1,"，招聘门槛：",accept_rate,"】")
        applyed=set()
        for obj in obj_list_jobseeker_set:
            if obj.apply[i]==True:
                applyed.add(obj)
        #accept_rate=round((len(applyed)/len(obj_list_jobseeker))**2,4)  #企业的招聘门槛（待定，暂用（申请人数/总人数）**2，取4位小数，企业5招聘门槛仍然过高）
        
        candidate=[] #候选人列表：将投递该企业并且满足企业的招聘门槛的求职者加入列表
        for obj in applyed:   #按照乱序，让5个公司挨个进行录用流程
            if accept_rate <= obj.hr_ability:
                candidate.append(obj)
        candidate.sort(key=lambda x:x.hr_ability,reverse=True) #根据人力资本对候选人进行降序排序
        #根据Hospital-Resident算法，对候选人列表进行邀请：1.无offer的直接录用；2.已有offer不如i公司，放弃原公司，被i公司录用；3.已有offer更好，不发offer
        for cand in candidate:
            if job_num_list[i]>0:
                if cand.enterprise==0:
                    cand.enterprisedef(i+1)
                    job_num_list[i]=job_num_list[i]-1  #本公司招聘数量-1
                    #print(cand.name,i+1,accept_rate,cand.hr_ability)
                elif cand.enterprise!=0 and cand.apply_rate[i]>cand.apply_rate[int(cand.enterprise)-1]:
                    job_num_list[int(cand.enterprise)-1]=job_num_list[int(cand.enterprise)-1]+1  #所放弃的原录用公司招聘数量+1
                    cand.enterprisedef(i+1)
                    job_num_list[i]=job_num_list[i]-1  #本公司招聘数量-1
                    #print(cand.name,i+1,accept_rate,cand.hr_ability)    
            else:
                break
                    
    #计算本轮录用结果：obj_list_jobseeker中enterprise不为0的求职者即为求职成功，将其从求职者列表中删除，加入在职者列表   
    obj_list_jobseeker1=[]
    for item in obj_list_jobseeker:
        if item.enterprise!=0:
            employee_N[int(item.enterprise)-1].append(item) 
            successed[int(item.enterprise)-1].append(item)
            obj_list_employee.append(item)
        else:
            obj_list_jobseeker1.append(item)
    obj_list_jobseeker=obj_list_jobseeker1
    new_employed.append([len(successed[0]),len(successed[1]),len(successed[2]),len(successed[3]),len(successed[4])]) #记录本轮成功求职的总人数
    print("各企业成功招聘人数：",new_employed)
    return obj_list_jobseeker,obj_list_employee,job_num_list,employee_N


# ### 离职

# In[9]:


#模拟离职流程
def departure(employee_N,obj_list_employee,obj_list_jobseeker,job_num_list):
    exited=[[],[],[],[],[]]
    for i in range(5): # 5个公司依次模拟离职
        for cand in employee_N[i]:
            p2=np.random.normal(0.5,0.167)
            if p2 <= exit_rate:
                cand.enterprisedef(0)
                employee_N[i].remove(cand) 
                obj_list_employee.remove(cand)
                obj_list_jobseeker.append(cand)
                job_num_list[i]=job_num_list[i]+1
                exited[i].append(cand)
    new_exited.append([len(exited[0]),len(exited[1]),len(exited[2]),len(exited[3]),len(exited[4])])  #记录本轮离职的人数
   # print("各企业离职人数：",new_exited)    
    return employee_N,obj_list_employee,obj_list_jobseeker,job_num_list


# ## 新增人际联系 

# In[10]:


#函数：新增人际联系
def relationship(num1,num2): #传入2个0，1变量，num1控制是否允许三度连接，num2控制是否社团闭包
    enable_3_con=int(num1)
    enable_club=int(num2)
    other_rel_rate=0.2  #除了求职者-在职者节点对建立联系
    new_relation_1=[] #记录本轮新增连接
    for node1 in obj_list_jobseeker:#对每一个求职者对象，寻找其二度、三度好友列表，分别存储在degree_nodes_2、degree_nodes_3
        #----------------------------update------------更正 找出符合三度建联机制的节点对象---------------------------------
        #
        #degree_nodes_2=list(nx.single_source_shortest_path_length(G,node1.name,2))
        #degree_nodes_3=list(nx.single_source_shortest_path_length(G,node1.name,3))
        degree_nodes_2=[]
        degree_nodes_3=[]
        source_path_lengths = nx.single_source_shortest_path_length(G,node1.name,3)
        for v,l in source_path_lengths.items():
            if l == 2:
                degree_nodes_2.append(v)
            elif l == 3:
                degree_nodes_3.append(v)
        #----------------------------update------------更正 找出符合三度建联机制的节点对象---------------------------------
        for i in range(5):
            for node2 in employee_N[i]: #对每个企业中的在职者，在二度、三度好友列表中，则可能新增联系
                if node2.name in degree_nodes_2: #2度好友
                    p3=np.random.random()
                    n=len(list(nx.common_neighbors(G,node1.name,node2.name))) #n为共同好友的数量，三元闭包的影响暂取0.3，多个共同好友时为[1-(1-0.3)**n]
                    if node1.school==node2.school: #两节点若为校友关系，则考虑社团闭包，影响暂取0.2
                        if p3<=min(1,((1-(1-0.3)**n)+0.2*enable_club-(1-(1-0.3)**n)*0.2*enable_club)):  #三元闭包+社团闭包-三元闭包*社团闭包
                            #G.add_edge(node1.name,node2.name)  #新增联系
                            new_relation_1.append((node1.name,node2.name))
                    else:
                        if p3<=min(1,(1-(1-0.3)**n)): #非校友时，仅考虑三元闭包
                            #G.add_edge(node1,node2)
                            new_relation_1.append((node1.name,node2.name))       
                elif node2.name in degree_nodes_3:#3度好友，三度连接影响暂取0.2，不受中间好友影响。
                    p3=np.random.random()
                    if node1.school==node2.school:
                        if p3<=min(1,(0.2*enable_3_con+0.2*enable_club-0.2*enable_3_con*0.2*enable_club)): #校友关系，三度连接+社团闭包-三度连接*社团闭包
                            #G.add_edge(node1.name,node2.name)
                            new_relation_1.append((node1.name,node2.name))
                    else:
                        if p3<=min(1,0.2*enable_3_con): #非校友时，仅考虑三度连接
                            #G.add_edge(node1.name,node2.name)
                            new_relation_1.append((node1.name,node2.name))
                            
        #----------------------------update----新增 允许其他节点对建联-----------------------------------------                    
        for node2 in obj_list_jobseeker:
            if node2.name in degree_nodes_2: #2度好友
                    p3=np.random.random()
                    n=len(list(nx.common_neighbors(G,node1.name,node2.name))) #n为共同好友的数量，三元闭包的影响暂取0.3，多个共同好友时为[1-(1-0.3)**n]
                    if p3<=(other_rel_rate * min(1,(1-(1-0.3)**n))): #非校友时，仅考虑三元闭包
                            new_relation_1.append((node1.name,node2.name))     
            elif node2.name in degree_nodes_3:#3度好友，三度连接影响暂取0.2，不受中间好友影响。
                    p3=np.random.random()
                    if p3<=(other_rel_rate* min(1,0.2*enable_3_con)): #非校友时，仅考虑三度连接
                            new_relation_1.append((node1.name,node2.name))
            
    for node1 in obj_list_employee:
        degree_nodes_2=[]
        degree_nodes_3=[]
        source_path_lengths = nx.single_source_shortest_path_length(G,node1.name,3)
        for v,l in source_path_lengths.items():
            if l == 2:
                degree_nodes_2.append(v)
            elif l == 3:
                degree_nodes_3.append(v)
        for node2 in obj_list_employee:
            if node2.name in degree_nodes_2: #2度好友
                    p3=np.random.random()
                    n=len(list(nx.common_neighbors(G,node1.name,node2.name))) #n为共同好友的数量，三元闭包的影响暂取0.3，多个共同好友时为[1-(1-0.3)**n]
                    if p3<=(other_rel_rate * min(1,(1-(1-0.3)**n))): #非校友时，仅考虑三元闭包
                            new_relation_1.append((node1.name,node2.name))     
            elif node2.name in degree_nodes_3:#3度好友，三度连接影响暂取0.2，不受中间好友影响。
                    p3=np.random.random()
                    if p3<=(other_rel_rate* min(1,0.2*enable_3_con)): #非校友时，仅考虑三度连接
                            new_relation_1.append((node1.name,node2.name))
        #----------------------------update-------------新增 允许其他节点对建联--------------------------------      
        
    #new_relation.append(len(new_relation_1)) #记录本轮新增关系数量
    #print("【新增联系总数，罗列新增联系】")
    #print(len(new_relation))
    return new_relation_1


# ## 主函数

# In[11]:


count_circle=0
count_effected_list=[]
aver_affect_rate_list=[]
aver_rise_rate_list=[]
count_effected2_list=[]
aver_affect_rate2_list=[]
totle_effect_list=[]
aver_rise_rate2_list=[]
new_relation_list=[]
increase_relation=[]
density_list=[]
job_list=[[],[],[],[],[]]
#----------------------------update----待定修改 演化轮次50→20-----------------------------------------
while count_circle <20:
#----------------------------update----待定修改 演化轮次50→20-----------------------------------------
    print("【count=",count_circle,"】")
    indicator1=cal_apply_rate() #计算投递概率
    indicator2_df,count_apply=apply()  #模拟投递行为
    obj_list_jobseeker,obj_list_employee,job_num_list,employee_N=accept(obj_list_jobseeker,obj_list_employee,job_num_list,employee_N)  #模拟录取行为
    employee_N,obj_list_employee,obj_list_jobseeker,job_num_list=departure(employee_N,obj_list_employee,obj_list_jobseeker,job_num_list)  #模拟离职行为
    job_list[0].append(job_num_list[0])
    job_list[1].append(job_num_list[1])
    job_list[2].append(job_num_list[2])
    job_list[3].append(job_num_list[3])
    job_list[4].append(job_num_list[4])
    #量化对求职意愿的影响
    indicator1_df=pd.DataFrame(indicator1)
    rise_rate1=[]
    count_effected=0
    for i in range(5):
        rise_rate1.append(round(sum(indicator1_df[i])/len(indicator1_df),4))
    for j in range(len(indicator1_df)):
        if sum(indicator1_df.iloc[j]) != 0:
            count_effected=count_effected+1
    aver_affect_rate=round(count_effected/len(indicator1_df),4)
    aver_rise_rate=round(sum(rise_rate1)/5,4)
    count_effected_list.append(count_effected)
    aver_affect_rate_list.append(aver_affect_rate)
    aver_rise_rate_list.append(aver_rise_rate)

    #量化对投递决策的影响
    count_effected2=0
    totle_effect=0
    for k in list(indicator2_df.columns):
        totle_effect=totle_effect+sum(indicator2_df[k])
        if sum(indicator2_df[k]) != 0:
            count_effected2=count_effected2+1
    aver_affect_rate2=round(count_effected2/len(indicator2_df.columns),4)
    aver_rise_rate2=round(totle_effect/count_apply,4)
    count_effected2_list.append(count_effected2)
    aver_affect_rate2_list.append(aver_affect_rate2)
    totle_effect_list.append(totle_effect)
    aver_rise_rate2_list.append(aver_rise_rate2)
    every_new_relation=[]
    new_relation_4=relationship(0,0) 
    every_new_relation.append(len(new_relation_4))
    new_relation_3=relationship(0,1) 
    every_new_relation.append(len(new_relation_3))
    new_relation_2=relationship(1,0) 
    every_new_relation.append(len(new_relation_2))
    new_relation_1=relationship(1,1) 
    every_new_relation.append(len(new_relation_1))
    new_relation_list.append(every_new_relation)
    incre_relation=[]
    incre_relation.append(round((every_new_relation[1]-every_new_relation[0])/every_new_relation[0],4))
    incre_relation.append(round((every_new_relation[2]-every_new_relation[0])/every_new_relation[0],4))
    incre_relation.append(round((every_new_relation[3]-every_new_relation[0])/every_new_relation[0],4))
    increase_relation.append(incre_relation)
    
    density_list.append(nx.density(G))
    new_relation.append(len(new_relation_1))
    print(new_relation_1)
    G.add_edges_from(list(new_relation_1))
    count_circle=count_circle+1  


# ## 相关指标

# In[12]:


#会员闭包影响投递过程的情况
print("每轮演化中投递意愿受影响的求职者人数：",count_effected_list)
print("每轮演化中投递意愿受影响的求职者占比：",aver_affect_rate_list)
print("每轮演化中投递意愿增加率：",aver_rise_rate_list)


# In[20]:


#剩余求职人数
print("剩余求职人数为:",len(obj_list_jobseeker))
print("剩余岗位列表为:",job_num_list)
#每个企业的空岗数量变化情况
plt.clf()
x=list(range(20))
for i in range(5):
    y=job_list[i]
    plt.plot(x,y,label=i+1)
    plt.legend()


# In[21]:


#从剩下的求职者列表中随机选择1个节点，绘制其三度内好友的子图（可多次重复执行）
item1=random.choice(obj_list_jobseeker)
dt1=nx.single_source_shortest_path_length(G,item1.name,3) 
G1=nx.subgraph(G,list(dt1.keys()))
nx.draw(G1)
print("网络密度为：",nx.density(G1))


# In[22]:


#统计剩余求职者与其三度内好友节点的连接情况，子图密度几乎都为1，即邻域内全连接。
density_sat=[]
for item in obj_list_jobseeker:
    G1=nx.Graph()
    dt2=nx.single_source_shortest_path_length(G,item.name,3) 
    G1=nx.subgraph(G,list(dt2.keys()))
    density_sat.append(nx.density(G1))
print("剩余求职者的三度内节点子图密度统计：",pd.value_counts(density_sat))


# In[23]:


#量化对投递决策的影响
print("每轮演化投递决策受到影响的人数为：",count_effected2_list)
print("每轮演化投递决策受到影响的比例为：",aver_affect_rate2_list)
print("每轮演化受到影响的投递总次数为：",totle_effect_list)
print("每轮演化受到影响的投递总占比为：",aver_rise_rate2_list)


# In[24]:


#每轮演化新增联系数量
plt.clf()
y=[]
for i in range(1,len(new_relation)):
    y.append(new_relation[i])
x=list(range(len(y)))
plt.plot(x,y)


# In[25]:


#不同设定下新增联系数量的演化情况:0:无三度连接+无社团闭包；1：无三度连接+有社团闭包；2：有三度连接+无社团闭包；3：有三度连接+有社团闭包
plt.clf()
x=list(range(20))
for i in range(4):
    lst=[]
    for j in range(20):
        lst.append(new_relation_list[j][i])
    y=lst
    print(i,y)
    plt.plot(x,y,label=i)
    plt.legend()


# In[26]:


#每轮演化网络密度变化
plt.clf()
y=density_list
x=list(range(len(density_list)))
plt.plot(x,y)


# In[27]:


#剩余求职者列表
for i in obj_list_jobseeker:
    print(i.name,i.apply,i.hr_ability)

