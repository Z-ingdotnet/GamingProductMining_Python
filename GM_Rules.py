import pyodbc 
import numpy as np
import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
import seaborn as sns

from efficient_apriori import apriori

dir(mlxtend.frequent_patterns)
import mlxtend
from mlxtend.preprocessing import TransactionEncoder
#from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import apriori, association_rules 
#from PyARMViz import PyARMViz
from datetime import timedelta
from sqlalchemy import create_engine
import time

import gc
gc.collect()


import arulesviz

#import imp
#imp.reload(mlxtend)


cnxn = pyodbc.connect('driver={Custom Defined Driver};server=125.84.184.**:8000,18000;database=mydb;trusted_connection=yes')  
cursor = cnxn.cursor()


chunksize = 10000
SQLReader  = pd.read_sql_query("select *,TimeLength/3600 Hours from Analysis.Ops_ProductPref",cnxn,chunksize=chunksize)
df = pd.concat(SQLReader, ignore_index=True)


for col in df.columns: 
    print(col)

list(df.columns) 
sorted(df) 

#df.columns.get_loc("x")


df["Hours2"]=df['TimeLength']/3600
df_sub=df[['Product','Hours2','DenomAmount','Area','Bank','TransId','CustomerID']]
df_sub[df_sub["Hours2"] > 0.5]


np.unique(df_sub[['Area']])
#array(['Alpha Lounge', 'Bacqu', 'Butterfly', 'Carnival', 'Casino Alley',
#       'Crystal', 'Down Town', 'Dragon', 'Dragon Corner', 'High Limit',
#       'High Limit Expansion', 'Live TG West', 'Mansion One', 'Phoenix',
#       'Platinum', 'Stadium', 'Supreme', 'Supreme 88',
#       'Supreme Expansion', 'Uptown'], dtype=object)
output=np.unique(df['DenomAmount'])
np.set_printoptions(suppress=True)
print(output)
#[  0.05   0.1    0.2    0.25   0.5    1.     2.     5.    10.   100.  ]

#machineproductbasket_Phoenix = ( df[(df['Area'] =="Phoenix") & (df['DenomAmount']==0.05)] 
#          .groupby(['CustomerID', 'Product'])['Hours2'] 
#          .sum()
#          .round(2)
#          .unstack()
#          .reset_index()
#          .fillna(0) 
#          .set_index('CustomerID')) 

del df['combinedid']

del x
del x2

df['combinedid']=df["CustomerID"].astype(str)+'_'+df["Gdate"].astype(str)
x=df[(df['Area'] =="Phoenix") & (df['DenomAmount']==0.05) & (df['Gdate']>='2019-09-01')& (df['Gdate'] <='2019-12-31')]

machineproductbasket_Phoenix = ( df[(df['Area'] =="Phoenix") & (df['DenomAmount']==0.05) & (df['Gdate']>='2019-09-01')& (df['Gdate'] <='2019-12-31')] 
          .groupby(['combinedid', 'Product'])['Hours2'] 
          .sum()
          .round(2)
          .unstack()
          .reset_index()
          .fillna(0) 
          .set_index('combinedid')) 

def hot_encode(x):
    if x <= 0:
        return 0
    if x > 0:
        return 1

basket_encoded = machineproductbasket_Phoenix.applymap(hot_encode) 
machineproductbasket_Phoenix = basket_encoded 



frq_items =apriori(machineproductbasket_Phoenix, min_support = 0.05, use_colnames= True,low_memory=True)


rfm.to_csv(r'X:\Users\IZ\cluster.csv')
frq_items.to_csv(r'X:\Users\IZ\frq_gameproduct_Phoenix.csv')
rules.to_csv(r'X:\Users\IZ\assocated_gameproduct_Phoenix_.5.csv')





# Collecting the inferred rules in a dataframe 
rules = association_rules(frq_items, metric ="lift", min_threshold = 1) 
rules = rules.sort_values(['confidence', 'lift'], ascending =[False, False]) 
print(rules.head()) 


dir(arulesviz)
import arulesviz
g = arulesviz.Arulesviz(machineproductbasket_Phoenix, 0.001, 0.3, 12, products_to_drop=[])
g.create_rules()
g.plot_graph(width=1800, directed=False, charge=-150, link_distance=20)




















def draw_graph(rules, rules_to_show):
  import networkx as nx  
  G1 = nx.DiGraph()
   
  color_map=[]
  N = 50
  colors = np.random.rand(N)    
  strs=['R0', 'R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8', 'R9', 'R10', 'R11']   
   
   
  for i in range (rules_to_show):      
    G1.add_nodes_from(["R"+str(i)])
    
     
    for a in rules.iloc[i][1]:
                
        G1.add_nodes_from([a])
        
        G1.add_edge(a, "R"+str(i), color=colors[i] , weight = 2)
       
    for c in rules.iloc[i][2]:
             
            G1.add_nodes_from()
            
            G1.add_edge("R"+str(i), c, color=colors[i],  weight=2)
 
  for node in G1:
       found_a_string = False
       for item in strs: 
           if node==item:
                found_a_string = True
       if found_a_string:
            color_map.append('yellow')
       else:
            color_map.append('green')       
 
 
   
  edges = G1.edges()
  colors = [G1[u][v]['color'] for u,v in edges]
  weights = [G1[u][v]['weight'] for u,v in edges]
 
  pos = nx.spring_layout(G1, k=16, scale=1)
  nx.draw(G1, pos, edges=edges, node_color = color_map, edge_color=colors, width=weights, font_size=16, with_labels=False)            
   
  for p in pos:  # raise text positions
           pos[p][1] += 0.07
  nx.draw_networkx_labels(G1, pos)
  plt.show()
