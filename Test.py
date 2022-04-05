import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

#Read the files 
data=pd.read_csv('SP500_2010Q4.csv')
data2=pd.read_csv('SP500_constituents.csv')

#Convert 
for j in range(1,len(data.columns)):                                                #Do a loop on column (outer loop)
    for i in range(1,len(data)):                                                    #Do a loop on row (inner loop), 
            stock=(float(data.iloc[-i,j])-float(data.iloc[-i-1,j]))/(data.iloc[-i-1,j]) #Pass the value of the equation to variable stock, with i and j looped through the whole dataframe
            data.iloc[-i,j]=stock                                                   #Pass the value of stock to the indices of data
data.to_csv('convertedData.csv',index=False)                                        #Write a new file with the new data

#Read the converted file (we only do things with this file now)
convertData = pd.read_csv('convertedData.csv')                                      
convertData.set_index('time',inplace=True)                                   #Set time column as index                           
convertData=convertData.T                                                    #Switch index with header (in this case, time with company symbols)
convertFeatures = np.array(convertData.iloc[:,1:])                           #Create an array that only uses the converted data
Date=np.array(convertData.columns)                                           #Create an array with the column names (in this case, it's the date)

#Determine the number of cluster 
SSE = []
for iClus in range(1,31):
    km = KMeans(n_clusters=iClus, init = 'k-means++', max_iter =300, n_init = 10, random_state = 0)  
    km.fit(convertFeatures)  
    SSE.append(km.inertia_) 
plt.plot(np.arange(1,31),SSE,marker = "o")
plt.xlabel('Number of clusters')
plt.ylabel('Sum of sq distances')
plt.show()

###Plot
convertData=convertData.iloc[:,1:]      #Change the dataframe to the converted dataframes (because there is one line that is not converted)
kmeans = KMeans(n_clusters=5)           #Create clusters with the number of clusters found through the scree plot above
kmeans.fit(convertData)                 #fit Data to the convertData dataframe
y_kmeans =kmeans.predict(convertData)   #go through every elements in the row
Xcompany=10                             
Ycompany=11
plt.scatter(convertData.iloc[:, Xcompany], convertData.iloc[:,Ycompany], c=y_kmeans, s=20, marker="o")
centers = kmeans.cluster_centers_
plt.scatter(centers[:, Xcompany], centers[:, Ycompany], c='black', s=150, alpha=0.5)
plt.xlabel(Date[Xcompany])
plt.ylabel(Date[Ycompany])
plt.title('Clusters of the stock price')
plt.show()

#Create a new dataframe to show the information of the clusters
cluster_map=pd.DataFrame()
cluster_map['cluster']=kmeans.labels_                           #create a column called cluster that contains an array of cluster labels
cluster_map['Symbol']=convertData.index.values                  #Column that contains the index value of convertData dataframe
data2=data2.reindex(columns=["Symbol","Constituent"])           #Change the position of columns in the data2 dataframe (the one that has company names)
clustmerg=pd.merge(cluster_map,data2,how='left',on=['Symbol'])  #Use pandas merge to merge two datas based on the mutual Symbol column 

print("\nThere are 5 clusters total")                       
inputclust=input("\nWhich clusters do you want to look?: ")     #input the cluster 
print(clustmerg[clustmerg.cluster==int(inputclust)])            #print out the dataframes in which cluster column values only have the input number
