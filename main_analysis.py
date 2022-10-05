import os
import geopandas as gpd
import utm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np 
from sklearn.cluster import KMeans
import project_h as azr 

def clustering (features, centroids):
    clusters = KMeans(
                    init=centroids,
                    n_clusters=len(centroids),
                    n_init=1,
                    max_iter=300,
                    random_state=42
                )
    return clusters.fit(features)


for (root,dirs,files) in os.walk('Agurim', topdown=True):
        # print (root)
        # print (dirs)
        for dyr in dirs:
            if dyr.startswith('pix') :
                file = ''
                for f in files : 
                    if f[-4:] == '.csv': file = f
                table = gpd.read_file(os.path.join(root, file))
                table = table.drop(table.columns[3:-1], axis=1) #geomatry must be kept for table to stay GPD 
                table = table.rename(columns={'field_1': 'filename', 'field_2': 'LAT', 'field_3': 'LON'})
                table = table.set_index('filename')

                table['LAT'] = table['LAT'].apply(float)
                table['LON'] = table['LON'].apply(float)
                geomatry = table.apply(lambda x: utm.from_latlon(x.LAT, x.LON), axis=1)
                coordinates = np.array([np.array([ float (i) for i in x[:2]]) for x in geomatry.values]) #from_latlon returns more values tha needed
                table['X'], table['Y'] = coordinates[:,0], coordinates[:,1] 
                table = table.drop(table.columns[0:2], axis=1)
                centroids = {0: [743106.672644, 3.666385e+06], 1: [742996.672644, 3.666385e+06], 2: [743106.672644, 3.666385e+06]}

                clusters = clustering(coordinates, list(centroids.values()))
                table['Cluster'] = clusters.labels_
                table['Average_distance_to_neighbors'], table['Average_distance_STD'], table['Water_temperature'], \
                    table['Air_temperature'] = None, None, None, None
                shore_history = {i : [] for i in centroids.keys()} 

                for (branch, sub_dirs, tif_files) in os.walk(os.path.join(root, dyr), topdown=True):
                    for f in table.index:
                        if f in tif_files:
                            img = os.path.join(root, dyr, f)
                            Average_distance_to_neighbors, Average_std, shore_bi, Water_temperature = azr.analyis(img)
                            table.loc[f, 'Average_distance_to_neighbors'], table.loc[f, 'Water_temperature'] \
                                = Average_distance_to_neighbors.round(3), Water_temperature.round(3)
                            table.loc[f, 'Average_distance_STD'] = Average_std.round(3)
                            shore_history[table.loc[f,'Cluster']] += shore_bi
                        else : 
                            table.drop(f, inplace=True)

                plt.figure()
                for cluster in centroids.keys():
                    if shore_history[cluster] :
                        b = np.histogram(shore_history[cluster], np.linspace(np.min(shore_history[cluster]), np.max(shore_history[cluster]), 20))
                        y = 100 * b[0] / np.sum(b[0])
                        plt.bar(b[1][:-1], y, width=8, color='blue')
                        plt.xlabel("Distance to shore")
                        plt.ylabel("Crane precentage")
                        plt.title(f"Histogram of shore distance for cluster {cluster}")
                        plt.pause(1)
                        plt.show()
                    else :
                        plt.plot(0,0)
                        plt.xlabel("Distance to shore")
                        plt.ylabel("Crane precentage")
                        plt.title(f"No photographed cranes on shoreline for cluster {cluster}")
                        plt.pause(1)
                        plt.show()

                table = table.round(3)
                pd.options.display.float_format = "{:,.3f}".format
                print(table) 
                table.to_excel('Final_results.xlsx')   
        #print (files)
        print ('--------------------------------')
