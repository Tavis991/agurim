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
        print (root)
        print (dirs)
        for dyr in dirs:
            if dyr.startswith('pix') :
                file = ''
                for f in files : 
                    if f[-4:] == '.csv': file = f
                table = gpd.read_file(os.path.join(root, file))
                table = table.drop(table.columns[3:-1], axis=1) #geomatry must be kept to stay GPD 
                table = table.rename(columns={'field_1': 'filename', 'field_2': 'LAT', 'field_3': 'LON'})
                table = table.set_index('filename')

                table['LAT'] = table['LAT'].apply(float)
                table['LON'] = table['LON'].apply(float)
                geomatry = table.apply(lambda x: utm.from_latlon(x.LAT, x.LON), axis=1)
                coordinates = np.array([np.array([ float (i) for i in x[:2]]) for x in geomatry.values])
                table['X'], table['Y'] = coordinates[:,0], coordinates[:,1] 
                centroids = [[743106.672644, 3.666385e+06], [742996.672644, 3.666385e+06], [743106.672644, 3.666385e+06]]

                clusters = clustering(coordinates, centroids)
                table['cluster'] = clusters.labels_
                table['avg_of_avgs'], table['water_tmp'], table['air_tmp'] = None, None, None, 
                shore_history = []

                for (branch, sub_dirs, tif_files) in os.walk(os.path.join(root, dyr), topdown=True):
                    for f in table.index:
                        if f in tif_files:
                            img = os.path.join(root, dyr, f)
                            avg_of_avgs, shore_bi, water_tmp = azr.analyis(img)
                            table.loc[f, 'avg_of_avgs'], table.loc[f, 'water_tmp'] = avg_of_avgs, water_tmp
                            shore_history += shore_bi
                        else : 
                            table.drop(f, inplace=True)
                    
                    print("this is branch", branch)
                    print(sub_dirs)
                    #print(tif_files)
                plt.figure()
                plt.hist(shore_history, bins=17)
                plt.pause(1)
                plt.show()
                print(table)    
        #print (files)
        print ('--------------------------------')
