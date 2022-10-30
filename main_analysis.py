import os
import geopandas as gpd
import utm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np 
from sklearn.cluster import KMeans
import project_h as azr 
import json


def clustering (features, centroids):
    clusters = KMeans(
                    init=centroids,
                    n_clusters=len(centroids),
                    n_init=1,
                    max_iter=300,
                    random_state=42
                )
    return clusters.fit(features)


for dyr in os.listdir('Agurim'):
    final_cols = ['filename', 'x', 'y', 'Cluster', 'cluster_size', 'Average_distance_to_neighbors',\
         'Average_distance_STD', 'Water_temperature']

    for f in os.listdir(os.path.join('Agurim', dyr)):
        date_table = pd.DataFrame(columns=final_cols) 
        if f[-4:] == '.csv': 
            md_clusters = pd.read_csv(os.path.join('Agurim', dyr, f), delimiter='\t') 
            centroids = {i : [md_clusters.iloc[i]['x'], md_clusters.iloc[i]['y']] for i in range(len(md_clusters)) }
            shore_history = {i : [] for i in centroids.keys()} 
        date_table = date_table.set_index('filename')
    for f in os.listdir(os.path.join('Agurim', dyr)): 

        if 'leg' in f :
            files = os.listdir(os.path.join('Agurim', dyr, f))
            table_leg = None
            for file in files :
                if file[-4:] == '.csv': 
                    table_leg = pd.read_csv(os.path.join('Agurim', dyr, f, file), header=None)
            for file in files : 
                if 'pix' in file :
                    path = os.path.join('Agurim', dyr, f, file)
                    table_leg = table_leg.drop(table_leg.columns[3:], axis=1) 
                    table_leg = table_leg.rename(columns={0: 'filename', 1: 'LAT', 2: 'LON'})
                    table_leg = table_leg.set_index('filename')

                    table_leg['LAT'] = table_leg['LAT'].apply(float)
                    table_leg['LON'] = table_leg['LON'].apply(float)
                    geomatry = table_leg.apply(lambda x: utm.from_latlon(x.LAT, x.LON), axis=1)
                    coordinates = np.array([np.array([ float (i) for i in x[:2]]) for x in geomatry.values]) #from_latlon returns more values tha needed
                    table_leg['x'], table_leg['y'] = coordinates[:,0], coordinates[:,1] 
                    table_leg = table_leg.drop(table_leg.columns[0:2], axis=1)

                    clusters = clustering(coordinates, list(centroids.values()))
                    table_leg['Cluster'] = clusters.labels_
                    # for i in range (len(md_clusters)) :
                    #     table_leg[table_leg[table_leg['Cluster']] == i] = md_clusters.iloc[i]['cluster_size']

                    table_leg['Average_distance_to_neighbors'], table_leg['Average_distance_STD'], table_leg['Water_temperature'], \
                        table_leg['Air_temperature'], table_leg['wind_magnitude'], \
                        table_leg['moon_lux_daily'], table_leg['moon_lux_real']    = None, None, None, \
                            md_clusters.iloc[0]['air_temperature'], md_clusters.iloc[0]['wind_magnitude'], \
                            md_clusters.iloc[0]['moon_lux_daily'], md_clusters.iloc[0]['moon_lux_real'] 
                    
                    
                    for (branch, sub_dirs, tif_files) in os.walk(path):
                        for f in table_leg.index:
                            if f in tif_files:
                                img = path + '/' + f
                                Average_distance_to_neighbors, Average_std, shore_bi, Water_temperature = azr.analyis(img)
                                table_leg.loc[f, 'Average_distance_to_neighbors'], table_leg.loc[f, 'Water_temperature'] \
                                    = Average_distance_to_neighbors.round(3), Water_temperature.round(3)
                                table_leg.loc[f, 'Average_distance_STD'] = Average_std.round(3)
                                shore_history[table_leg.loc[f,'Cluster']] += shore_bi
                            else : 
                                table_leg.drop(f, inplace=True)
            date_table = pd.concat([date_table, table_leg])
        
    for cluster in centroids.keys():
        
        plt.figure()
        # with open (os.path.join("Agurim", dyr, "shore_history.json"), "w") as sfile:
        #     json.dump(shore_history, sfile)

        if shore_history[cluster] :
            b = np.histogram(shore_history[cluster], np.linspace(np.min(shore_history[cluster]), np.max(shore_history[cluster]), 20))
            y = 100 * b[0] / np.sum(b[0])
            plt.bar(b[1][:-1], y, width=8, color='blue')
            plt.xlabel("Distance to shore (cm)")
            plt.ylabel("Crane precentage")
            plt.title(f"Histogram of shore distance for cluster {cluster} date {dyr}")
            # plt.pause(1)
            plt.savefig(f"Agurim/{dyr}/shore_distance_histogram_date_{dyr}_cluster_{cluster}.png", dpi=300, bbox_inches='tight' )
        
        else :
            plt.plot(0,0)
            plt.xlabel("Distance to shore")
            plt.ylabel("Crane precentage")
            plt.title(f"No photographed cranes on shoreline for cluster {cluster} date {dyr}")
            # plt.pause(1)
            plt.savefig(f"Agurim/{dyr}/shore_distance_histogram_date_{dyr}_cluster_{cluster}.png", dpi=300, bbox_inches='tight')
       
    date_table = date_table.round(3)
    pd.options.display.float_format = "{:,.3f}".format
    print(date_table) 
    date_table.to_excel(f'Agurim/{dyr}/Final_results_{dyr}.xlsx')   
    #print (files)
    print ('--------------------------------')
