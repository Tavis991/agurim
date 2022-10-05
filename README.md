# agurim
This is a final project. 

The project is made of two files -
1) main_analysis which run on all directories, parses input and builds a table, where each entry is photo id (name)
to hold all information (a pandas dataframe) per photo.
main_analysis also saves information per cluster - histogram of nearest to shore cranes, off the table, in a dictionary. 

2) project_h, which analyses each photo, returning several parameters back to main_analysis, 
which updates the information in it's table, and saving some information per cluster. 

comments are written where neccessary.
commented are also debug prints (photos actually), which i have left to be activated to produce any step of the algorithm along the way.

in project_h there are several hard-coded global params (FACTORS), feel free to play them as you like and observe results. 

instruction: 
place files in directory containing an 'Agurim' (name does not matter) folder,
with its subfolders - legs, pix4d ('pix' does matter) and flight_data.csv (.csv is a must) 
press 'play' in main analysis. 
