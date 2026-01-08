import pandas as pd

#loading in the data

file_path = "assignments/hw01/data/gisette_train.data"

#file was seperated via spaces.  We know that its dimensions are 6000 rows and 5000 columns.  
#needed to use delim_whitespace instead of sep, and specify no header
data = pd.read_csv(file_path, delim_whitespace=True, header=None)

print(data.shape)