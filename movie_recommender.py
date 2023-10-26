import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import re

#--------------Functions--------------
#head over to jupyter-notebook implementation for the explanation
#of this function
def extractProduction(row_value):
    #row_value is a string
    prod_house=''
    #using regular expressions to extract the name 
    rexp = "name\":.*?,"
    preprocessList = re.findall(rexp,row_value)
    rexp1 = '"(.*)"'
    for temp in preprocessList:
        temp = temp.split(":")[1]
        #using exception because some values in temp
        #is an empty list.
        try:
            house = re.findall(rexp1,temp)[0]
            prod_house+=house+" "
        except:
            pass
    return prod_house.rstrip()

#function to combine row values
def combineRows(value):
    v = ' '.join(map(str,list(value.values)))
    
    return v

#function to change every string value to lowercase
def toLowerDf(df):
    columns = list(df.columns)
    for col in columns:
        if df[col].dtype=='O':
            df[col]=df[col].str.lower()

#getting the index of the input movie given by the user
def get_index_movie_title(movie):
    movie = movie.lower()
    indexL = df.index[df['title']==movie].tolist()
    if len(indexL)==0:
        return -1
    else:
        return indexL[0]
    
def get_movie_title_from_index(index):
    return df.iloc[index]['title']

#getting the similar movies based on similarity scores
def get_similar_movies(movie_name,no_of_results=5):
    
    movie_index = get_index_movie_title(movie_name)
    if movie_index==-1:
        return []
    similar_movies = list(enumerate(similarity_scores[movie_index]))
    similar_movies = sorted(similar_movies,key=lambda x:x[1],reverse=True)
    similar_movies = similar_movies[1:no_of_results+1]
    results=[]
    
    for res in similar_movies:
        #here res is a tuple -> (index,similarity_score)
        value = get_movie_title_from_index(res[0])
        results.append(value)
        
    return results

#read the dataset
df = pd.read_csv('movie_dataset.csv')

#feature extraction
#these are the features I found to be useful
features = ['keywords','cast','genres','director']

#copy the data to another dataframe with those features to preprocess
df1 = df[features].copy()

#making the values of production_companies to a normal string
df['production_companies']=df.production_companies.apply(extractProduction)

df1['text']=df1.apply(combineRows,axis=1)

#combining original dataset with the "text" column of the new dataset
df['combined_features']=df1['text']

#getting rid of null values
df.keywords.fillna(' ',inplace=True)

#convert the dataframe string values to lowercase
toLowerDf(df)

#this sklearn method will count the frequency of the words occuring in 
#the "combined_features" field 
cv = CountVectorizer() 

count_matrix = cv.fit_transform(df['combined_features'])

#finding the similarity scores using cosine distance betweeb two vectors formula
similarity_scores = cosine_similarity(count_matrix)

#getting the input of the user movie name and no. of results
print()
movie_name = input("Enter the movie name: ")
try:
    no_of_results = int(input("Enter the no. of results to display(leave blank to set default value 5): "))
except:
    no_of_results=5



results = get_similar_movies(movie_name,no_of_results)
print()
if len(results)!=0:
    print("These are the recommended movies : ")
    print()
    for i,movie in enumerate(results):
        print(f"{i+1}. {movie}")
else:
    print("Sorry no movie named \""+movie_name+"\" present in the database. Or please check the spelling!")
print()