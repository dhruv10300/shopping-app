import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import warnings
import os
import importlib
warnings.filterwarnings('ignore')
sns.set_theme(color_codes=True)

# user-defined function to check library is installed or not, if not installed then it will install automatically at runtime.
def check_and_install_library(library_name):
    try:
        importlib.import_module(library_name)
        print(f"{library_name} is already installed.")
    except ImportError:
        print(f"{library_name} is not installed. Installing...")
        try:
            import pip
            pip.main(['install', library_name])
        except:
            print("Error: Failed to install the library. Please install it manually.")
            
#dataset amazon 
file_path = r"C:\Users\Dhruv\Downloads\dataset\archive (3)\ratings_Electronics.csv"
df= pd.read_csv(file_path, names=['user_id', 'item_id', 'rating', 'timestamp'])

print(df.head())
print("Shape of the loaded data:", df.shape)

df.columns
#Since our dataset is too big and it will be difficult to analyze the entire dataset due to limited resources,thats'why I am randomly taking 20% of the data as sample out of the whole dataset which is 2000000
electronics_data=df.sample(n=200000,ignore_index=True)

#after taking samples drop df to release the memory occupied by entire dataframe
del df

#print top 5 records of the dataset
electronics_data.head()

#print the concise information of the dataset
electronics_data.info()

#drop timestamp column
electronics_data.drop('timestamp',axis=1,inplace=True)

electronics_data.describe()


#handle missing values
electronics_data.isnull().sum()

#handling duplicate records
electronics_data[electronics_data.duplicated()].shape[0]

electronics_data.head()

plt.figure(figsize=(8,4))
sns.countplot(x='rating',data=electronics_data)
plt.title('Rating Distribution')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.grid()
plt.show()

print('Total rating : ',electronics_data.shape[0])
print('Total unique users : ',electronics_data['user_id'].unique().shape[0])
print('Total unique products : ',electronics_data['item_id'].unique().shape[0])

no_of_rated_items_per_user = electronics_data.groupby(by='user_id')['rating'].count().sort_values(ascending=False)
no_of_rated_items_per_user.head()

print('No of rated item more than 50 per user : {} '.format(sum(no_of_rated_items_per_user >= 50)))

#Popularity Based Recommendation
data=electronics_data.groupby('item_id').filter(lambda x:x['rating'].count()>=50)
data.head()

no_of_rating_per_item=data.groupby('item_id')['rating'].count().sort_values(ascending=False)
no_of_rating_per_item.head()

#top 20 item
no_of_rating_per_item.head(40).plot(kind='bar')
plt.xlabel('item_id')
plt.ylabel('num of rating')
plt.title('top 40 item')
plt.show()

#average rating item
mean_rating_item_count=pd.DataFrame(data.groupby('item_id')['rating'].mean())

mean_rating_item_count.head()

#plot the rating distribution of average rating item
plt.hist(mean_rating_item_count['rating'],bins=100)
plt.title('Mean Rating distribution')
plt.show()

#check the skewness of the mean rating data
mean_rating_item_count['rating'].skew()
#it is highly negative skewed
mean_rating_item_count['rating_counts'] = pd.DataFrame(data.groupby('item_id')['rating'].count())

mean_rating_item_count.head()

#highest mean rating product
mean_rating_item_count[mean_rating_item_count['rating_counts']==mean_rating_item_count['rating_counts'].max()]

#min mean rating item
print('min average rating item : ',mean_rating_item_count['rating_counts'].min())
print('total min average rating items : ',mean_rating_item_count[mean_rating_item_count['rating_counts']==mean_rating_item_count['rating_counts'].min()].shape[0])

#plot the rating count of mean_rating_item_count
plt.hist(mean_rating_item_count['rating_counts'],bins=100)
plt.title('rating count distribution')
plt.show()

#joint plot of rating and rating counts
sns.jointplot(x='rating',y='rating_counts',data=mean_rating_item_count)
plt.title('Joint Plot of rati`ng and rating counts')
plt.tight_layout()
plt.show()

plt.scatter(x=mean_rating_item_count['rating'],y=mean_rating_item_count['rating_counts'])
plt.show()

print('Correlation between Rating and Rating Counts is : {} '.format(mean_rating_item_count['rating'].corr(mean_rating_item_count['rating_counts'])))

#Collaberative filtering (Item-Item recommedation)
#import surprise library for collebrative filtering
check_and_install_library('surprise')
from surprise import KNNWithMeans
from surprise import Dataset
from surprise import accuracy
from surprise import Reader
from surprise.model_selection import train_test_split

#Reading the dataset
reader = Reader(rating_scale=(1, 5))
surprise_data = Dataset.load_from_df(data,reader)

#Splitting surprise the dataset into 80,20 ratio using train_test_split
trainset, testset = train_test_split(surprise_data, test_size=0.3,random_state=42)

# Use user_based true/false to switch between user-based or item-based collaborative filtering
algo = KNNWithMeans(k=5, sim_options={'name': 'pearson_baseline', 'user_based': False})
algo.fit(trainset)


#make prediction using testset
test_pred=algo.test(testset)

#print RMSE
print("Item-based Model : Test Set")
accuracy.rmse(test_pred ,verbose=True)

data2=data.sample(20000)
ratings_matrix = data2.pivot_table(values='rating', index='user_id', columns='item_id', fill_value=0)
ratings_matrix.head()

#check the shape of the rating_matrix
ratings_matrix.shape

#transpose the metrix to make column (productId) as index and index as column (user_id)
x_ratings_matrix=ratings_matrix.T
x_ratings_matrix.head()


x_ratings_matrix.shape

#Decomposition of the matrix using Singular Value Decomposition technique
from sklearn.decomposition import TruncatedSVD
SVD = TruncatedSVD(n_components=10)
decomposed_matrix = SVD.fit_transform(x_ratings_matrix)
decomposed_matrix.shape


#Correlation Matrix
correlation_matrix = np.corrcoef(decomposed_matrix)
correlation_matrix.shape

x_ratings_matrix.index[10]

i="B00001P4ZH"
item_names=list(x_ratings_matrix.index)
item_id=item_names.index(i)
print(item_id)

#Correlation for all items with the item purchased by this customer based on items rated by other customers people who bought the same product

correlation_item_id = correlation_matrix[item_id]
correlation_item_id.shape

correlation_matrix[correlation_item_id > 0.75].shape

#Recommending top 20 highly correlated products in sequence
recommend = list(x_ratings_matrix.index[correlation_item_id > 0.75])
recommend[:10]











