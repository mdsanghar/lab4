import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
# read in the data and check the first 5 
file_path='C:/Users/Shahid Sanghar/Desktop/mohammed/ASSIGNMENTS/sem2/AI in Enterprise system/MDLab4'
df=pd.read_csv(file_path+'/Fish.csv')
df.head()
# Split the data into training and testing sets
x = df.drop(['Weight','Species'],axis=1)
y = df['Weight']
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.33, random_state=100)
#calling Linear Regression algorithm from sklearn
classifier = LinearRegression()
classifier.fit(x_train, y_train)

#making predictions
y_predictions = classifier.predict(x_test)
print(y_predictions)

print("Accuarcy ", classifier.score(x_test, y_test))

#create a pickle file
import pickle
pickle_out = open(file_path+"/classifier.pkl","wb")
pickle.dump(classifier,pickle_out)
pickle_out.close()
classifier.predict([[242,23,25,30,4.123]])