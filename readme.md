EX -1	Impute missing values in data inputs

Aim:
	To Implement missing values in data input 
Program:
import pandas as pd
import numpy as np
null_values=pd.Series([1,np.NaN,2,np.NaN,4,5])
null_values
 
null_values.isna()

 
null_values.notna()
 

null_values.fillna(0)
 

null_values.fillna(np.mean(null_values))
 

null_values.interpolate()
 
Result:
 
EX -2 	Use feature selection/extraction method to perform dimensionality reduction

Aim:
	To use feature selection/extraction method to perform dimensionality reduction
Program:
	from google.colab import files

uploaded =files.upload()

 
from pandas import read_csv
from numpy import set_printoptions
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
filename = '/content/pima-indians-diabetes.csv'
# names =['preg','plas','pres','skin','test','mass','pedi','age','class']
dataframe = read_csv(filename,names=names)
dataframe

 


 
array = dataframe.values
X=array[:,0:8]
Y=array[:,8]
test = SelectKBest(score_func=f_classif,k=4)
fit = test.fit(X,Y)

set_printoptions(precision=3)
print(fit.scores_)
features = fit.transfrm(X)
# summarize selected features
print(features[0:5,:])
 

 Result: 
EX -3	Demonstrate Naïve Bayes Classification
Aim:
	To demonstrate Naïve Bayes Classification.
Program:
#load the iris ataset
from sklearn.datasets import load_iris
iris = load_iris()

#store the feature matrix (x) and response vector (y)
X = iris.data
Y = iris.target 

#splitting x and y into training and testing sets
from sklearn.model_selection import train_test_split
X_train,X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, random_state=1)

#training the model on training set
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, Y_train)

#making predictions on the testing set
Y_pred = gnb.predict(X_test)

#comparing actual response values (Y_test) with predicted response values (Y_pred) 
from sklearn import metrics
print("Gaussian Naive Bayes model accuracy (in %)",metrics.accuracy_score(Y_test, Y_pred)*100)

Output:
	 


Result:
 
EX-4 Classify the input dataset using decision tree

Aim:
	To classify the input dataset using decision tree
Program:
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from google.colab import files
uploaded=files.upload()
 
col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']
pima = pd.read_csv("diabetes.csv", header=0, names=col_names)
pima.head()
 
feature_cols = ['pregnant', 'insulin', 'bmi', 'age','glucose','bp','pedigree']
X = pima[feature_cols] 
y = pima.label 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

clf = DecisionTreeClassifier()
clf = clf.fit(X_train,y_train)

y_pred = clf.predict(X_test)

print("Accuracy in testing set:",metrics.accuracy_score(y_test, y_pred))
 
from sklearn.tree import export_graphviz
from six import StringIO
from IPython.display import Image
import pydotplus
dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,  
                    filled=True, rounded=True,
                    special_characters=True, feature_names = feature_cols,class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('diabetes.png')
Image(graph.create_png())


 
clf = DecisionTreeClassifier(criterion="entropy", max_depth=3)
clf = clf.fit(X_train,y_train)
y_pred_test = clf.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred_test))
	 
 
from six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus
dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,  
                    filled=True, rounded=True,
                    special_characters=True, feature_names = feature_cols,class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('diabetes.png')
Image(graph.create_png())

 
 
EX-5 Perform classification using Support Vector Machines
Aim:
	To Perform classification using Support Vector Machines
Program:
import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
from google.colab import files

uploaded=files.upload()
 
data=pd.read_csv("diabetes.csv")
data.head()
 
non_zero = ['Glucose','BloodPressure','SkinThickness','Insulin','BMI']
for coloumn in non_zero:
    data[coloumn] = data[coloumn].replace(0,np.NaN)
    mean = int(data[coloumn].mean(skipna = True))
    data[coloumn] = data[coloumn].replace(np.NaN,mean)
    print(data[coloumn])
from sklearn.model_selection import train_test_split
X =data.iloc[:,0:8]
y =data.iloc[:,8]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0, stratify=y)
X.head()

0      148.0
1       85.0
2      183.0
3       89.0
4      137.0
       ...  
763    101.0
764    122.0
765    121.0
766    126.0
767     93.0
Name: Glucose, Length: 768, dtype: float64
0      72.0
1      66.0
2      64.0
3      66.0
4      40.0
       ... 
763    76.0
764    70.0
765    72.0
766    60.0
767    70.0
Name: BloodPressure, Length: 768, dtype: float64
0      35.0
1      29.0
2      29.0
3      23.0
4      35.0
       ... 
763    48.0
764    27.0
765    23.0
766    29.0
767    31.0
Name: SkinThickness, Length: 768, dtype: float64
0      155.0
1      155.0
2      155.0
3       94.0
4      168.0
       ...  
763    180.0
764    155.0
765    112.0
766    155.0
767    155.0
Name: Insulin, Length: 768, dtype: float64
0      33.6
1      26.6
2      23.3
3      28.1
4      43.1
       ... 
763    32.9
764    36.8
765    26.2
766    30.1
767    30.4
Name: BMI, Length: 768, dtype: float64
	
 

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
from sklearn import svm
svm1 = svm.SVC(kernel='linear', C = 0.01)
svm1.fit(X_test,y_test)
svm.SVC(C=0.01, kernel='linear')
y_train_pred = svm1.predict(X_train)
y_test_pred = svm1.predict(X_test)
y_test_pred
  from sklearn.metrics import accuracy_score,confusion_matrix
confusion_matrix(y_test,y_test_pred)
  accuracy_score(y_test,y_test_pred)
 
Result:
 
EX-6  Perform multivariate classification and regression
Aim:
	To Perform multivariate classification and regression
Program:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from google.colab import files
upladed=files.upload()
  test=pd.read_csv("california_housing_test.csv")
test.head()
  upladed=files.upload()
  train=pd.read_csv("california_housing_train.csv")
plt.figure()
sns.heatmap(train.corr(),cmap='coolwarm')
plt.show()
sns.lmplot(x='median_income',y='median_house_value',data=train)
sns.lmplot(x='housing_median_age',y='median_house_value',data=train)
      data=train
data=data[['total_rooms','total_bedrooms','housing_median_age','median_income','population','households']]
data.info()
data['total_rooms']=data['total_rooms'].fillna(data['total_rooms'].mean())
data['total_bedrooms']=data['total_bedrooms'].fillna(data['total_bedrooms'].mean())
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 17000 entries, 0 to 16999
Data columns (total 6 columns):
 #   Column              Non-Null Count  Dtype  
---  ------              --------------  -----  
 0   total_rooms         17000 non-null  float64
 1   total_bedrooms      17000 non-null  float64
 2   housing_median_age  17000 non-null  float64
 3   median_income       17000 non-null  float64
 4   population          17000 non-null  float64
 5   households          17000 non-null  float64
dtypes: float64(6)
memory usage: 797.0 KB
<ipython-input-8-2fa8111f12dc>:4: SettingWithCopyWarning: 
A value is trying to be set on a copy of a slice from a DataFrame.
Try using .loc[row_indexer,col_indexer] = value instead

See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
  data['total_rooms']=data['total_rooms'].fillna(data['total_rooms'].mean())
<ipython-input-8-2fa8111f12dc>:5: SettingWithCopyWarning: 
A value is trying to be set on a copy of a slice from a DataFrame.
Try using .loc[row_indexer,col_indexer] = value instead

See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
  data['total_bedrooms']=data['total_bedrooms'].fillna(data['total_bedrooms'].mean())
train.head()
  from sklearn.model_selection import train_test_split
y=train.iloc[:,8]
x_train, x_test, y_train, y_test = train_test_split(train, y,test_size=0.2,random_state=0) 
print(y.name)
    print(regressor.intercept_)
print(regressor.coef_)
  predictions=regressor.predict(x_test)
predictions=predictions.reshape(-1,1)
print(predictions)
  from sklearn.metrics import mean_squared_error
print('MSE:',mean_squared_error(y_test,predictions))
print('RMSE:',np.sqrt(mean_squared_error(y_test,predictions)))
  import numpy as np
arr=np.array([1,2,3,4,5,6,7,8,9,10,11,12])
newarr=arr.reshape(-1,1)
print(newarr)
 

Result:
 
EX-7 Develop a program to implement feed-forward neural networks
Aim:
	To develop a program to implement feed-forward neural networks
Program:
import math
import pandas as pd
from keras import models, layers, optimizers, regularizers
import numpy as np
import random
from sklearn import model_selection, preprocessing
import tensorflow as tf
from tqdm import tqdm
import matplotlib.pyplot as plt
file_name = '/content/SAheart.data'
data = pd.read_csv(file_name, sep=',', index_col=0)
data['famhist'] = data['famhist'] == 'Present'
data.head()
  n_test = int(math.ceil(len(data) * 0.3))
random.seed(42)
test_ixs = random.sample(list(range(len(data))), n_test)
train_ixs = [ix for ix in range(len(data)) if ix not in test_ixs]
train = data.iloc[train_ixs, :]
test = data.iloc[test_ixs, :]
print(len(train))
print(len(test))
  #features = ['sbp', 'tobacco', 'ldl', 'adiposity', 'famhist', 'typea', 'obesity', 'alcohol', 'age']
features = ['adiposity', 'age']
response = 'chd'
x_train = train[features]
y_train = train[response]
x_test = test[features]
y_test = test[response]
x_train = preprocessing.normalize(x_train)
x_test = preprocessing.normalize(x_test)
hidden_units = 10     # how many neurons in the hidden layer
activation = 'relu'   # activation function for hidden layer
l2 = 0.01             # regularization - how much we penalize large parameter values
learning_rate = 0.01  # how big our steps are in gradient descent
epochs = 5            # how many epochs to train for
batch_size = 16       # how many samples to use for each gradient descent update
# create a sequential model
model = models.Sequential()

# add the hidden layer
model.add(layers.Dense(input_dim=len(features),
                       units=hidden_units, 
                       activation=activation))

# add the output layer
model.add(layers.Dense(input_dim=hidden_units,
                       units=1,
                       activation='sigmoid'))

# define our loss function and optimizer
model.compile(loss='binary_crossentropy',
              # Adam is a kind of gradient descent
              optimizer=optimizers.Adam(lr=learning_rate),
              metrics=['accuracy'])
/usr/local/lib/python3.10/dist-packages/keras/optimizers/legacy/adam.py:117: UserWarning: The `lr` argument is deprecated, use `learning_rate` instead.
  super().__init__(name, **kwargs)
# train the parameters
history = model.fit(x_train, y_train, epochs=10, batch_size=batch_size)

# evaluate accuracy
train_acc = model.evaluate(x_train, y_train, batch_size=32)[1]
test_acc = model.evaluate(x_test, y_test, batch_size=32)[1]
print('Training accuracy: %s' % train_acc)
print('Testing accuracy: %s' % test_acc)

losses = history.history['loss']
plt.plot(range(len(losses)), losses, 'r')
plt.show()

### RUN IT AGAIN! ###
   

Result:
 
EX-8  Implement K-means clustering
Aim:
	To Implement K-means clustering
Program:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import  KMeans
import sklearn.metrics as metrics
from google.colab import files
uploaded = files.upload()
  df = pd.read_csv("/content/user1.csv")
df
  x=df.iloc[:,[0,1]].values
print(x)
   
[1 0 0 0 1 1 0 1 0 1]
Cluster Center are:
[[0.488 0.29 ]
 [0.248 0.862]]
/usr/local/lib/python3.10/dist-packages/sklearn/cluster/_kmeans.py:870: FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning
  warnings.warn(
plt.scatter(x[:,0],x[:,1],c=y_kmeans2,cmap='viridis')
plt.show()
 


Result:
 
EX-9 Develop a simple application to demonstrate reinforcement learning
Aim:
	To develop a simple application to demonstrate reinforcement learning
Program:
import gym
env = gym.make("Taxi-v3").env
env.render()
 
env.reset() # reset environment to a new, random state
env.render()
 
print("Action Space {}".format(env.action_space))
print("State Space {}".format(env.observation_space))
 
state = env.encode(3, 1, 2, 0) # (taxi row, taxi column, passenger index, destination index)
print("State:", state)
 
env.s = state
env.render()
 
 
env.P[328]
 
env.s = 328  # set environment to illustration's state

epochs = 0
penalties, reward = 0, 0

frames = [] # for animation

done = False

while not done:
    action = env.action_space.sample()
    state, reward, done, info = env.step(action)

    if reward == -10:
        penalties += 1
    
    # Put each rendered frame into dict for animation
    frames.append({
        'frame': env.render(mode='ansi'),
        'state': state,
        'action': action,
        'reward': reward
        }
    )

    epochs += 1
    
    
print("Timesteps taken: {}".format(epochs))
print("Penalties incurred: {}".format(penalties))

 
 
def print_frames(frames):
    for i, frame in enumerate(frames):
        clear_output(wait=True)
        print(frame['frame'])
        print(f"Timestep: {i + 1}")
        print(f"State: {frame['state']}")
        print(f"Action: {frame['action']}")
        print(f"Reward: {frame['reward']}")
        sleep(.1)
        
print_frames(frames)

 


Result:
 
EX-10 Assess machine learning algorithms using cross validation methods
Aim:
	To assess machine learning algorithms using cross validation methods
Program:
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import datasets
from sklearn import svm
X, y = datasets.load_iris(return_X_y=True)
X.shape, y.shape
 
X_train, X_test, y_train, y_test = train_test_split(
  X, y, test_size=0.3, random_state=0)
X_train.shape, y_train.shape
  X_test.shape, y_test.shape
  clf = svm.SVC(kernel='linear', C=1)
clf.fit(X_train, y_train)
  Y_predict = clf.predict(X_test)
print(classification_report(Y_predict,y_test))
  from sklearn.model_selection import cross_val_score
clf = svm.SVC(kernel='linear', C=1, random_state=42)
scores = cross_val_score(clf, X, y, cv=10)
scores
  print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))
 


Result:




