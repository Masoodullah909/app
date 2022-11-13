# pehlay to sari important libraries import karlo 
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# app ki heading 
st.write("""
# Explore different ML models and Datasets
Daikhtay Han kon sa best ha in may say?
By Masoodullah Thanks Codanics
""")

# Data set k name ak box may daal ki sidebar pay laga do 
datasets_name = st.sidebar.selectbox(
    "Select Dataset",
    ("Iris", "Brest Cancer","Wine")
)

# Or iskay nichay classifier k name ak dabay may dal do
classifier_name = st.sidebar.selectbox(
    "Select Classifier",
    ("KNN","SVM", "Random Forest")
)

# Ab hum nay ak function define karna hai dataset ko load krna hai dataset ko load krnay k liye
def get_dataset(datasets_name):
    data=None
    if datasets_name == "Iris":
        data = datasets.load_iris()
    elif datasets_name == "Wine":
        data = datasets.load_wine()
    else:
        data = datasets.load_breast_cancer()
    x = data.data
    y = data.target
    return x, y

# ab is function ko bula lay gayn  or X,y variable k equal rakh layn gay
X, y = get_dataset(datasets_name)

# Ab hum apnay data set ki shape ko ap pay print kar dayn gay
st.write('Shape of dataset:', X.shape)
st.write('number of classes:', len(np.unique(y)))

# Next hum different classifier k parameter ko user input may add karayn gay
def add_parameter_ui(classifier_name):
    params = dict()  # Create an empty dictionary
    if classifier_name == 'SVM':
        C = st.sidebar.slider('C', 0.01, 10.0)
        params["C"] = C # Its the degree of correct classification
    elif classifier_name == "KNN":
        K = st.sidebar.slider("K", 1, 15)
        params["K"] = K # Its the number of nearest neighbour
    else:
        max_depth = st.sidebar.slider('max_depth', 2, 15)
        params['max_depth'] = max_depth  # depth of every tree that grow in random forest
        n_estimators = st.sidebar.slider('n_estimators',1,100)
        params['n_estimators'] = n_estimators # number of trees
    return params

# Ab is function ko bula lay gayn or params variable k equal rakh layn gay
params = add_parameter_ui(classifier_name)

# ab hum classifier bnayen gay base on classifier_name and params
def get_classifier(classifier_name, params):
    clf = None
    if classifier_name == 'SVM':
        clf = SVC(C=params['C'])
    elif classifier_name == 'KNN':
        clf = KNeighborsClassifier(n_neighbors=params['K'])
    else:
        clf = clf = RandomForestClassifier(n_estimators=params['n_estimators'],
        max_depth = params['max_depth'], random_state=1234)
    return clf

# Ab is function ko bula lay gayn or clf variable k equal rakh layn gay
clf = get_classifier(classifier_name, params)

# Ab hum apnay dataset ko test and train data may split kar laytay hay by 80/20 ratio
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# Ab hum apnay classifier ki training karni hai
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Model ka accuracy score check kr layna ha or isay app pay print kar dayna ha
acc = accuracy_score(y_test, y_pred)
st.write(f'Classifier={classifier_name}')
st.write(f'Accuracy=', acc)

###### PLOT DATASET ######
# ab hum apnay saray saray features ko 2 dimensional plot pay draw kar dayn gay using pca
pca = PCA(2)
X_projected = pca.fit_transform(X)

# ab hum apna data 0 or 1 dimenssion may slice kar kar dayn gay
x1 = X_projected[:,0]
x2 = X_projected[:,1]

fig = plt.figure()
plt.scatter(x1, x2, c=y, alpha=0.8, cmap='viridis')

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar()

# plt.show()
st.pyplot(fig)


















