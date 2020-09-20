import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.preprocessing import MinMaxScaler
from pandas.errors import ParserError
import plotly.express as px
import plotly.graph_objects as go

from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt 
import matplotlib
matplotlib.use("Agg")
import seaborn as sns
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC



st.title('Insurance Product Recommendation')

df = pd.read_csv("data4.csv")
 

df2 = pd.read_csv("data3.csv")



st.header("Exploratory Data Analysis")

data = "data3.csv"
if data is not None:
	df = pd.read_csv(data)
	#st.dataframe(df.head())

	if st.checkbox('Show dataframe'):
   		st.write(df)





chosen_classifier = st.selectbox("Choose to view", ('Show Shape','Show Columns', 'Summary', 'Show Selected Columns')) 
if chosen_classifier == 'Show Shape': 
		
	st.write(df.shape)

		
elif chosen_classifier == 'Show Columns': 
		
	all_columns = df.columns.to_list()
	st.write(all_columns)

elif chosen_classifier == 'Summary': 
	st.write(df.describe())

elif chosen_classifier == 'Show Selected Columns': 
	all_columns = df.columns.to_list()
	selected_columns = st.multiselect("Select Columns",all_columns)
	new_df = df[selected_columns]
	st.dataframe(new_df)






st.header("Data Visualization")
data = "data4.csv"
if data is not None:
	df = pd.read_csv(data)

st.subheader('Pie plot')
all_columns = df.columns.to_list()
column_to_plot = st.selectbox("Select 1 Column",all_columns)
pie_plot = df[column_to_plot].value_counts().plot.pie(autopct="%1.1f%%")
st.write(pie_plot)
st.pyplot()


st.subheader('Bar/Line plot')		
# Customizable Plot

all_columns_names = df.columns.tolist()
type_of_plot = st.selectbox("Select Type of Plot",["bar","line"])
selected_columns_names = st.multiselect("Select Columns To Plot",all_columns_names)

if st.button("Generate Plot"):
	st.success("Generating Customizable Plot of {} for {}".format(type_of_plot,selected_columns_names))

	#Plot By Streamlit
	

	if type_of_plot == 'bar':
		cust_data = df[selected_columns_names]
		st.bar_chart(cust_data)

	elif type_of_plot == 'line':
		cust_data = df[selected_columns_names]
		st.line_chart(cust_data)

	# Custom Plot 
	elif type_of_plot:
		cust_plot= df[selected_columns_names].plot(kind=type_of_plot)
		st.write(cust_plot)
		st.pyplot()




st.subheader('Histogram')
col0 = st.selectbox('Select a factor?', df.columns[0:51]) 
species = st.multiselect('Select unique value needed from the factor?', df[col0].unique())
#col1 = st.selectbox('Which feature on x?', df.columns[0:51])
#col2 = st.selectbox('Which feature on y?', df.columns[0:51])

new_df = df[(df[col0].isin(species))]
#st.write(new_df)
# create figure using plotly express
#fig = px.scatter(new_df, x =col1,y=col2, color=col0)
# Plot!

#st.plotly_chart(fig)

#st.subheader('Histogram')
 
feature = st.selectbox('Select the second factor?', df.columns[0:51])
# Filter dataframe
new_df2 = df[(df[col0].isin(species))][feature]
fig2 = px.histogram(new_df, x=feature, color=col0, marginal="rug")
st.plotly_chart(fig2)

st.subheader('Scatter plot')
fig2 = px.scatter(new_df, x=feature, color=col0)
st.plotly_chart(fig2)

st.subheader('Heatmap')
fig = px.density_heatmap(new_df, x=feature, y=col0, marginal_x="box", marginal_y="violin")
st.plotly_chart(fig)

st.header('Building ML Models')
y = df2.Plan1
y1 = df2.Plan2
X=df2.drop(['Plan1'],1)
colnames = X.columns


from imblearn.over_sampling import SMOTE
from sklearn.naive_bayes import GaussianNB
os=SMOTE(random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y.values, test_size=0.3)
columns=X_train.columns
os_data_X,os_data_y=os.fit_sample(X_train,y_train)

os_data_X =pd.DataFrame(data=os_data_X,columns=columns)
os_data_y=pd.DataFrame(data=os_data_y,columns=['y'])

os_data_X,os_data_y=os.fit_sample(X_train,y_train)


X_train,X_test,y_train,y_test=train_test_split(os_data_X,os_data_y,test_size=0.3,random_state=0)


type = st.selectbox("Algorithm type", ("Classification", "None"))
if type == "Classification":
	chosen_classifier = st.selectbox("Please choose a classifier", ('None','Naive Bayes', 'Random Forest', 'K Neighbors')) 

	if chosen_classifier == 'None': 
		
		pass

	elif chosen_classifier == 'Naive Bayes': 
		
		nb=GaussianNB()
		nb.fit(X_train,y_train)
		nb_ored=nb.predict(X_test)
		print1 = nb.score(X_train, y_train)
		print2 = nb.score(X_test, y_test)
		st.write('Accuracy on training set: ', print1)
		st.write('Accuracy on test set: ', print2)

		
	elif chosen_classifier == 'Random Forest': 
		
		classifier = RandomForestClassifier(n_estimators = 50,class_weight="balanced")
		classifier.fit(X_train, y_train)
		y_pred = classifier.predict(X_test)
		arr = classifier.score(X_train,y_train)
		arr2 = classifier.score(X_test, y_test)
		st.write('Accuracy on training set: ', arr)
		st.write('Accuracy on test set: ', arr2)
		confusion_majority=confusion_matrix(y_test,y_pred)
		st.write('Confusion Matrix')
		st.write(confusion_majority)

	elif chosen_classifier == 'K Neighbors': 
		pass
		knn=KNeighborsClassifier(n_neighbors=5)
		knn.fit(X_train, y_train)
		kb_ored=knn.predict(X_test)
		arr = knn.score(X_train,y_train)
		arr2 = knn.score(X_test, y_test)
		st.write('Accuracy on training set: ', arr)
		st.write('Accuracy on test set: ', arr2)
		confusion_majority=confusion_matrix(y_test,kb_ored)
		st.write('Confusion Matrix')
		st.write(confusion_majority)



elif type == "None":
	pass
