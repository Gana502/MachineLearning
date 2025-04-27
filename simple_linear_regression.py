import Pandas as pd # To import Pandas pacakage to use dataframes
from sklearn.model_selection import train_test_split # Import function used to create train and test datasets
from sklearn.linear_model import LinearRegression # Import class to create LinearRegression model
from sklearn.metrics import r2_score #Import function to evaluate the model
import pickle # package to save a model

dataset=pd.read_csv("salary_data.csv") # Import csv file into pandas dataframe
independent=dataset[["YearExperience"]] # Split input from the dataframe and assign to a variable
dependent=dataset[["salary"]]# Split output from the dataframe and assign to a variable

X_train, X_test, Y_train, Y_test = train_test_split(independent, dependent, test_size=0.30, random_state=0) #Call to function train_test_split to create train and test tests for input and output datasets
regressor=LinearRegression() # object regressor created for class LinearRegression
regressor.fit(X_train,Y_train) # Model created using training dataset
weight=regressor.coef_ # to find slope
bias=regressor.intercept_ # to find bias

Y_pred=regressor.predict(X_test) # To predict the output using the model created
r_score=r2_score(Y_test,Y_pred) # Evaluate the model and store the output to a variable
print("Accuracy of the model is: ",r_score) # Print the model accuracy

filename = "finalized_linear_model.sav" # Specifies filename to save the model
pickle.dump(regressor,open(filename,'wb') # Step to save the model to the filename

loaded_model=pickle.load(open("finalized_linear_model.sav",'rb')) # Load the saved model
result=loaded_model.predict([[15]]) # Predict output using loaded model
print("Predicted Salary by the Linear model is: ",result) # print the output to console
