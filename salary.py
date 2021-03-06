from os import system
import pandas as pa
import sklearn
import numpy as num
import joblib as jb
from sklearn.linear_model import LinearRegression

db = pa.read_csv("SalaryData.csv")

Y = db["Salary"]

X = db["YearsExperience"].values.reshape(-1,1)

model=LinearRegression()
model.fit(X , Y)

while True:
	print("------------------ Welcome To Salary Predictor Program ----------------")
	ex = float(input("\nEnter the experience what you have: "))
	print("predicted salary:", model.predict([[ex]]))
	print("-----------------------------------------------------------------------")
	
	exit = input("\n press E to exit and for continue press Enter: ")
	print()
	if exit == "E":
		break

jb.dump(model, 'Salary.pk1')




