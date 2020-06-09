#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 16:03:04 2020

@author: damian
"""

from pyspark.sql import SparkSession

spark = SparkSession.builder.appName('lin').getOrCreate()

from pyspark.ml.regression import LinearRegression

df = spark.read.format('libsvm').load('sample_LR_data.txt')


train_data,test_data = df.randomSplit([0.7, 0.3])

lr = LinearRegression(featuresCol='features',labelCol='label',predictionCol='prediction')
lr_Model = lr.fit(train_data)
lr_Model.coefficients
lr_Model.intercept
training_summary = lr_Model.summary
# How much variance your model can explain? r^2 will tell you that
training_summary.r2
training_summary.rootMeanSquaredError

test_result = lr_Model.evaluate(test_data)
test_result.meanSquaredError

# A more realistic scenario (model deployment case in which no labels are provided)
unlabel_data = test_result.select('features')
predictions = lr_Model.transform(unlabel_data)

print('Second exercise'.upper())
data = spark.read.csv('Ecommerce_Customers.csv', inferSchema=True, header=True)
data.printSchema()
print('The idea is to predict the yearly amoun Spent'.upper())

from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import format_number

assemb = VectorAssembler(inputCols=['Avg Session Length', 'Time on App', 'Time on Website', 'Length of Membership'],outputCol='features')
output = assemb.transform(data)
output.printSchema()
output.head(1)
print("Note that the features are represented as dense vectors".upper())
final_data = output.select(output['features'], output['Yearly Amount Spent'].alias('Output_data'))
final_data.show()
train_data, testdata = final_data.randomSplit([0.7, 0.3])

lr = LinearRegression(labelCol='Output_data')
lr_model = lr.fit(train_data)
test_results = lr_model.evaluate(test_data)
test_results.residuals.show()

print('Third exersize'.upper())
df_2 = spark.read.csv('cruise_ship_info.csv',inferSchema=True,header=True)
df_2.columns

for ship in df_2.head(5):
    print(ship)
    print('\n')

print("information regarding 'Cruise_line' feature should be transformed into numerical variables".upper())
df_2.groupBy('Cruise_line').count().show()
print('Each row of the data frame above represent a different index')
from pyspark.ml.feature import StringIndexer
indexer = StringIndexer(inputCol='Cruise_line',outputCol='cruise_num')
df_2_new = indexer.fit(df_2).transform(df_2)

# Include 'cruise_num' if you want to work with d2_2new 
assemb = VectorAssembler(inputCols=['Age', 'Tonnage', 'passengers', 'cabins', 'passenger_density'],outputCol='features')
output = assemb.transform(df_2) # Use d2_2new with 'cruise_num' 

df_final = output.select('features', 'crew')
train_data, testdata = df_final.randomSplit([0.7, 0.3])

lr = LinearRegression(labelCol='crew')
lr_model = lr.fit(train_data)
test_results = lr_model.evaluate(test_data)
test_results.rootMeanSquaredError
test_results.r2
test_results.residuals.show()

from pyspark.sql.functions import corr 
df_2.select(corr('crew','passengers')).show()
df_2.select(corr('crew','cabins')).show()
print('The crew has a high correlation with the number of passengers and the number of cabins'.upper())
