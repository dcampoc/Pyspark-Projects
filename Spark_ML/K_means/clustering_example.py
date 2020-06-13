#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Kmeans clustering algorithm on pyspark

@author: damian
"""

from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('cluster').getOrCreate()
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler
data = spark.read.csv('seeds_dataset.csv', header=True, inferSchema=True)
data.printSchema()

assembler = VectorAssembler(inputCols=data.columns, outputCol='features')   
final_data = assembler.transform(data)
final_data = final_data.select('features')
from pyspark.ml.feature import StandardScaler

# Scaling dimensions that have different orders of magnitude
scaler = StandardScaler(inputCol='features', outputCol='scaledFeatures')
# Extract std and mean information from the data and then transform it
final_data = scaler.fit(final_data).transform(final_data)

# Assuming knowledge of 3 groups (it could be tested with 1, 2 and 4; see the differences)
num_clusters = 3 
kmeans = KMeans(featuresCol='scaledFeatures').setK(num_clusters).setSeed(1)
model = kmeans.fit(final_data)
# Calculate the witin cluster sum of square errors
wssse = model.computeCost()
print('WSSSE: {}'.format(wssse))
print('\n')
print('Cluster centers:')
centers = model.clusterCenters()
print(centers)
print('\n')
print('Predictions:'.upper())
model.transform(final_data).select('prediction').show()
