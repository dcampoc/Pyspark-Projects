#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Examples of different tree ML algorithms

@author: damian
"""

from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('tree').getOrCreate()
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier, GBTClassificationModel, DecisionTreeClassifier

# For regression tasks,  it is possible to use the following import scheme
#from pyspark.ml.regression import DecisionTreeRegressor, GBTRegressionModel, RandomForestRegressor

data = spark.read.csv('College.csv',inferSchema=True, header=True)
data.show()
assembler = VectorAssembler(inputCols=['Apps', 'Accept', 'Enroll', 'Top10perc', 'Top25perc', 'F_Undergrad', 'P_Undergrad', 'Outstate', 'Room_Board', 'Books', 'Personal', 'PhD', 'Terminal', 'S_F_Ratio', 'perc_alumni', 'Expend', 'Grad_Rate'],outputCol='features')
data = assembler.transform(data)
# Transform output data from string type to integer type 
data = data.replace(['Yes','No'], ['1', '0'])
data = data.withColumn('Private_num', data['Private'].cast('integer'))
data_final = data.select('features', 'Private_num')
'''
from pyspark.ml.feature import StringIndexer
indexer = StringIndexer(inputCol='Private',outputCol='Private_num')
data = indexer.fit(data)
'''
train_data, test_data = data.randomSplit([0.7, 0.3])
# Initialize 3 different models  
decision_tree = DecisionTreeClassifier(featuresCol='features',labelCol='Private_num')
# The default number of trees in the random forest is 20, which is kind of low
random_forest = RandomForestClassifier(featuresCol='features',labelCol='Private_num',numTrees=150)
gradient_boost = GBTClassificationModel(featuresCol='features',labelCol='Private_num')

decision_tree_fit = decision_tree.fit(train_data)
random_forest_fit = random_forest.fit(train_data)
gradient_boost_fit = gradient_boost.fit(train_data)

decision_tree_pred = decision_tree_fit.transform(test_data)
random_forest_pred = random_forest_fit.transform(test_data)
gradient_boost_pred = gradient_boost_fit.transform(test_data)

from pyspark.ml.evaluation import MulticlassClassificationEvaluator
acc_eval = MulticlassClassificationEvaluator(labelCol='Private_num', metricName='accuracy')

print('Decision tree performance: {}'.format(acc_eval.evaluate(decision_tree_pred)))
print('\n')
print('Random forest performance: {}'.format(acc_eval.evaluate(random_forest_pred)))
print('\n')
print('gradient_boost_pred: {}'.format(acc_eval.evaluate(gradient_boost_pred)))
print('\n')
print('Low performances in gradient boosting trees w.r.t a regular decision tree can be caused by a wrong choise of hyperparameters (check the default parameters of pyspark)'.upper())
# Ability to calculate feature importance (the higher the number the more importance it is)
# decision_tree_fit.featureImportances


