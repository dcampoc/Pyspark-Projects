# -*- coding: utf-8 -*-
"""
Spam detector

Author: Damian
"""
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('spam').getOrCreate()
# Data is separated by tabs and not by commas, that's why we use '\t'
data = spark.read.csv('smsspamcollection/SMSSpamCollection', inferSchema=True, sep='\t')
# Change name of columns 
data = data.withColumnRenamed('_c0', 'class').withColumnRenamed('_c1','text')

from pyspark.sql.functions import length
data = data.withColumn('length', length(data['text']))
print('average lenght of good emails and spams')
data.groupBy('class').mean().show()

from pyspark.ml.feature import Tokenizer, StopWordsRemover,CountVectorizer, IDF, StringIndexer

tokenizer = Tokenizer(inputCol='text', outputCol='token_text')
stop_remove = StopWordsRemover(inputCol='token_text', outputCol='stop_token')
count_vec = CountVectorizer(inputCol='stop_token', outputCol='c_vec')
idf = IDF(inputCol='c_vec', outputCol='tf_idf')
# We convert 'spam' and 'ham' into numeric features (eros and ones)
classes_to_numeric = StringIndexer(inputCol='class', outputCol='label')

from pyspark.ml.feature import VectorAssembler
data_features = VectorAssembler(inputCols=['tf_idf', 'length'], outputCol='features')

data = data.replace(['spam','ham'], ['1', '0'])
data = data.withColumn('class_num', data['class'].cast('float'))

data_1 = tokenizer.transform(data)
data_1 = stop_remove.transform(data_1)
data_1 = count_vec.fit(data_1).transform(data_1)
data_1 = idf.fit(data_1).transform(data_1)
data_1 = data_features.transform(data_1)

data_final = data_1.select('class', 'class_num', 'features')
print('Final data set up'.upper())
data_final.show()

train_data, test_data = data_final.randomSplit([0.7, 0.3])
# Whatever ml classification model can be used here
from pyspark.ml.classification import LogisticRegression
log_reg = LogisticRegression(labelCol='class_num', featuresCol='features')
spam_detector = log_reg.fit(train_data)
test_results = spam_detector.transform(test_data)
test_results.columns 
# Select columnsb to be compared
test_results = test_results.select('class','class_num', 'prediction')
test_results.show()

results = test_results.withColumn('check', test_results['class_num'] == test_results['prediction'])
acc = results.groupBy('check').count()
acc.show()

print("In the DataFrame above, 'true' and 'false' represent correct and incorrect classifications respectively")
num = results.groupBy('class_num').count()
print('Consider that {} are spam and {} are ham'.format(num.head(2)[0][1], num.head(2)[1][1]))

#################################################################################
from pyspark.sql.functions import abs
num_1 = test_results.groupBy('class').count()

num_2 = results.groupBy('class').sum()
num_2 = num_2.withColumn('diff',abs(num_2['sum(class_num)']-num_2['sum(prediction)']))

print('Percent accuracy at detecting ham: {}'.format(100 - (100 * num_2.head(2)[0][3]/num_1.head(2)[0][1])))
print('\n')
print('Percent accuracy at detecting spam: {}'.format(100 - (100 * num_2.head(2)[1][3]/num_1.head(2)[1][1])))

#from pyspark.ml.evaluation import MulticlassClassificationEvaluator
#acc_eval = MulticlassClassificationEvaluator(labelCol='class_num', predictionCol='prediction')
#acc_f1 = acc_eval.evaluate(results)
