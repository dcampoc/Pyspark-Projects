# -*- coding: utf-8 -*-
"""
Three different examples of binary classification using logistic regression
"""
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('logreg').getOrCreate()
from pyspark.ml.classification import LogisticRegression
my_data = spark.read.format('libsvm').load('sample_libsvm_data.txt')

my_data.show()
print('Note that the labels suggest a binary classification task'.upper())

lg_model = LogisticRegression()
lg_train, lg_test = my_data.randomSplit([0.7, 0.3])

fitted_lg = lg_model.fit(lg_train)
lg_summary = fitted_lg.summary
pred_train = lg_summary.predictions
pred_train.printSchema()
pred_train.columns

eval_test = fitted_lg.evaluate(lg_test)
pred_test = eval_test.predictions
pred_test.columns

results = pred_test.select('label', 'rawPrediction', 'probability', 'prediction')
results.show()

'''
print('Explore the evaluation on testing data'.upper())

from pyspark.ml.evaluation import (BinaryClassificationEvaluator, MulticlassClassificationEvaluator)

my_eval = BinaryClassificationEvaluator()
area_roc = my_eval.evaluate(results)
'''
results = results.withColumn('check', results['label'] == results['prediction'])
acc = results.groupBy('check').count()
acc.show()
print("In the DataFrame above, 'true' and 'false' represent correct and incorrect classifications respectively")
print('\n')
print('\n')

print('second exercise'.upper())
data = spark.read.csv('titanic.csv', inferSchema=True, header=True)
data.printSchema()
# Select only useful data
data = data.select('Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked')
# Dealing with missing data by dropping it 
data = data.na.drop()

from pyspark.ml.classification import LogisticRegression
# Predict whether the titanic population survuved or not 
lg_titanic = LogisticRegression(featuresCol='features', labelCol='Survived')

from pyspark.ml.feature import (VectorAssembler, VectorIndexer, OneHotEncoder, StringIndexer)
'''
# Using a pipeline and substuting categorical evaluation with numerical values
# Transformation of categories (strings) into numbers
gender_indexer = StringIndexer(inputCol='Sex',outputCol='SexIndex')
gender_encoder = OneHotEncoder(inputCol='SexIndex',outputCol='SexVec')

embark_indexer = StringIndexer(inputCol='Embarked',outputCol='EmbarkIndex')
embark_encoder = OneHotEncoder(inputCol='EmbarkIndex',outputCol='EmbarkVec')

assembler = VectorAssembler(inputCols=['Pclass','SexVec', 'EmbarkVec', 'Age', 'SibSp', 'Parch', 'Fare'], outputCol='features')

from pyspark.ml import Pipeline 

# Creating pipeline (for managing different stages of the ML process)
pipeline = Pipeline(stages=[gender_indexer,embark_indexer,gender_encoder,embark_encoder,assembler,lg_titanic])

train_data, test_data = data.randomSplit([0.7, 0.3])

fit_model = pipeline.fit(train_data)

test = fit_model.transform(test_data)

from pyspark.ml.evaluation import BinaryClassificationEvaluator
eval = BinaryClassificationEvaluator(rawPrediction='prediction',labelCol='Survived')
AUC = eval.evaluation(test)
'''

assembler_simple = VectorAssembler(inputCols=['Pclass', 'Age', 'SibSp', 'Parch', 'Fare'], outputCol='features')
data_new = assembler_simple.transform(data)

data_new = data_new.select('features','Survived')
train_data, test_data = data_new.randomSplit([0.7, 0.3])

fitted_lg = lg_titanic.fit(train_data)
lg_summary = fitted_lg.summary
pred_train = lg_summary.predictions
pred_train.printSchema()
pred_train.columns

eval_test = fitted_lg.evaluate(test_data)
pred_test = eval_test.predictions
pred_test.columns

results = pred_test.select('Survived', 'rawPrediction', 'probability', 'prediction')
results.show()

results = results.withColumn('check', results['Survived'] == results['prediction'])
acc = results.groupBy('check').count()
acc.show()
print("In the DataFrame above, 'true' and 'false' represent correct and incorrect classifications respectively")

num = results.groupBy('Survived').count()
print('Consider that {} passengers survived and {} passengers died'.format(num.head(2)[0][1], num.head(2)[1][1]))

print('third exersize'.upper())
# Create a model to predict whether a customer will hurn or not
df = spark.read.csv('customer_churn.csv', inferSchema=True, header=True)
df = df.na.drop()
df.columns
df_new = df.select('Age', 'Total_Purchase', 'Years', 'Num_Sites', 'Onboard_date', 'Churn')

from pyspark.sql.functions import (dayofmonth,hour,dayofyear,month,year,weekofyear,format_number,date_format)

df_new = df_new.withColumn('Onboard_month', month(df_new['Onboard_date']).cast('double'))
df_new = df_new.withColumn('Onboard_year', year(df_new['Onboard_date']).cast('double'))

from pyspark.ml.feature import VectorAssembler
assembler = VectorAssembler(inputCols=['Age', 'Total_Purchase', 'Years', 'Num_Sites'], outputCol='features')
data_new = assembler.transform(df_new)

data_final = data_new.select('features','Churn')
lg_customer_churn = LogisticRegression(featuresCol='features', labelCol='Churn')


train_data, test_data = data_new.randomSplit([0.7, 0.3])

fitted_lg = lg_customer_churn.fit(train_data)
lg_summary = fitted_lg.summary
pred_train = lg_summary.predictions
pred_train.printSchema()
pred_train.columns

eval_test = fitted_lg.evaluate(test_data)
pred_test = eval_test.predictions
pred_test.columns

results = pred_test.select('Churn', 'rawPrediction', 'probability', 'prediction')
results.show()

results = results.withColumn('check', results['Churn'] == results['prediction'])
acc = results.groupBy('check').count()
acc.show()
print("In the DataFrame above, 'true' and 'false' represent correct and incorrect classifications respectively")
num = results.groupBy('Churn').count()
print('Consider that {} belong to a class and {} to the other one'.format(num.head(2)[0][1], num.head(2)[1][1]))
print('\n')
print('\n')
print("Now let's test some new custumers")

df = spark.read.csv('new_customers.csv', inferSchema=True, header=True)
df = df.na.drop()
df.columns
data_new = assembler.transform(df)
eval_test_final = fitted_lg.transform(data_new)
eval_test_final.select('Company', 'prediction').show()




