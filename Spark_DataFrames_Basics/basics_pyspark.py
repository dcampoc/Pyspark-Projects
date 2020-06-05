# -*- coding: utf-8 -*-
"""
Spyder Editor

@author: damian
"""
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('Basics').getOrCreate()
df = spark.read.json('people.json')
df.show()

df.printSchema()
df.columns
from pyspark.sql.types import StructField,StringType,IntegerType,StructType

#Define an schema (what are the expected values of the columns (data)) True means that it may take null values. Remember that “age” and “name” are the fields (keys) of the dataframe
data_schema = [StructField('age',IntegerType(),True),StructField('name',StringType(),True)]
final_struc = StructType(fields=data_schema)

#Read the file with a schema in mind 
df = spark.read.json('people.json',schema=final_struc)
df.show()
df.printSchema()

# Select vs grab the data (it is preferable to select)
print('First option: Grab a column:')
df_age = df['age']
print(type(df_age))
print('Second option: Select a column')  
df_age_select = df.select('age')
print(type(df_age_select))
df_age_select.show()
print('See that the second option actually extracts the data we are interested in')

# Getting rows (objects)
df.head(2)
df.head(2)[0]

# Take multiple columns (provide a list of keys)
df.select(['age','name'])

# Addiding new columns (the orginal df remains the same though)
df.withColumn('newage',df['age']).show()
df.withColumn('doubleage',df['age']*2).show()
df.withColumn('squaredage',df['age']**2).show()

# Change the name of the columns
df.withColumnRenamed('age','newage').show()

# Registered temporary view (To make sql queries)
df.createOrReplaceTempView('persons')
'''
# Advantages of using SQL (this approach won't be used the course) but it is useful to know that spark supports SLQ queries
results = spark.sql("SELECT * FROM persons")
results.show()
new_results = spark.sql("SELECT * FROM persons HERE age=30")
new_results.show()
'''