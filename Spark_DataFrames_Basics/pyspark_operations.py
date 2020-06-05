#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 30 12:56:35 2020

@author: damian
"""
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('ops').getOrCreate()

df = spark.read.csv('appl_stock.csv',inferSchema=True,header=True)
df.printSchema()
df.show()

''' SQL style
# Filter data whose Close information is less than 500
df.filter("Close < 500").show()
# Select (grab) only the information from the field Open when Close is less than 500
df.filter("Close < 500").select('Open').show()
df.filter("Close < 500").select(['Open','Close']).show()
'''

df.filter(df['Close']<500).show()
df.filter(df['Close']<500).select('Open').show()
df.filter(df['Close']<500).select(['Open', 'Close']).show()

print('Dates where the actions Closed in less than 200 and open in more than 200:'.upper())
df.filter( (df['Close']< 200) & (df['Open']>200) ).select('Date').show()

df.filter(df['Low'] == 197.16).show()
# df.filter(df['Low'] == 197.16).collect()

# GroupBy and Aggregate data
print('second exersize'.upper())
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('aggs').getOrCreate()
df_2 = spark.read.csv('sales_info.csv',inferSchema=True, header=True)
df_2.show()
df_2.printSchema()

# Mean of each company, sum, min and max per company
df_2.groupby("Company").mean().show()
df_2.groupby("Company").sum().show()
df_2.groupby("Company").min().show()
df_2.groupby("Company").max().show()
# How many rows per company?
df_2.groupby("Company").count().show()

# What about if I do not want to distinguish among companies?
print('Total sales!'.upper())
# See that the argument of agg is a dictionary whose key is the field of imprtance and the value is the operation that I want to perform
df_2.agg({'Sales':'sum'}).show()
print('Average sales:'.upper())
df_2.agg({'Sales':'mean'}).show()

#The following lines are more realistic in case of manipulating data:
df_comp = df_2.groupby("Company")
df_comp.agg({'Sales':'mean'}).show()

# The following option could be more handy than using dictionaries
from pyspark.sql.functions import countDistinct,avg,stddev,format_number
df_2.select(countDistinct('Company').alias('N. of companies')).show()

df_2.select(avg('Sales')).show()
df_2.select(avg('Sales').alias('Average sales man!')).show()
df_2.select(stddev('Sales').alias('STD man!')).show()
print('testing different formats'.upper())
# First we refer to the column as "std"
df_2_STD = df_2.select(stddev('Sales').alias('std'))
# Then, we change the format of std to whow only two digits and change the name of the column as "formated std"
df_2_STD.select(format_number('std',2).alias('formated std')).show()

# Ascendeing order by a given column
df_2.orderBy('Sales').show()
df_2.orderBy('Company').show()
df_2.orderBy('Person').show()

# Descending order by a given column
df_2.orderBy(df_2['Sales'].desc()).show()

# Missing data
print('Third exersize'.upper())
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('missData').getOrCreate()
df_3 = spark.read.csv('ContainsNull.csv',inferSchema=True, header=True)
df_3.show()
# Dropping missing data 
df_3.na.drop().show()
# Dropping rows with less than 2 real values (they must have at least 2 real values)
df_3.na.drop(thresh=2).show()

# Default dropping
df_3.na.drop(how='any').show()

# Dropping the rows with all null values
df_3.na.drop(how='all').show()

# Default dropping based on only a column
df_3.na.drop(subset=['Sales']).show()

# Fill values in null instances of a particular row
df_3.na.fill('No name',subset=['Name']).show()
df_3.na.fill(0,subset=['Sales']).show()

from pyspark.sql.functions import mean
# mean_val = df_3.select(mean(df_3['Sales'])).collect()[0][0]
mean_val = 400.5

df_3.na.fill(mean_val, subset=['Sales']).show()


# Dates 
print('fourth exersize'.upper())
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('dates').getOrCreate()
df_4 = spark.read.csv('appl_stock.csv',inferSchema=True, header=True)
df_4.show()
df_4.printSchema()
df_4.select(['Date','Open']).show()

from pyspark.sql.functions import (dayofmonth,hour,dayofyear,month,year,weekofyear,format_number,date_format)
df_4.select(dayofmonth(df_4['Date'])).show(5)
df_4.select(month(df_4['Date'])).show(5)

# Create a new column and save the new df
new_df = df_4.withColumn("Year",year(df_4['Date']))

# We group rows containing the same years and create a single row out of them 
# containing the mean of all column data 
# In this case we only select the columns "Year" and "avg(Close)"
result = new_df.groupBy("Year").mean().select(["Year","avg(Close)"])
result.show()
print("Let's apply some formatting".upper())
result = result.withColumnRenamed("avg(Close)","avg")
result = result.select(['Year',format_number('avg',2).alias('Average closing price')])
#result = result.withColumnRenamed('format_number(avg, 2)','Average closing price')
result.show()






















