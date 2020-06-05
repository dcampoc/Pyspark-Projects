#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 21:00:55 2020

@author: damian
"""

print('Start a simple Spark session and load data'.upper())
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('exc').getOrCreate()
df = spark.read.csv('walmart_stock.csv',inferSchema=True,header=True)
print('What are the column names?'.upper())
df.columns

print('What is the schema?'.upper())
df.printSchema()

print('Print out the first 5 rows'.upper())
df.show(5)
print('OR'.upper())
for i in df.head(5):
    print(i)
    print('\n')
    
print('Reduce the decimals')
#df.describe()
from pyspark.sql.functions import format_number
df_new = df.select(['Date', format_number('Open', 2).cast('float').alias('Open'), format_number('High', 2).cast('float').alias('High'), format_number('Low', 2).cast('float').alias('Low') , format_number('Close', 2).cast('float').alias('Close'), 'Volume',format_number( 'Adj Close',2).cast('float').alias('Adj Close')])

print('Create new data frame with an extra column Volume/High')
df_new = df_new.withColumn('V/H ratio',format_number(df_new['Volume']/df_new['High'],2))

print('What day preseted the highest peak in High'.upper())
df_ord_H = df_new.orderBy(df_new['High'].desc())
df_ord_H.select('Date').show(1)
print('OR')
df_new.orderBy(df_new['High'].desc()).head(1)[0][0]

print('What about day preseted the 2nd highest peak in High?'.upper())
# Note that I need the second row, i.e., head(2)
df_new.orderBy(df_new['High'].desc()).head(2)[1][0]

print('What is the mean of the Close column?'.upper())
from pyspark.sql.functions import mean
df_new.select(format_number(mean('Close'),2).alias('avg')).show()

print('What is the min and max values of the Volume column?'.upper())
from pyspark.sql.functions import min, max
df_new.select(format_number(min('Volume'),2).alias('min_volume'), format_number(max('Volume'),2).alias('max_volume')).show()

print('How many days had the Close lower than 60 dollars'.upper())
from pyspark.sql.functions import count
df_filt = df_new.filter(df_new['Close']<60)
df_filt.select(count(df_filt['Close'])).show()

print('Percentage of days where High>80'.upper())
df_filt = df_new.filter(df_new['High']>80)
df_filt = df_filt.select(count(df_filt['Date']).alias('days_80'))

df_filt2 = df_new.select(count(df_new['Date']).alias('total_days'))

percentage = 100 * df_filt.head(1)[0][0]/df_filt2.head(1)[0][0]
print('The percentage is {}'.format(percentage))

# Alternative method in which dataframes are fused (but the command .show() does not work at the end)
# ta = df_filt.alias('ta')
# tb = df_filt2.alias('tb')
# df_join = ta.join(tb, ta.days_80 == tb.total_days, 'inner')
# df_join.select(df_join['days_80'])/df_join['total_days'])
# df_join.select(df_join['days_80']/df_join['total_days'])

print('What is the pearson coefficient between High and Volume'.upper())
from pyspark.sql.functions import corr
df_new.select(corr('High', 'Volume')).show()

print('What is the max High per year'.upper())
from pyspark.sql.functions import year
df_year = df_new.withColumn('Year', year(df_new['Date']))
df_year = df_year.groupby('Year').max()
df_year = df_year.select('Year',df_year['max(High)'].alias('Maxium High'))
df_year.orderBy('Year').show()

print('What is average Close per month'.upper())
from pyspark.sql.functions import month
df_month = df_new.withColumn('Month', month(df_new['Date'])).groupby('Month').mean()
# df_month = df_month.groupby('Month').mean()
df_month = df_month.select('Month', df_month['avg(Close)'].alias('Average Close'))
df_month.orderBy('Month').show()



