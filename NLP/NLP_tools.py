# -*- coding: utf-8 -*-
"""
NLP functions

Author: Damian
"""

from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('nlp').getOrCreate()

# Tokenazation: taking text and breaking it into individual elemnts, e.g., words
# Regular expression for breaking the text

from pyspark.ml.feature import Tokenizer,RegexTokenizer
from pyspark.sql.functions import col,udf, count, explode
from pyspark.sql.types import IntegerType

sen_df = spark.createDataFrame([(0, 'Hi I heard about Spark'), (1, 'I wish jave could use case classes'),(2, 'Logistic, regression, models, are, neat')],['id','sentence'])

tokenizer = Tokenizer(inputCol='sentence', outputCol='words')
# The 'pattern' parameter extracts tokens based on the pattern you provide 
regex_tok = RegexTokenizer(inputCol='sentence', outputCol='words', pattern='||w')

# count_toks = udf(lambda words:len(words), IntegerType())
tokenized = tokenizer.transform(sen_df)
tokenized.show()

# tokenized.withColumn('N_tokens', count(col('words'))).show()

tokenized_2 = tokenized.withColumn('exploded',explode(tokenized['words'])).groupBy('id').count()
tok_final = tokenized.join(tokenized_2, on=['id'])
tok_final = tok_final.select(['id', 'sentence', 'words', tok_final['count'].alias('N_tokens')])
# tok_final_2 = tok_final.selectEpr('count as N_tokens')
print('final tokenized data'.upper())
tok_final.show()

from pyspark.ml.feature import StopWordsRemover

sentence_DF = spark.createDataFrame([(0, ['I', 'saw', 'the', 'green', 'horse']), (1, ['Mary', 'had', 'a', 'little', 'lamb'])], ['id', 'tokens']) 
print('initial data frame'.upper())
sentence_DF.show()

remover = StopWordsRemover(inputCol='tokens', outputCol='filtered')
print('data frame where common words are removed'.upper())
remover.transform(sentence_DF).show()

# n-grams: sequence of tokens of consecutive 'n' words  
from pyspark.ml.feature import NGram
ngram = NGram(n=2, inputCol='words', outputCol='grams')
ngram.transform(tok_final).show() 
tok_final_n = ngram.transform(tok_final)
tok_final_n.select('grams').show(truncate=False)
# The n-grams help explore relationships between close words

from pyspark.ml.feature import HashingTF, IDF

hashing_tf = HashingTF(inputCol='words',outputCol='rawFeatures')
feature_data = hashing_tf.transform(tok_final)
idf = IDF(inputCol='rawFeatures',outputCol='Features')
idf_model = idf.fit(feature_data)
rescaled_data = idf_model.transform(feature_data)

# See how words were transformed into numbers, this is ready for a supervides machine learning algorithm
rescaled_data.select('id','Features').show(truncate=False)

############

from pyspark.ml.feature import CountVectorizer
df = spark.createDataFrame([(0, ['hello', 'are', 'you', 'man']),(1,['hello', 'hello', 'man', 'I', 'am', 'great', 'I', 'am', 'fantastic', 'you', '?', 'you', 'okay', '?'])],['id', 'tokens'])
cv = CountVectorizer(inputCol='tokens', outputCol='countVec', vocabSize=10, minDF=2.0) 
# minDF: minimum number of documents in which a word must belong in order to be considered as a feature
cv.fit(df).transform(df).show(truncate=False)

print("Note that 'hello' and 'you' were repeated twice in the last document")
print('Only words in both documents are considered!')
