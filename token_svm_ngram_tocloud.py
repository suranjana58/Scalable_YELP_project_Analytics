from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
from pyspark import SparkConf
from pyspark.sql.functions import count
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
import csv
import pandas as pd
from pyspark.sql.functions import regexp_replace
from pyspark.mllib.classification import SVMWithSGD
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.linalg import Vector as MLLibVector, Vectors as MLLibVectors
conf = SparkConf().setMaster("local[*]").set("spark.executer.memory", "4g")
sc = SparkContext.getOrCreate()
spark = SparkSession(sc)
spark.sparkContext.setLogLevel("WARN")

"""**Load all datasets**"""

#1. Clean the dataset
df_review = spark.read.format("csv").option("header", "true").option("multiline","true").load("yelp_review.csv")
df_review.show()

#Let us print schema for all
df_review.printSchema()

df_review = df_review.withColumn("label", df_review["stars"].cast("double"))
df_review= df_review.withColumn("useful", df_review['useful'].cast("int")) 
df_review= df_review.withColumn("funny", df_review['funny'].cast("int"))
df_review= df_review.withColumn("cool", df_review['cool'].cast("int"))
df = df_review.select('text', 'label')

#df_review = df_review.withColumn("stars", regexp_replace("stars", "[^0-9.]+", ""))
#df_review = df_review.withColumn('useful', regexp_replace('useful', '[^0-9]', ''))
#df_review = df_review.withColumn('cool', regexp_replace('cool', '[^0-9]', ''))
#df_review = df_review.withColumn('funny', regexp_replace('funny', '[^0-9]', ''))

n_rows = df_review.count()
n_cols = len(df_review.columns)
print('before removing NAN')
print("Shape of the DataFrame: ({}, {})".format(n_rows, n_cols))

df_review = df_review.dropna(subset=['text','label','useful','funny','cool'])
df_review = df_review.dropna(subset=["stars"])

df_review = df_review.filter(df_review.label.isin(1.0,2.0,3.0,4.0,5.0))
n_rows = df_review.count()
n_cols = len(df_review.columns)
#Print the shape of the DataFrame
print('after removing NAN')
print("Shape of the DataFrame: ({}, {})".format(n_rows, n_cols))

df_review.groupBy('useful').count().show()
df_review.groupBy('funny').count().show()
df_review.groupBy('cool').count().show()

"""**Removing stopwords ,tokenizing**"""

# Define the remove_punct() function
def remove_punct(text):
    if text is None:
        return ""
    # remove punctuation from text
    return re.sub(r'[^\w\s]','',text)

import re
from pyspark.sql.functions import udf
import random

#Define the remove_punct() function
def remove_punct(text):
    if text is None:
        return None
    else:
        # remove punctuation from text
        return re.sub(r'[^\w\s]','',text)

#Define the convert_rating() function
def convert_rating(rating):
    rating = int(rating)
    if rating >=4: return 1
    else: return 0

#Define the UDFs
remove_punct_udf = udf(remove_punct)
convert_rating_udf = udf(convert_rating)

#Apply the UDFs to the DataFrame
review_df = df_review.select('review_id','user_id','label', remove_punct_udf('text'), convert_rating_udf('stars')) \
                             .withColumnRenamed('<lambda>(text)', 'text') \
                             .withColumnRenamed('<lambda>(stars)', 'label') \
                             .dropna() \
                             .limit(3002194)

#Show the resulting DataFrame
review_df.show()

from pyspark.ml.feature import Tokenizer
from pyspark.ml.feature import StopWordsRemover

#tokenize
tok = Tokenizer(inputCol="remove_punct(text)", outputCol="words")
review_tokenized = tok.transform(review_df)

#remove stop words
stopword_rm = StopWordsRemover(inputCol='words', outputCol='words_nsw')
review_tokenized = stopword_rm.transform(review_tokenized)

review_tokenized.show(5)

def convert_rating(rating):
    try:
        return int(float(rating))
    except ValueError:
        print(f"Invalid value encountered: {rating}")
        return None

from pyspark.sql.functions import udf
from pyspark.sql.types import IntegerType

#Define the UDF
convert_rating_udf = udf(convert_rating, IntegerType())

#Apply the UDF to the 'stars' column
review_tokenized = review_tokenized.withColumn('convert_rating(stars)', convert_rating_udf('convert_rating(stars)'))

def convert_rating(rating):
    return int(float(rating))

"""**SVM**"""

from pyspark.sql.functions import col, regexp_replace
from pyspark.ml.feature import HashingTF, IDF
from pyspark.ml.classification import LinearSVC
from pyspark.ml import Pipeline

#Remove punctuations from the 'text' column and convert the 'stars' column to numeric type
cleaned_df = review_tokenized.select(regexp_replace(col("convert_rating(stars)"), "[^a-zA-Z\\s]", "").alias("text"), 
                       col("convert_rating(stars)").cast("double"))

#Split the data into training and test sets
train_df, test_df = cleaned_df.randomSplit([0.7, 0.3], seed=42)

#Tokenize the words in the 'text' column and remove stop words
tokenizer = Tokenizer(inputCol="text", outputCol="words")
stopword_remover = StopWordsRemover(inputCol=tokenizer.getOutputCol(), outputCol="words_nsw")

#Create a HashingTF object to create feature vectors from tokenized words
hashingTF = HashingTF(inputCol="words_nsw", outputCol="rawFeatures")

#Create an IDF object to calculate the IDF of each term in the document
idf = IDF(inputCol="rawFeatures", outputCol="features")

#Create a Pipeline object that combines the tokenizer, stopword_remover, HashingTF, and IDF stages
pipeline = Pipeline(stages=[tokenizer, stopword_remover, hashingTF, idf])

"""This SVM model code takes time to run"""

#Fit the pipeline on the training data and transform the test data to get tfidf_df
tfidf_df = pipeline.fit(train_df).transform(test_df)

#Train a SVM model using the tfidf_df
svm = LinearSVC(featuresCol="features", labelCol="convert_rating(stars)")
svm_model = svm.fit(tfidf_df)

from pyspark.ml.evaluation import MulticlassClassificationEvaluator

#Make predictions on the test data using the trained SVM model
predictions = svm_model.transform(tfidf_df)

#Evaluate the predictions using MulticlassClassificationEvaluator
evaluator = MulticlassClassificationEvaluator(predictionCol="prediction", labelCol="convert_rating(stars)", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)

#Print the accuracy
print("SVM Accuracy:", accuracy)



"""**N-GRAM**"""

from pyspark.ml.feature import NGram
from pyspark.sql.functions import col

n = 3  #Change n to the desired value of n for n-grams ,here 3 for trigram
ngram = NGram(n=n, inputCol='words', outputCol='ngram')
add_ngram = ngram.transform(review_tokenized.select(col('words'), col('convert_rating(stars)')))
add_ngram.show(5)

"""N gram modelling takes some time to run"""

from pyspark.ml.classification import LinearSVC
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import CountVectorizer, IDF, VectorAssembler, StringIndexer
from pyspark.sql.functions import col

#create CountVectorizer object
cv_ngram = CountVectorizer(inputCol='ngram', outputCol='tf_ngram')
cvModel_ngram = cv_ngram.fit(add_ngram)
cv_df_ngram = cvModel_ngram.transform(add_ngram)

#create IDF model and transform the data
idf_ngram = IDF(inputCol='tf_ngram', outputCol='tfidf_ngram')
tfidfModel_ngram = idf_ngram.fit(cv_df_ngram)
tfidf_df_ngram = tfidfModel_ngram.transform(cv_df_ngram)

#VectorAssembler to combine features
assembler = VectorAssembler(inputCols=['tfidf_ngram'], outputCol='features')

#convert the label column to a numeric type
label_indexer = StringIndexer(inputCol="convert_rating(stars)", outputCol="label")
label_indexer_model = label_indexer.fit(tfidf_df_ngram)
tfidf_df_ngram = label_indexer_model.transform(tfidf_df_ngram)

#transform the data with the assembler
data = assembler.transform(tfidf_df_ngram).select(['features', 'label'])

#split the data into training and test sets
(train_data, test_data) = data.randomSplit([0.7, 0.3], seed=12345)

#fit SVM model of trigrams
svm = LinearSVC(maxIter=50, regParam=0.3, labelCol='label')
svm_model = svm.fit(train_data)

#make predictions on test data
predictions = svm_model.transform(test_data)

#evaluate the model
evaluator = MulticlassClassificationEvaluator(predictionCol="prediction", labelCol="label", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)

print('N gram ')
print()
print("Accuracy = %g" % accuracy)

"""Post finding accuracy of two models we will find group of words that occur most frequently in Reviews

"""

#split into training & testing set
splits_ngram = tfidf_df_ngram.select(['tfidf_ngram', 'label']).randomSplit([0.7,0.3],seed=100)
train_ngram = splits_ngram[0].cache()
test_ngram = splits_ngram[1].cache()

#convert to LabeledPoint vectors
train_lb_ngram = train_ngram.rdd.map(lambda row: LabeledPoint(row[1], MLLibVectors.fromML(row[0])))
test_lb_ngram = train_ngram.rdd.map(lambda row: LabeledPoint(row[1], MLLibVectors.fromML(row[0])))

#fit SVM model of trigrams
numIterations = 50
regParam = 0.3
svm = SVMWithSGD.train(train_lb_ngram, numIterations, regParam=regParam)

vocabulary_ngram = cvModel_ngram.vocabulary
coefficients_ngram = svm_model.coefficients.toArray()
svm_coeffs_df_ngram = pd.DataFrame({'ngram': vocabulary_ngram, 'weight': coefficients_ngram}) #ngram and weight table

"""**Top Words occuring in 5 Star review**"""

print(svm_coeffs_df_ngram.sort_values('weight').head(20))

"""***Top Words occuring in 1 Star review ***"""

print(svm_coeffs_df_ngram.sort_values('weight', ascending=False).head(20))





