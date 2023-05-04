from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
from pyspark import SparkConf

from pyspark.ml.clustering import KMeans
from pyspark.ml.linalg import Vectors

from pyspark.ml.feature import Tokenizer,RegexTokenizer
from pyspark.ml.feature import StopWordsRemover
from pyspark.ml.feature import HashingTF, IDF 
from pyspark.ml import Pipeline
from pyspark.sql.functions import col,udf
from pyspark.sql.types import IntegerType
#from pyspark.ml.feature import CountVectorizer 
from pyspark.sql.functions import sum,avg,max,count

# Started Spark Context with Spark Session 
conf = SparkConf().setMaster("local").setAppName("kMeansNoMLlib")
sc = SparkContext.getOrCreate()
spark = SparkSession(sc)

# Read in the external file
df_review = spark.read.format("csv").option("header", "true").option("multiline","true").load("yelp_review.csv")
df_review.show()

df_review= df_review.withColumn("label", df_review["stars"].cast("double"))   
read_data = df_review.select('text','label','useful','funny','cool')
df = read_data.dropna(subset=['text','label','useful','funny','cool'])
df = df.filter(df.label.isin(1.0,2.0,3.0,4.0,5.0))

#text processing
tokenizer = Tokenizer(inputCol="text", outputCol="tokens")
remover = StopWordsRemover(inputCol="tokens", outputCol="stopWordsRemovedTokens")
TF = HashingTF (inputCol="stopWordsRemovedTokens", outputCol="tfFeatures")
idf = IDF(inputCol="tfFeatures", outputCol="features")

kmeans = KMeans().setK(2).setSeed(1234)
pipeline = Pipeline(stages = [tokenizer,remover,TF,idf,kmeans])
km_model = pipeline.fit(df)
clustertable=km_model.transform(df)
print('the first prediction')
result_pred=clustertable.groupBy("prediction").count().show()

#check prediction=1
#print('prediction=1')
#clustertable.filter(clustertable.prediction.isin(1.0)).show()

################################################################
#only keep prediction=0
#df = clustertable.filter(clustertable.prediction.isin(0.0))
#df=df.select('text','label','useful','funny','cool')
#tokenizer = Tokenizer(inputCol="text", outputCol="tokens")
#remover = StopWordsRemover(inputCol="tokens", outputCol="stopWordsRemovedTokens")
#TF = HashingTF (inputCol="stopWordsRemovedTokens", outputCol="tfFeatures")
#idf = IDF(inputCol="tfFeatures", outputCol="features")
#kmeans = KMeans().setK(2).setSeed(1234)
#pipeline = Pipeline(stages = [tokenizer,remover,TF,idf,kmeans])
#km_model = pipeline.fit(df)
#clustertable=km_model.transform(df)
#result_pred=clustertable.groupBy("prediction").count().show()

################################################################
#only keep prediction=0
#df = clustertable.filter(clustertable.prediction.isin(0.0))
#df=df.select('text','label','useful','funny','cool')
#tokenizer = Tokenizer(inputCol="text", outputCol="tokens")
#remover = StopWordsRemover(inputCol="tokens", outputCol="stopWordsRemovedTokens")
#TF = HashingTF (inputCol="stopWordsRemovedTokens", outputCol="tfFeatures")
#idf = IDF(inputCol="tfFeatures", outputCol="features")
#kmeans = KMeans().setK(5).setSeed(1234)
#pipeline = Pipeline(stages = [tokenizer,remover,TF,idf,kmeans])
#km_model = pipeline.fit(df)
#clustertable=km_model.transform(df)
#result_pred=clustertable.groupBy("prediction").count().show()

###############################################################
#after fiting model, find two clusters and outlier
countTokens = udf(lambda words: len(words), IntegerType())
df_new = clustertable.filter(clustertable.prediction.isin(0.0,1.0))
df_new=df_new.select('tokens','label','useful','funny','cool','prediction').withColumn("count", countTokens(col("tokens")))
df_new.schema
df_new.groupBy("prediction").agg(avg('label').alias('avglabel'),avg('count').alias('avgtoken')).show()

