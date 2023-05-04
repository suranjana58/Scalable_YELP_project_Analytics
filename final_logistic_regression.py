from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
from pyspark import SparkConf
from pyspark.ml.feature import Tokenizer, HashingTF, IDF
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql.functions import udf, col, when, monotonically_increasing_id
from pyspark.sql.types import DoubleType
from pyspark.sql.functions import col, when

conf = SparkConf().setMaster("local[*]").set("spark.executer.memory", "4g")

sc = SparkContext(conf=conf)
spark = SparkSession(sc).builder.getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

# Load the Yelp dataset
df = spark.read.format("csv").option("header", "true").option("multiline", "true").load("yelp_review.csv")

# Convert the 'stars' column to double type and rename it to 'label'
df = df.withColumn("label", when(col("stars") >= 4, 1.0).otherwise(0.0))

# Select the 'text' and 'label' columns
read_data = df.select('text', 'label')

# Remove rows with missing values
df = read_data.dropna(subset=['text', 'label'])

# Split the data into training and testing sets
(trainingData, testData) = df.randomSplit([0.7, 0.3], seed=100)

# Define the ML pipeline
tokenizer = Tokenizer(inputCol="text", outputCol="words")
TF = HashingTF(inputCol=tokenizer.getOutputCol(), outputCol="tfFeatures")
idf = IDF(inputCol="tfFeatures", outputCol="features")
lr = LogisticRegression(maxIter=10, regParam=0.001)

pipeline = Pipeline(stages=[tokenizer, TF, idf, lr])

# Train the model on the training data
model = pipeline.fit(trainingData)

# Evaluate the model on the testing data
predictions = model.transform(testData)

# Calculate accuracy
accuracy = predictions.filter(predictions.label == predictions.prediction).count() / float(testData.count())
print("Accuracy:", accuracy)

# Calculate precision, recall, and F1 score
tp = predictions.filter(predictions.label == 1.0).filter(predictions.prediction == 1.0).count()
fp = predictions.filter(predictions.label == 0.0).filter(predictions.prediction == 1.0).count()
tn = predictions.filter(predictions.label == 0.0).filter(predictions.prediction == 0.0).count()
fn = predictions.filter(predictions.label == 1.0).filter(predictions.prediction == 0.0).count()

precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1_score = 2 * (precision * recall) / (precision + recall)

print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1_score)

# Calculate AUC
evaluator = BinaryClassificationEvaluator(labelCol="label")
auc = evaluator.evaluate(predictions)
print("AUC:", auc)

# Display confusion matrix
conf_matrix = predictions.stat.crosstab("label", "prediction")
conf_matrix.show()

# Define a UDF to extract the probability of the positive sentiment
get_positive_prob = udf(lambda prob: float(prob[1]), DoubleType())

# Define a UDF to extract the probability of the negative sentiment
get_negative_prob = udf(lambda prob: float(prob[0]), DoubleType())

# Load the Yelp dataset
df = spark.read.format("csv").option("header", "true").option("multiline", "true").load("yelp_review.csv")

# Add an index column to the DataFrame
df = df.withColumn("index", monotonically_increasing_id())

# Convert the 'stars' column to double type and rename it to 'label'
df = df.withColumn("label", when(col("stars") >= 4, 1.0).otherwise(0.0))

# Select the 'text' and 'label' columns
read_data = df.select('text', 'label')

# Remove rows with missing values
df = read_data.dropna(subset=['text', 'label'])

# Filter for the reviews with an index greater than 5000 and less than or equal to 5100
new_data = df.filter((col("index") > 5000) & (col("index") <= 5100))

# Use the model to predict the sentiment and probability of the new reviews
predictions = model.transform(new_data)

# Extract the probability of the positive and negative sentiment using the UDFs
predictions = predictions.withColumn("positive_prob", get_positive_prob("probability"))
predictions = predictions.withColumn("negative_prob", get_negative_prob("probability"))

# Display the predicted sentiment, probability of positive sentiment, and probability of negative sentiment for each new review
predictions.select("text", "prediction", "positive_prob", "negative_prob").show(truncate=False)
