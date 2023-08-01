from sentence_transformers import SentenceTransformer, InputExample, losses, util
from torch.utils.data import DataLoader
from pyspark.sql import SparkSession
from pyspark.sql.types import StringType, FloatType, StructType, StructField

# Initialize Spark session
spark = SparkSession.builder.appName("CreateDataFrameExample").getOrCreate()

# Sample training data
data = [("My first sentence", "My second sentence", 0.8),
        ("Another pair", "Unrelated sentence", 0.3)]

# Define schema for the DataFrame
schema = StructType([
    StructField("sentence_1", StringType(), True),
    StructField("sentence_2", StringType(), True),
    StructField("label", FloatType(), True)
])

# Create DataFrame from the sample data and schema
df = spark.createDataFrame(data, schema)

# Show the DataFrame
df.show()

#Define the model. Either from scratch of by loading a pre-trained model
model = SentenceTransformer('distilbert-base-nli-mean-tokens')

#Define your train examples. You need more than just two examples...
# train_examples = [InputExample(texts=['My first sentence', 'My second sentence'], label=0.8),
#     InputExample(texts=['Another pair', 'Unrelated sentence'], label=0.3)]

# Assuming you have a PySpark DataFrame with columns 'sentence_1' and 'sentence_2'
# Convert the DataFrame to a list of InputExample objects
# train_examples = df.rdd.map(lambda row: InputExample(texts=[row['sentence_1'], row['sentence_2']], label=row['label'])).collect()
train_data_list = df.rdd.map(lambda row: [row['sentence_1'], row['sentence_2'], row['label']]).collect()
print(train_data_list)

train_examples = [InputExample(texts=[i[0], i[1]], label=i[2]) for i in train_data_list]

#Define your train dataset, the dataloader and the train loss
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
train_loss = losses.CosineSimilarityLoss(model)

#Tune the model
model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=1, warmup_steps=100)

#Sentences are encoded by calling model.encode()
emb1 = model.encode("This is a red cat with a hat.")
emb2 = model.encode("Have you seen my red cat?")

cos_sim = util.cos_sim(emb1, emb2)
print("Cosine-Similarity:", cos_sim)