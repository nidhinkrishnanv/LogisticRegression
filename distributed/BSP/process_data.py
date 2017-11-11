from pyspark.sql import SparkSession
from pyspark import SparkContext
from options import Options
import re
from pyspark.ml.feature import CountVectorizer

stopWords = set(['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven', 'isn', 'ma', 'mightn', 'mustn', 'needn', 'shan', 'shouldn', 'wasn', 'weren', 'won', 'wouldn'])
myStopWords = set(['(', ')','', ' ', '``', '@', '.', 'en', ',', '–', '[', ']'])
stopWords.update(myStopWords)

opt = Options()

paths = {"verysmall":"hdfs:///user/ds222/assignment-1/DBPedia.verysmall/",
        "full":"hdfs:///user/ds222/assignment-1/DBPedia.full/"}

dset_types = ['train', 'test', 'devel']

input_files = [paths[opt.DATA_SIZE] + opt.DATA_SIZE + "_" + dset_type for dset_type in dset_types]

def get_input_file(dset_type):
    return paths[opt.DATA_SIZE] + opt.DATA_SIZE + "_" + dset_type + ".txt"


def label_data_point(line):
    sents = line.split("\t")

    labels = sents[0].split(",")
    labels[-1] = labels[-1].strip()
    sentences = sents[1].split(" ", 2)

    # Split data based on space and remove stop words
    tokens = []
    sentence = sentences[2].split()
    sentence = re.split("\W+", sentences[2])

    tokens = [token.lower() for token in sentence if token not in stopWords]

    return (labels, tokens)

def filter_words(wd_count_pair):
    if wd_count_pair[1] > opt.min_word_count:
        return True
    else:
        return False



train_file = get_input_file('train')
sc = SparkContext("local", "BSP")
train_data = sc.textFile(train_file)

# Split data into labels and list of tokens
label_tokens = train_data.map(label_data_point)
label_tokens.saveAsTextFile("hdfs:///user/nidhinkrishnanv/label_tokensed")
print(hasattr(label_tokens, "toDF"))

# # Create Spark session
# spark = SparkSession(sc)
# print(hasattr(label_tokens, "toDF"))

# # Convert list of label and 'list of tokens' into dataframe
# dataFram = label_tokens.toDF(['labels', 'tokens'])
# dataFram.rdd.saveAsTextFile("hdfs:///user/nidhinkrishnanv/dataFram")


# # Count words in list of words in dataframe
# cv = CountVectorizer(inputCol="tokens", outputCol="features", minDF=4.0)
# model = cv.fit(dataFram)
# result = model.transform(dataFram)
# # result.show(truncate=False)
# result.rdd.saveAsTextFile("hdfs:///user/nidhinkrishnanv/result")

token_count = label_tokens.flatMap(lambda line:line[1]).map(lambda word:(word, 1)).reduceByKey(lambda a, b: a+b)
vocab = token_count.filter(filter_words)

labels = label_tokens.flatMap(lambda line:line[0]).distint()


