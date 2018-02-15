from pyspark import SparkContext
from pyspark import SparkConf

import json

if __name__ == "__main__":

    # init spark program
    conf = SparkConf().setMaster("local")
    sc = SparkContext(conf=conf)

    # make text file into RDD and then map each row by splitting on the tabs
    movie_data = sc.textFile("vids/annotationOutput.txt")
    rows = movie_data.map(lambda line: line.split(" - ")).filter(lambda x: x[0][0] != "*")

    # word count on the rdd
    counts = rows.map(lambda x: x[0], 1)
    kv = counts.map(lambda y: (y, 1)).reduceByKey(lambda a, b: a+b).sortBy(lambda x: x[1], ascending=False)

    # word senement analysis
    sentiment = rows.map(lambda x: (x[0], float(x[1])))
    senti = sentiment.reduceByKey(lambda a,b: a+b)

    kv_sorted = kv.sortBy(lambda x:x[0]).collect()
    senti_sorted = senti.sortBy(lambda x:x[0]).collect()

    most_sentiment={}
    for i in range(len(kv_sorted)):
        most_sentiment[kv_sorted[i]] = senti_sorted[i][1] / kv_sorted[i][1]
    sorted_ms = sorted(most_sentiment.items(), key= lambda x:x[1], reverse=True)

    #write to file, repeat for each mainframe output
    fp = open('vid-results/annotationResult.txt','w')
    fp.write("------((Feature,Count): Accuracy), Ranked by Highest Accuracy Average on Mainframe Images -----\n")
    for item in sorted_ms:
        fp.write(str(item)+"\n")
    fp.write("\n------Ranked word count of Features on Mainframes Images -----\n")
    for item in kv.collect():
        fp.write(str(item)+"\n")

    fp.close()
    sc.stop()


