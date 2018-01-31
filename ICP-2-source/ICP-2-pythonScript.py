from pyspark import SparkContext
from pyspark import SparkConf

if __name__ == "__main__":

    conf = SparkConf().setMaster("local")
    sc = SparkContext(conf=conf)

    words = sc.textFile("input.txt").flatMap(lambda x: x.split(" "))
    group_words = words.groupBy(lambda x: x[0])

    print([(w[0], [i for i in w[1]]) for w in group_words.collect()])

    output = [(w[0], [i for i in w[1]]) for w in group_words.collect()]

    ic = sc.parallelize(output)
    ic.saveAsTextFile("output2")

    sc.stop()

