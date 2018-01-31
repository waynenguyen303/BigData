from pyspark import SparkContext
from pyspark import SparkConf

if __name__ == "__main__":

    # init spark program
    conf = SparkConf().setMaster("local")
    sc = SparkContext(conf=conf)

    # make text file into RDD and then map each row by splitting on the tabs
    movie_data = sc.textFile("u.data")
    rows = movie_data.map(lambda line: line.split("\t"))

    # group together all rows by user ID (key) and then make list of all movie they had ranked (value=list)
    idToitem = rows.map(lambda r: (int(r[0]), int(r[1]))).groupByKey()
    print(idToitem.count())
    print(idToitem.map(lambda x: {x[0]: len(x[1])}).collect())

    # lab assignment ask for Id's that have above 25 (at least 26) movie critics
    idRank25 = idToitem.filter(lambda x: len(x[1]) > 25)
    print(idRank25.count())
    print(idRank25.map(lambda x: {x[0]: len(x[1])}).collect())

    # make dictionary of user ID and list of movies, dictionary of user ID and length of list
    # then turn it into RDD to write to file
    id_list = sc.parallelize(idRank25.map(lambda x: {x[0]: list(x[1])}).collect())
    id_len = sc.parallelize(idRank25.map(lambda x: {x[0]: len(x[1])}).collect())

    id_list.saveAsTextFile("IdAndMovieList")
    id_len.saveAsTextFile("IdAndListCount")

    sc.stop()


