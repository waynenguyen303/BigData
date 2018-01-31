import org.apache.spark.{SparkConf, SparkContext}

object movieRank {

  def main(args: Array[String]): Unit = {

    // spark init
    val conf = new SparkConf().setAppName("Movie Ranker").setMaster("local")
    val sc = new SparkContext(conf)

    // load movie data and group movie list by user ID
    val input = sc.textFile("u.data")
    val rows = input.map(line => line.split("\t",4))
      .map(id =>(id(0), id(1)))
      .groupByKey()

    // filter by movie list that is above 25 (at least 26), replace movie list by its length
    val filter25 = rows.filter( _._2.size > 25)
    val filter25_size = filter25.map(id => (id._1, id._2.size))

    // print to console the number of user IDs before and after filter
    System.out.println(rows.count())
    System.out.println(filter25.count())

    // save to file the user id and movie list, user id and movie list length
    filter25.saveAsTextFile("UserId_And_movie_list_output")
    filter25_size.saveAsTextFile("UserId_And_movie_List_length_output")
    System.out.println("Scala Spark Works")
    sc.stop()
  }
}
