import org.apache.spark.{SparkConf, SparkContext}


object wordcount {

  def main(args: Array[String]): Unit ={


    val conf = new SparkConf().setAppName("word count").setMaster("local")
    val sc = new SparkContext(conf)
    val input = sc.textFile("Words.txt")
    val count = input.flatMap(line => line.split(" "))
        .map(word => (word,1))
        .reduceByKey(_+_)
    count.saveAsTextFile("output")
    System.out.println("Scala Spark Works")

  }

}
