import org.apache.log4j.{Logger, Level}
import org.apache.spark.{SparkContext, SparkConf}
import org.apache.spark.mllib.clustering.{KMeans, KMeansModel}
import org.apache.spark.mllib.linalg.Vectors

object kmeans {

  def main(args: Array[String]): Unit = {

    // Code format from TA and Apache Spark k-means https://spark.apache.org/docs/2.2.0/mllib-clustering.html

    //Init spark
    val sparkConf = new SparkConf().setAppName("K-Means").setMaster("local[*]")
    val sc=new SparkContext(sparkConf)

    // Turn off Info Logger for Consolexxx
    Logger.getLogger("org").setLevel(Level.OFF);
    Logger.getLogger("akka").setLevel(Level.OFF);

    // Load and parse the data
    val data = sc.textFile("3D_spatial_network.txt")
    val parsedData = data.map(s => Vectors.dense(s.split(',').map(_.toDouble))).cache()

    //Look at how training data is!
    parsedData.foreach(f=>println(f))

    // Cluster the data into two classes using KMeans
    val numClusters = 4
    val numIterations = 50
    val clusters = KMeans.train(parsedData, numClusters, numIterations)

    // Evaluate clustering by computing Within Set Sum of Squared Errors
    val WSSSE = clusters.computeCost(parsedData)
    println("Within Set Sum of Squared Errors = " + WSSSE)

    //Look at how the clusters are in training data by making predictions
    println("Clustering on training data: ")
    clusters.predict(parsedData).zip(parsedData).foreach(f=>println(f._2,f._1))

    // Save and load model
    clusters.save(sc, "Cluster4")
    val sameModel = KMeansModel.load(sc, "Cluster4")
  }
}
