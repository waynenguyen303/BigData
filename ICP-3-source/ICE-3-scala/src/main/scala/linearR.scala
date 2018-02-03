import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.regression.LinearRegressionModel
import org.apache.spark.mllib.regression.LinearRegressionWithSGD
import org.apache.spark.mllib.regression.LinearRegressionWithSGD._

object linearR {

  def main(args: Array[String]): Unit = {

    // spark init
    val conf = new SparkConf().setAppName("Movie Ranker").setMaster("local")
    val sc = new SparkContext(conf)

    // load movie data and group movie list by user ID
    val input = sc.textFile("lpsa.data")

    input.take(5).foreach(f=>println(f))
    val parsedData = input.map { line =>
      val parts = line.split(',')
      LabeledPoint(parts(0).toDouble, Vectors.dense(parts(1).split(' ').map(_.toDouble)))
    }.cache()

    val Array(training, test) = parsedData.randomSplit(Array(0.7, 0.3))

    parsedData.take(1).foreach(f=>println(f))
    // Building the model
    val numIterations = 50
    val stepSize = 0.00000001
    val model = LinearRegressionWithSGD.train(training, numIterations, stepSize)

    // Evaluate model on training examples and compute training error
    val valuesAndPreds = training.map { point =>
      val prediction = model.predict(point.features)
      (point.label, prediction)
    }
    val MSE = valuesAndPreds.map{ case(v, p) => math.pow((v - p), 2) }.mean()
    println("training Mean Squared Error = " + MSE)

    // Evaluate model on test examples and compute training error
    val valuesAndPreds2 = test.map { point =>
      val prediction = model.predict(point.features)
      (point.label, prediction)
    }
    val MSE2 = valuesAndPreds2.map{ case(v, p) => math.pow((v - p), 2) }.mean()
    println("test Mean Squared Error = " + MSE2)
    valuesAndPreds.take(5).foreach(f=>println(f))
    sc.stop()
  }
}
