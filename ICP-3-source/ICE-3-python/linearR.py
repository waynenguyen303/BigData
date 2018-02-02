from pyspark import SparkContext
from pyspark import SparkConf
from pyspark.mllib.regression import LabeledPoint, LinearRegressionWithSGD


if __name__ == "__main__":

    # init spark program
    conf = SparkConf().setMaster("local")
    sc = SparkContext(conf=conf)

    # Load and parse the data
    def parse_point(line):
        values = [float(x) for x in line.replace(',', ' ').split(' ')]
        return LabeledPoint(values[0], values[1:])

    # Make RDD and parse data
    spatial_data = sc.textFile("lpsa.data")
    parsed_data = spatial_data.map(parse_point)

    print(parsed_data.collect())

    # Split data in 70% training and 30% test
    (train_data, test_data) = parsed_data.randomSplit([0.7, 0.3])

    # make linear regression model
    lr_model = LinearRegressionWithSGD.train(train_data, iterations=50, step=.000001)

    # use the predict function and then calculate the MSE for the training data
    vp = train_data.map(lambda x: (x.label, lr_model.predict(x.features)))
    mse = vp.map(lambda r: (r[0] - r[1])**2).reduce(lambda x, y: x + y) / vp.count()
    print(vp.collect())
    print("Training MSE = " + str(mse))

    # use the predict function and then calculate the MSE for the test data
    vp1 = test_data.map(lambda x: (x.label, lr_model.predict(x.features)))
    mse1 = vp1.map(lambda r: (r[0] - r[1]) ** 2).reduce(lambda x, y: x + y) / vp1.count()
    print("Test MSE = " + str(mse1))

    sc.stop()
