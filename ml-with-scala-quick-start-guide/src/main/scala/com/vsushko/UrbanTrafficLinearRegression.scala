package com.vsushko

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.mllib.evaluation.RegressionMetrics
import org.apache.spark.sql._

/**
  *
  * @author vsushko
  */
object UrbanTrafficLinearRegression {

  def main(args: Array[String]): Unit = {
    import org.apache.spark.sql.SparkSession

    // initialize spark session
    val spark = SparkSession
      .builder()
      .master("local[*]")
      .config("spark.sql.warehouse.dir", "/temp")
      .appName(s"OneVsRestExample")
      .getOrCreate()

    Logger.getLogger("org").setLevel(Level.FATAL)
    Logger.getLogger("akka").setLevel(Level.ERROR)

    import org.apache.spark.sql.types._
    import spark.implicits._

    var rawTrafficDF = spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .option("delimiter", ";")
      .format("com.databricks.spark.csv")
      .load("data/UrbanTraffic.csv")
      .cache

    rawTrafficDF.show()
    rawTrafficDF.printSchema()
    rawTrafficDF.describe().show()

    println(rawTrafficDF.count())

    // see a snapshot of the dataset
    rawTrafficDF.select("Hour (Coded)", "Immobilized bus", "Broken Truck",
      "Vehicle excess", "Fire", "Slowness in traffic (%)").show(5)

    var newTrafficDF = rawTrafficDF.withColumnRenamed("Slowness in traffic (%)", "label")
    newTrafficDF = newTrafficDF.withColumn("label", newTrafficDF.col("label").cast(DataTypes.DoubleType))

    // create a temporary view
    newTrafficDF.createOrReplaceTempView("slDF")
    // average the slowness in the form of a percentage
    spark.sql("SELECT avg(label) FROM slDF").show(5)

    // feature engineering and data preparation
    val colNames = newTrafficDF.columns.dropRight(1)

    val assembler = new VectorAssembler()
      .setInputCols(colNames)
      .setOutputCol("features")

    // embed selected columns into a single vector column
    val assembleDF = assembler.transform(newTrafficDF).select("features", "label")
    assembleDF.show(10)

    val seed = 12345L
    // use 60% of the data to train the model and the over 40% to evaluate the model
    val splits = assembleDF.randomSplit(Array(0.60, 0.40), seed)
    val (trainingData, testData) = (splits(0), splits(1))

    // cache in memory for quicker access
    trainingData.cache()
    testData.cache()

    // create an LR estimator
    val lr = new LinearRegression()
      .setFeaturesCol("features")
      .setLabelCol("label")

    // building the Pipeline model for transformations and predictor
    println("Building ML regression model")
    val lrModel = lr.fit(trainingData)

    // save the workflow
    //lrModel.write.overwrite().save("model/LR_model")

    // load the workflow back
    //val sameLRModel = CrossValidatorModel.load("model/GLR_model")

    println("Evaluating the model on the test set and calculating the regression metrics")

    val trainPredictionsAndLabels = lrModel.transform(testData).select("label", "prediction")
      .map { case Row(label: Double, prediction: Double) => (label, prediction) }.rdd


    val testRegressionMetrics = new RegressionMetrics(trainPredictionsAndLabels)

    val results = "\n=====================================================================\n" +
      s"TrainingData count: ${trainingData.count}\n" +
      s"TestData count: ${testData.count}\n" +
      "=====================================================================\n" +
      s"TestData MSE = ${testRegressionMetrics.meanSquaredError}\n" +
      s"TestData RMSE = ${testRegressionMetrics.rootMeanSquaredError}\n" +
      s"TestData R-squared = ${testRegressionMetrics.r2}\n" +
      s"TestData MAE = ${testRegressionMetrics.meanAbsoluteError}\n" +
      s"TestData explained variance = ${testRegressionMetrics.explainedVariance}\n" +
      "=====================================================================\n"
    println(results)
  }
}
