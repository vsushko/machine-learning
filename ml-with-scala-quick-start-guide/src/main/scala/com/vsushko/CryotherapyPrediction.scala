package com.vsushko


import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.VectorAssembler

/**
  *
  * @author vsushko
  */
object CryotherapyPrediction {

  def main(args: Array[String]): Unit = {
    import org.apache.spark.sql.SparkSession

    // initialize spark session
    val sparkSession = SparkSession
      .builder()
      .master("local[*]")
      .config("spark.sql.warehouse.dir", "/temp")
      .appName("CrypotherapyPrediction")
      .getOrCreate()

    // reading the training dataset
    var CryotherapyDF = sparkSession.read
      .option("header", "true")
      .option("inferSchema", "true")
      .format("com.databricks.spark.csv")
      .load("data/Cryotherapy.csv")
      .cache()

    CryotherapyDF.printSchema()

    CryotherapyDF.createOrReplaceTempView("myTempDataFrame")

    // preprocessing and feature engineering
    CryotherapyDF = CryotherapyDF.withColumnRenamed("Result_of_treatment", "label")
    CryotherapyDF.printSchema()

    val selectedCols = Array("sex", "age", "Time", "Number_of_Warts", "Type", "Area")

    val verctorAssembler = new VectorAssembler()
      .setInputCols(selectedCols)
      .setOutputCol("features")

    val numericDF = verctorAssembler.transform(CryotherapyDF)
      .select("label", "features")

    numericDF.show()

    // preparing training data and training a classifier
    val splits = numericDF.randomSplit(Array(0.8, 0.2))
    val trainDF = splits(0)
    val testDF = splits(1)

    // instantiate a decision tree classifier
    val dt = new DecisionTreeClassifier()
      .setImpurity("gini")
      .setMaxBins(10)
      .setMaxDepth(30)
      .setLabelCol("label")
      .setFeaturesCol("features")

    // preform the training
    val dtModel = dt.fit(trainDF)

    // evaluating the model
    val evaluator = new BinaryClassificationEvaluator().setLabelCol("label")

    // evaluate the trained model on the test set
    val predictionDF = dtModel.transform(testDF)

    // compute the classification accuracy:
    val accuracy = evaluator.evaluate(predictionDF)
    println("Accuracy = " + accuracy)

    sparkSession.stop()
  }

}