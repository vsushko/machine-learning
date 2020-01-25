package com.vsushko

import org.apache.spark.ml.feature.VectorAssembler

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
      .appName("CrypotherapyPrediction")
      .getOrCreate()

    val rawTrafficDF = spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .option("delimiter", ";")
      .format("com.databricks.spark.csv")
      .load("data/UrbanTraffic.csv")
      .cache

    rawTrafficDF.printSchema()

    println(rawTrafficDF.count())

    // see a snapshot of the dataset
    rawTrafficDF.select("Hour (Coded)", "Immobilized bus", "Broken Truck",
      "Vehicle excess", "Fire", "Slowness in traffic (%)").show(5)

    var newTrafficDF = rawTrafficDF.withColumnRenamed("Slowness in traffic (%)", "label")

    // create a temporary view
    newTrafficDF.createOrReplaceTempView("slDF")

    // average the slowness in the form of a percentage
    spark.sql("SELECT avg(label) as avgSlowness FROM slDF").show()

    newTrafficDF = newTrafficDF.withColumnRenamed("Point of flooding", "NoOfFloodPoint")
    spark.sql("SELECT max('Point of flooding') FROM slDF").show()


    rawTrafficDF.select("Hour (Coded)", "Immobilized bus", "Broken Truck",
      "Point of flooding", "Fire", "Slowness in traffic (%)")
      .describe().show()

    // feature engineering and data preparation
    val colNames = newTrafficDF.columns.dropRight(1)

    val assembler = new VectorAssembler()
      .setInputCols(colNames)
      .setOutputCol("features")

    // embed selected columns into a single vector column
    val assembleDF = assembler.transform(newTrafficDF).select("features", "label")
    assembleDF.show()
  }
}
