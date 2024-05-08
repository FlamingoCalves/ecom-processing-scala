// Databricks notebook source
// Scala: List files in the FileStore
dbutils.fs.ls("/FileStore/tables/").foreach(println)

// COMMAND ----------

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.recommendation.ALS
import org.apache.spark.sql.functions._

object ProductRecommendationSystem {
  // Initialize Spark session
  val spark: SparkSession = SparkSession.builder()
    .appName("Product Recommendation System")
    .getOrCreate()

  import spark.implicits._

  // Method to load data
  def loadData(filePath: String): DataFrame = {
    spark.read.format("csv").option("header", "true").load(filePath)
  }

  // Method to perform indexing and return mapping DataFrame
  def indexColumn(df: DataFrame, inputCol: String, outputCol: String): (DataFrame, DataFrame) = {
    val indexer = new StringIndexer()
      .setInputCol(inputCol)
      .setOutputCol(outputCol)
      .fit(df)
    val indexed = indexer.transform(df)
    (indexed, indexed.select(outputCol, inputCol).distinct())
  }

  // Main method to run the recommendation system
  def run(): Unit = {
    // Load necessary tables
    val orderItems = loadData("/FileStore/tables/order_items.csv")
    val orders = loadData("/FileStore/tables/orders.csv")
    val orderReviews = loadData("/FileStore/tables/order_reviews.csv")
    val products = loadData("/FileStore/tables/products.csv")

    // Process order items
    val (indexedItems, productMapping) = indexColumn(orderItems, "product_id", "product_id_index")

    // First, join orders and orderItems on 'order_id'
    val firstMerge = orders.join(indexedItems, "order_id")
      .select("order_id", "customer_id", "product_id_index")  // Use indexed product_id

    // Join the result with orderReviews to include review scores
    val data = firstMerge.join(orderReviews, "order_id")
      .select("customer_id", "product_id_index", "review_score")

    // Convert customer_id to numeric indices
    val customerIndexer = new StringIndexer()
      .setInputCol("customer_id")
      .setOutputCol("customer_id_index")
      .fit(data)
    val finalData = customerIndexer.transform(data)

    // Rename review_score to rating and cast it to float
    val trainingData = finalData.withColumn("rating", col("review_score").cast("float"))

    // Train ALS model
    val als = new ALS()
      .setMaxIter(5)
      .setRegParam(0.01)
      .setUserCol("customer_id_index")
      .setItemCol("product_id_index")
      .setRatingCol("rating")

    val model = als.fit(trainingData)

    // Select necessary columns from products data
    val productDetails = products.select("product_id", "product_category_name")

    // Generate recommendations for all users
    val recommendations = model.recommendForAllUsers(10)
    val explodedRecommendations = recommendations.selectExpr("customer_id_index", "explode(recommendations) as recommendation")
      .select("customer_id_index", "recommendation.product_id_index", "recommendation.rating")

    val recommendationsWithProductIds = explodedRecommendations
      .join(productMapping, explodedRecommendations("product_id_index") === productMapping("product_id_index"))
      .select("customer_id_index", "product_id", "rating")

    // Join with product details to get readable product names and categories
    val recommendationDetails = recommendationsWithProductIds.join(productDetails, "product_id")
      .withColumn("type", lit("Recommendation"))
      .select("customer_id_index", "product_id", "product_category_name", "rating", "type")

    // Assuming 'finalData' is the DataFrame used for ALS training
    val initialPurchases = finalData.select("customer_id_index", "product_id_index").distinct()

    // Map back to original product IDs using the mapping DataFrame
    val initialPurchaseDetails = initialPurchases.join(productMapping, "product_id_index")
      .join(productDetails, "product_id")
      .select("customer_id_index", "product_id", "product_category_name")
      .groupBy("customer_id_index")
      .agg(collect_list("product_category_name").alias("purchased_products"))

    // Combine initial purchases with recommendations for each customer
    val fullRecommendationDetails = initialPurchaseDetails.join(recommendationDetails, "customer_id_index")
      .select("customer_id_index", "purchased_products", "product_category_name", "rating", "type")

    fullRecommendationDetails.show(false)
  }

  def main(args: Array[String]): Unit = {
    run()
  }
}

ProductRecommendationSystem.run()

// COMMAND ----------

import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.sql.functions.col
import org.apache.spark.ml.recommendation.ALS
import org.apache.spark.sql.functions._
import org.apache.spark.sql.DataFrame

val spark = SparkSession.builder().appName("Processing Application").getOrCreate()

// Load necessary tables
val orderItems = spark.read.format("csv").option("header", "true").load("/FileStore/tables/order_items.csv")
val orders = spark.read.format("csv").option("header", "true").load("/FileStore/tables/orders.csv")
val orderReviews = spark.read.format("csv").option("header", "true").load("/FileStore/tables/order_reviews.csv")
val products = spark.read.format("csv").option("header", "true").load("/FileStore/tables/products.csv")

// Indexing product_id
val productIndexer = new StringIndexer()
  .setInputCol("product_id")
  .setOutputCol("product_id_index")
  .fit(orderItems)

val indexedItems = productIndexer.transform(orderItems)

// Creating mapping from index back to original product IDs
val indexToProductIdMapping = indexedItems.select("product_id_index", "product_id").distinct()

// First, join orders and orderItems on 'order_id'
val firstMerge = orders.join(indexedItems, "order_id")
  .select("order_id", "customer_id", "product_id_index")  // Use indexed product_id

// Join the result with orderReviews to include review scores
val data = firstMerge.join(orderReviews, "order_id")
  .select("customer_id", "product_id_index", "review_score")

// Convert customer_id to numeric indices
val customerIndexer = new StringIndexer()
  .setInputCol("customer_id")
  .setOutputCol("customer_id_index")
  .fit(data)
val finalData = customerIndexer.transform(data)

// Rename review_score to rating and cast it to float
val trainingData = finalData.withColumn("rating", col("review_score").cast("float"))

// ALS Model
val als = new ALS()
  .setMaxIter(5)
  .setRegParam(0.01)
  .setUserCol("customer_id_index")
  .setItemCol("product_id_index")
  .setRatingCol("rating")

val model = als.fit(trainingData)

// Select necessary columns from products data
val productDetails = products.select("product_id", "product_category_name")

// Generate recommendations for all users
val recommendations = model.recommendForAllUsers(10)
val explodedRecommendations = recommendations.selectExpr("customer_id_index", "explode(recommendations) as recommendation")
  .select("customer_id_index", "recommendation.product_id_index", "recommendation.rating")

val recommendationsWithProductIds = explodedRecommendations
  .join(indexToProductIdMapping, explodedRecommendations("product_id_index") === indexToProductIdMapping("product_id_index"))
  .select("customer_id_index", "product_id", "rating")


// Join with product details to get readable product names and categories
val recommendationDetails = recommendationsWithProductIds.join(productDetails, "product_id")
  .withColumn("type", lit("Recommendation"))
  .select("customer_id_index", "product_id", "product_category_name", "rating", "type")


// Assuming 'finalData' is the DataFrame used for ALS training
val initialPurchases = finalData.select("customer_id_index", "product_id_index").distinct()

// Map back to original product IDs using the mapping DataFrame
val initialPurchaseDetails = initialPurchases.join(indexToProductIdMapping, "product_id_index")
  .join(productDetails, "product_id")
  .select("customer_id_index", "product_id", "product_category_name")
  .groupBy("customer_id_index")
  .agg(collect_list("product_category_name").alias("purchased_products"))

// Combine initial purchases with recommendations for each customer
val fullRecommendationDetails = initialPurchaseDetails.join(recommendationDetails, "customer_id_index")
  .select("customer_id_index", "purchased_products", "product_category_name", "rating", "type")

fullRecommendationDetails.show(false)

// COMMAND ----------

import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.sql.functions.col
import org.apache.spark.ml.recommendation.ALS
import org.apache.spark.sql.functions._
import org.apache.spark.sql.DataFrame

val spark = SparkSession.builder().appName("Processing Application").getOrCreate()

// Load necessary tables
val orderItems = spark.read.format("csv").option("header", "true").load("/FileStore/tables/order_items.csv")
val orders = spark.read.format("csv").option("header", "true").load("/FileStore/tables/orders.csv")
val orderReviews = spark.read.format("csv").option("header", "true").load("/FileStore/tables/order_reviews.csv")
val products = spark.read.format("csv").option("header", "true").load("/FileStore/tables/products.csv")

// COMMAND ----------

// Indexing product_id
val productIndexer = new StringIndexer()
  .setInputCol("product_id")
  .setOutputCol("product_id_index")
  .fit(orderItems)

val indexedItems = productIndexer.transform(orderItems)

// Creating mapping from index back to original product IDs
val indexToProductIdMapping = indexedItems.select("product_id_index", "product_id").distinct()

// First, join orders and orderItems on 'order_id'
val firstMerge = orders.join(indexedItems, "order_id")
  .select("order_id", "customer_id", "product_id_index")  // Use indexed product_id

// Join the result with orderReviews to include review scores
val data = firstMerge.join(orderReviews, "order_id")
  .select("customer_id", "product_id_index", "review_score")

// Convert customer_id to numeric indices
val customerIndexer = new StringIndexer()
  .setInputCol("customer_id")
  .setOutputCol("customer_id_index")
  .fit(data)
val finalData = customerIndexer.transform(data)

// Rename review_score to rating and cast it to float
val trainingData = finalData.withColumn("rating", col("review_score").cast("float"))

// ALS Model
val als = new ALS()
  .setMaxIter(5)
  .setRegParam(0.01)
  .setUserCol("customer_id_index")
  .setItemCol("product_id_index")
  .setRatingCol("rating")

val model = als.fit(trainingData)

// COMMAND ----------

// Select necessary columns from products data
val productDetails = products.select("product_id", "product_category_name")

// Generate recommendations for all users
val recommendations = model.recommendForAllUsers(10)
val explodedRecommendations = recommendations.selectExpr("customer_id_index", "explode(recommendations) as recommendation")
  .select("customer_id_index", "recommendation.product_id_index", "recommendation.rating")

val recommendationsWithProductIds = explodedRecommendations
  .join(indexToProductIdMapping, explodedRecommendations("product_id_index") === indexToProductIdMapping("product_id_index"))
  .select("customer_id_index", "product_id", "rating")


// Join with product details to get readable product names and categories
val recommendationDetails = recommendationsWithProductIds.join(productDetails, "product_id")
  .withColumn("type", lit("Recommendation"))
  .select("customer_id_index", "product_id", "product_category_name", "rating", "type")


// Assuming 'finalData' is the DataFrame used for ALS training
val initialPurchases = finalData.select("customer_id_index", "product_id_index").distinct()

// Map back to original product IDs using the mapping DataFrame
val initialPurchaseDetails = initialPurchases.join(indexToProductIdMapping, "product_id_index")
  .join(productDetails, "product_id")
  .select("customer_id_index", "product_id", "product_category_name")
  .groupBy("customer_id_index")
  .agg(collect_list("product_category_name").alias("purchased_products"))

// Combine initial purchases with recommendations for each customer
val fullRecommendationDetails = initialPurchaseDetails.join(recommendationDetails, "customer_id_index")
  .select("customer_id_index", "purchased_products", "product_category_name", "rating", "type")

fullRecommendationDetails.show(false)
