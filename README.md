# E-commerce Scala Processing Project

## Overview
This repository contains the Scala and Spark code for processing and analyzing [e-commerce customer purchase data](https://www.kaggle.com/datasets/terencicp/e-commerce-dataset-by-olist-as-an-sqlite-database) to build a product recommendation model. The project was developed in a Databricks notebook and is structured to demonstrate how to gather, transform, analyze, and model data effectively using Scala and Apache Spark.

## Project Objective
The main goal of this project is to utilize customer purchase data to generate relevant product recommendations. This involves a series of data transformations and the application of machine learning techniques to predict customer preferences and suggest products that they are likely to purchase.

## Features
- Data Gathering: Loads data directly from various sources.
- Data Transformation: Cleans and structures the data for analysis.
- Analytical Modeling: Uses the ALS (Alternating Least Squares) algorithm in Spark MLlib to build a recommendation model.
- Scalability: Designed to run efficiently on large datasets in a distributed computing environment with Spark.

## Technologies Used
- Scala: The primary programming language for the project.
- Apache Spark: Utilized for handling big data processing and building the machine learning model.
- Databricks: The development environment used for coding and testing.

## Note:
The file `sqlite_conversion.py` is included if you'd like to change the tables within the sqlite file to csv files.