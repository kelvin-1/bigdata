#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
import matplotlib.pyplot as plt
from pyspark.ml.regression import DecisionTreeRegressor, RandomForestRegressor, GBTRegressor
from pyspark.ml.feature import StandardScaler
from pyspark.ml import Pipeline


# In[2]:


# Initialize Spark session
spark = SparkSession.builder.appName("HousePriceAnalysis").getOrCreate()


# In[3]:


# Load dataset
data_path = "train.csv"
df = spark.read.csv(data_path, header=True, inferSchema=True)


# In[4]:


# Handle missing values and data cleaning
threshold = 0.3  # Define the threshold for dropping columns with more than 30% missing values
missing_percentages = [(col_name, df.filter(df[col_name].isNull()).count() / df.count()) for col_name in df.columns]
cols_to_drop = [col_name for col_name, missing_percentage in missing_percentages if missing_percentage > threshold]
df_cleaned = df.drop(*cols_to_drop)
print("Columns dropped due to high missing value percentage:", cols_to_drop)


# In[5]:


for col_name, missing_percentage in missing_percentages:
    if col_name not in cols_to_drop:
        if df_cleaned.schema[col_name].dataType in ('double', 'int'):
            median_value = df_cleaned.approxQuantile(col_name, [0.5], 0.25)[0]
            df_cleaned = df_cleaned.fillna({col_name: median_value})
        else:
            mode_value = df_cleaned.groupBy(col_name).count().orderBy(col("count").desc()).first()[col_name]
            df_cleaned = df_cleaned.fillna({col_name: mode_value})
print("Missing values filled.")


# In[6]:


# Handle any additional data cleaning steps, e.g., outlier removal, data transformation, etc.
# Outlier removal using IQR method
def remove_outliers_iqr(df, col_name):
    Q1 = df.approxQuantile(col_name, [0.25], 0.25)[0]
    Q3 = df.approxQuantile(col_name, [0.75], 0.25)[0]
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df.filter((col(col_name) >= lower_bound) & (col(col_name) <= upper_bound))


# In[7]:


# Apply outlier removal to selected numerical columns
columns_for_outlier_removal = ["GrLivArea", "LotArea", "OverallQual"]
for col_name in columns_for_outlier_removal:
    df_cleaned = remove_outliers_iqr(df_cleaned, col_name)


# In[8]:


# Apply logarithmic transformation to skewed features
from pyspark.sql.functions import log1p
skewed_cols = ["GrLivArea", "LotArea", "SalePrice"]  # Update with your skewed columns
for col_name in skewed_cols:
    df_cleaned = df_cleaned.withColumn(col_name + "_log", log1p(col(col_name)))


# In[9]:


# Handle categorical variables: Convert categorical columns to numerical using StringIndexer and OneHotEncoder
categorical_cols = [col_name for col_name, data_type in df_cleaned.dtypes if data_type == "string"]
indexers = [StringIndexer(inputCol=col_name, outputCol=col_name + "_index").fit(df_cleaned) for col_name in categorical_cols]
df_indexed = df_cleaned
for indexer in indexers:
    df_indexed = indexer.transform(df_indexed)


# In[10]:


# Apply OneHotEncoder after StringIndexer
encoders = [OneHotEncoder(inputCol=indexer.getOutputCol(), outputCol=col_name + "_encoded") for indexer, col_name in zip(indexers, categorical_cols)]
encoder_models = [encoder.fit(df_indexed) for encoder in encoders]
df_encoded = df_indexed
for encoder_model in encoder_models:
    df_encoded = encoder_model.transform(df_encoded)


# In[11]:


# Create a feature vector using VectorAssembler
numeric_cols = [col_name for col_name, data_type in df_encoded.dtypes if data_type in ("int", "double")]
feature_cols = numeric_cols + [col_name + "_encoded" for col_name in categorical_cols] + [col_name + "_log" for col_name in skewed_cols]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
df_assembled = assembler.transform(df_encoded)


# In[12]:


# Split the data into training and testing sets
train_data, test_data = df_assembled.randomSplit([0.8, 0.2], seed=123)


# In[13]:


# Feature scaling using StandardScaler
scaler = StandardScaler(inputCol="features", outputCol="scaled_features", withStd=True, withMean=True)
pipeline = Pipeline(stages=[scaler])
train_data_scaled = pipeline.fit(train_data).transform(train_data)
test_data_scaled = pipeline.fit(test_data).transform(test_data)


# In[14]:


# Train a Linear Regression model
lr = LinearRegression(featuresCol="features", labelCol="SalePrice")
lr_model = lr.fit(train_data)


# In[15]:


# Make predictions on the test data
predictions = lr_model.transform(test_data)


# In[16]:


# Evaluate the model
evaluator = RegressionEvaluator(labelCol="SalePrice", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predictions)
print("Root Mean Squared Error (RMSE):", rmse)


# In[17]:


# Feature importance analysis (coefficients of linear regression)
feature_importances = lr_model.coefficients.toArray()
feature_importance_dict = {feature_cols[i]: feature_importances[i] for i in range(len(feature_cols))}
print("Feature Importances:")
for feature, importance in feature_importance_dict.items():
    print(feature, ":", importance)


# In[18]:


# Retrieve the coefficients and intercept of the Linear Regression model
coefficients = lr_model.coefficients
intercept = lr_model.intercept

print("Coefficients:", coefficients)
print("Intercept:", intercept)


# In[19]:


# Evaluate the model
evaluator = RegressionEvaluator(labelCol="SalePrice", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predictions)
print("Root Mean Squared Error (RMSE):", rmse)

evaluator_mae = RegressionEvaluator(labelCol="SalePrice", predictionCol="prediction", metricName="mae")
mae = evaluator_mae.evaluate(predictions)
print("Mean Absolute Error (MAE):", mae)

evaluator_mse = RegressionEvaluator(labelCol="SalePrice", predictionCol="prediction", metricName="mse")
mse = evaluator_mse.evaluate(predictions)
print("Mean Squared Error (MSE):", mse)

evaluator_r2 = RegressionEvaluator(labelCol="SalePrice", predictionCol="prediction", metricName="r2")
r2 = evaluator_r2.evaluate(predictions)
print("R-squared (R2) Score:", r2)


# In[20]:


# Train a Linear Regression model
lr = LinearRegression(featuresCol="features", labelCol="SalePrice")
lr_model = lr.fit(train_data)


# In[21]:


# Make predictions on the test data
predictions = lr_model.transform(test_data)


# In[22]:


# Display the actual SalePrice and predicted SalePrice
predictions.select("SalePrice", "prediction").show()


# In[23]:


# Evaluate the model predictions
evaluator_rmse = RegressionEvaluator(labelCol="SalePrice", predictionCol="prediction", metricName="rmse")
rmse = evaluator_rmse.evaluate(predictions)
print("Root Mean Squared Error (RMSE):", rmse)

evaluator_mae = RegressionEvaluator(labelCol="SalePrice", predictionCol="prediction", metricName="mae")
mae = evaluator_mae.evaluate(predictions)
print("Mean Absolute Error (MAE):", mae)

evaluator_r2 = RegressionEvaluator(labelCol="SalePrice", predictionCol="prediction", metricName="r2")
r2 = evaluator_r2.evaluate(predictions)
print("R-squared (R2) Score:", r2)


# In[24]:


# Visualize predicted vs. actual SalePrice
actual_prices = predictions.select("SalePrice").collect()
predicted_prices = predictions.select("prediction").collect()

plt.figure(figsize=(10, 6))
plt.scatter(actual_prices, predicted_prices, color='blue')
plt.plot(actual_prices, actual_prices, color='red')
plt.xlabel("Actual SalePrice")
plt.ylabel("Predicted SalePrice")
plt.title("Actual vs. Predicted SalePrice")
plt.show()


# In[25]:


# Define a list of regression models to compare
models = [
    LinearRegression(featuresCol="scaled_features", labelCol="SalePrice"),
    DecisionTreeRegressor(featuresCol="scaled_features", labelCol="SalePrice"),
    RandomForestRegressor(featuresCol="scaled_features", labelCol="SalePrice"),
    GBTRegressor(featuresCol="scaled_features", labelCol="SalePrice")
]


# In[26]:


# Evaluate and compare the performance of each model
for model in models:
    model_name = model.__class__.__name__
    print("Training", model_name)


# In[27]:


# Train the model
model_fit = model.fit(train_data_scaled)


# In[28]:


# Make predictions on the test data
predictions = model_fit.transform(test_data_scaled)


# In[29]:


# Evaluate the model
evaluator = RegressionEvaluator(labelCol="SalePrice", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predictions)
print(model_name, "Root Mean Squared Error (RMSE):", rmse)


# In[30]:


# ...

# Evaluate and compare the performance of each model
for model in models:
    model_name = model.__class__.__name__
    print("Training", model_name)
    
    # Train the model
    model_fit = model.fit(train_data_scaled)
    
    # Make predictions on the test data
    predictions = model_fit.transform(test_data_scaled)
    
    # Evaluate the model
    evaluator = RegressionEvaluator(labelCol="SalePrice", predictionCol="prediction", metricName="rmse")
    rmse = evaluator.evaluate(predictions)
    print(model_name, "Root Mean Squared Error (RMSE):", rmse)


# In[31]:


# Train a Decision Tree Regression model with increased maxBins
dt = DecisionTreeRegressor(featuresCol="features", labelCol="SalePrice", maxBins=500)  # Adjust the value as needed
dt_model = dt.fit(train_data)

# Make predictions on the test data
dt_predictions = dt_model.transform(test_data)

# Display the actual SalePrice and predicted SalePrice
dt_predictions.select("SalePrice", "prediction").show()


# In[32]:


# Train a Random Forest Regression model with increased maxBins
rf = RandomForestRegressor(featuresCol="features", labelCol="SalePrice", maxBins=500)  # Adjust the value as needed
rf_model = rf.fit(train_data)

# Make predictions on the test data
rf_predictions = rf_model.transform(test_data)

# Display the actual SalePrice and predicted SalePrice
rf_predictions.select("SalePrice", "prediction").show()


# In[33]:


# Train a Gradient Boosting Regression model with increased maxBins
gbt = GBTRegressor(featuresCol="features", labelCol="SalePrice", maxBins=500)  # Adjust the value as needed
gbt_model = gbt.fit(train_data)

# Make predictions on the test data
gbt_predictions = gbt_model.transform(test_data)

# Display the actual SalePrice and predicted SalePrice
gbt_predictions.select("SalePrice", "prediction").show()


# In[34]:


import matplotlib.pyplot as plt

# ...

# Lists to store model names and RMSE values
model_names = []
rmse_values = []

# Evaluate and compare the performance of each model
for model in models:
    model_name = model.__class__.__name__
    print("Training", model_name)
    model_names.append(model_name)
    
    # Train the model
    model_fit = model.fit(train_data_scaled)
    
    # Make predictions on the test data
    predictions = model_fit.transform(test_data_scaled)
    
    # Evaluate the model
    evaluator = RegressionEvaluator(labelCol="SalePrice", predictionCol="prediction", metricName="rmse")
    rmse = evaluator.evaluate(predictions)
    rmse_values.append(rmse)
    print(model_name, "Root Mean Squared Error (RMSE):", rmse)

# Plotting the RMSE values
plt.figure(figsize=(10, 6))
plt.bar(model_names, rmse_values, color='blue')
plt.xlabel('Model')
plt.ylabel('RMSE')
plt.title('RMSE Comparison of Regression Models')
plt.xticks(rotation=45)
plt.tight_layout()

# Save the plot (optional)
plt.savefig('rmse_comparison.png')

# Show the plot
plt.show()

# Close Spark session
spark.stop()


# In[ ]:




