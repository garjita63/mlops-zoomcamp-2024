# Homework

The goal of this homework is to train a simple model for predicting the duration of a ride, but use Mage for it.

We'll use [the same NYC taxi dataset ](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page), the Yellow taxi data for 2023.


## Question 1. Run Mage

First, let's run Mage with Docker Compose. Follow the quick start guideline.

What's the version of Mage we run?

(You can see it in the UI)

### Answer of Question 1: v0.9.70


## Question 2. Creating a project

Now let's create a new project. We can call it "homework_03", for example.

How many lines are in the created metadata.yaml file?
- 35
- 45
- 55
- 65

**Solution**

  ```
  docker exec -it mlops-magic-platform-1 bash
  root@4c0edc9c9a86:/home/src# mage init homework_03
  root@4c0edc9c9a86:/home/src# cd homework_03
  root@4c0edc9c9a86:/home/src/mlops/homework_03# wc -l metadata.yaml
  55 metadata.yaml
  ```
  
### Answer of Question 2: 55


## Question 3. Creating a pipeline

Let's create an ingestion code block.

In this block, we will read the March 2023 Yellow taxi trips data.

How many records did we load?
- 3,003,766
- 3,203,766
- 3,403,766
- 3,603,766

**Solution**

  ```
    import requests
    from io import BytesIO
    from typing import List
    
    import pandas as pd
    import numpy as np
    
    if 'data_loader' not in globals():
        from mage_ai.data_preparation.decorators import data_loader
    
    
    @data_loader
    def ingest_files(**kwargs) -> pd.DataFrame:
        dfs: List[pd.DataFrame] = []

    #for year, months in [(2023, (3))]:
    #    for i in range(*months):
    response = requests.get(
        'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-03.parquet'
    )

    df = pd.read_parquet(BytesIO(response.content))
    df['tpep_pickup_datetime_cleaned'] = df['tpep_pickup_datetime'].astype(np.int64) // 10**9
    dfs.append(df)

    return pd.concat(dfs)

  ```

   ![image](https://github.com/garjita63/mlops-zoomcamp-2024/assets/77673886/b273cf6e-4588-40af-96f4-ae467f11c6fb)

    
### Answer of Question 3: 3,403,766


## Question 4. Data preparation

Let's use the same logic for preparing the data we used previously. We will need to create a transformer code block and put this code there.

This is what we used (adjusted for yellow dataset):
   
    def ead_dataframe(filename):
        df = pd.read_parquet(filename)
        df.tpep_dropoff_datetime = pd.to_datetime(df.tpep_dropoff_datetime)
        df.tpep_pickup_datetime = pd.to_datetime(df.tpep_pickup_datetime)
    
        df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
        df.duration = df.duration.dt.total_seconds() / 60
    
        df = df[(df.duration >= 1) & (df.duration <= 60)]
    
        categorical = ['PULocationID', 'DOLocationID']
        df[categorical] = df[categorical].astype(str)
        
        return df

Let's adjust it and apply to the data we loaded in question 3.

What's the size of the result?
- 2,903,766
- 3,103,766
- 3,316,216
- 3,503,766

**Solution**

    ```
    from typing import Tuple

    import pandas as pd
    
    from mlops.utils.data_preparation.yellow_cleaning import clean
    from mlops.utils.data_preparation.feature_engineering import combine_features
    from mlops.utils.data_preparation.feature_selector import select_features
    from mlops.utils.data_preparation.splitters import split_on_value
    
    if 'transformer' not in globals():
        from mage_ai.data_preparation.decorators import transformer
    
    @transformer
    def ead_dataframe(filename):
        df = pd.read_parquet('/home/src/yellow_tripdata_2023-03.parquet')
        df.tpep_dropoff_datetime = pd.to_datetime(df.tpep_dropoff_datetime)
        df.tpep_pickup_datetime = pd.to_datetime(df.tpep_pickup_datetime)

    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df.duration = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)
    
    return df
    ```

    ![image](https://github.com/garjita63/mlops-zoomcamp-2024/assets/77673886/02173880-01da-43b7-af77-49b142e9298a)

### Answer of Question 4: 3,316,216


## Question 5. Train a model

We will now train a linear regression model using the same code as in homework 1

    - Fit a dict vectorizer
    - Train a linear regression with default parameres
    - Use pick up and drop off locations separately, don't create a combination feature

Let's now use it in the pipeline. We will need to create another transformation block, and return both the dict vectorizer and the model

What's the intercept of the model?

Hint: print the intercept_ field in the code block
- 21.77
- 24.77
- 27.77
- 31.77

**Solution**

    ```
    from typing import Tuple
    import pandas as pd
    import numpy as np
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.linear_model import LinearRegression
    
    if 'transformer' not in globals():
        from mage_ai.data_preparation.decorators import transformer
    
    @transformer
    def transform(
        df: pd.DataFrame, **kwargs
    ) -> Tuple[DictVectorizer, LinearRegression]:
        print("Starting the transform function")

    # Compute the duration in minutes
    df['duration'] = (df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']).dt.total_seconds() / 60
    print("Duration computed")

    # Filter the records to keep only those with duration between 1 and 60 minutes (inclusive)
    df_filtered = df[(df['duration'] >= 1) & (df['duration'] <= 60)].copy()
    print(f"Data filtered: {df_filtered.shape[0]} records")

    # Cast IDs to string after ensuring they are of object type
    df_filtered['PULocationID'] = df_filtered['PULocationID'].astype('object').astype(str)
    df_filtered['DOLocationID'] = df_filtered['DOLocationID'].astype('object').astype(str)
    print("IDs casted to string")

    # Prepare feature list of dictionaries
    dicts = df_filtered[['PULocationID', 'DOLocationID']].to_dict(orient='records')
    print("Converted to list of dictionaries")

    # Fit a dictionary vectorizer
    dv = DictVectorizer()
    X_train = dv.fit_transform(dicts)
    print(f"Dictionary vectorizer fitted: {X_train.shape}")

    # Prepare the target variable
    y_train = df_filtered['duration'].values
    print(f"Target variable prepared: {y_train.shape}")

    # Train a linear regression model
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    print("Linear regression model trained")

    # Print the intercept of the model
    print(f"Model intercept: {lr.intercept_}")

    # Return the dictionary vectorizer and the model
    return dv, lr
    ```
    
  ![image](https://github.com/garjita63/mlops-zoomcamp-2024/assets/77673886/a8ef78de-d3ec-4d3e-b8d1-d1dc7843f757)

### Answer of Question 5:


## Question 6. Register the model

The model is trained, so let's save it with MLFlow.

If you run mage with docker-compose, stop it with Ctrl+C or

    ``
    docker-compose down
    ``

Let's create a dockerfile for mlflow, e.g. mlflow.dockerfile:

    ```
    FROM python:3.10-slim
    
    RUN pip install mlflow==2.12.1
    
    EXPOSE 5000
    
    CMD [ \
        "mlflow", "server", \
        "--backend-store-uri", "sqlite:///home/mlflow/mlflow.db", \
        "--host", "0.0.0.0", \
        "--port", "5000" \
    ]
    ```

And add it to the docker-compose.yaml:

    ```
      mlflow:
        build:
          context: .
          dockerfile: mlflow.dockerfile
        ports:
          - "5000:5000"
        volumes:
          - "${PWD}/mlflow:/home/mlflow/"
        networks:
          - app-network
    ```

Note that app-network is the same network as for mage and postgre containers. If you use a different compose file, adjust it.

We should already have mlflow==2.12.1 in requirements.txt in the mage project we created for the module. If you're starting from scratch, add it to your requirements.

Next, start the compose again and create a data exporter block.

In the block, we

    - Log the model (linear regression)
    - Save and log the artifact (dict vectorizer)

If you used the suggested docker-compose snippet, mlflow should be accessible at http://mlflow:5000.

Find the logged model, and find MLModel file. What's the size of the model? (model_size_bytes field):
    - 14,534
    - 9,534
    - 4,534
    - 1,534
    
  Note: typically we do two last steps in one code block

### Answer of Question 6:


## Submit the results

- Submit your results here: https://courses.datatalks.club/mlops-zoomcamp-2024/homework/hw3
 - If your answer doesn't match options exactly, select the closest one
