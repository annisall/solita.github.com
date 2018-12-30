---
layout: post
title: Machine learning pipeline something something
author: anniinasa
excerpt: Lorem ipsum 
tags:
- AWS
- SageMaker
- machine learning
- data science
---

## Introduction
Small group of Solita employees visited Amazon London office last November and participated workshop where we got known to AWS service called SageMaker. SageMaker turned out to be very interesting and in this blog post I'm going to tell more about it and demonstrate how it works with short code snippets. I'm also going to tell about some AWS services that have pretrained models and discuss pros and cons of using them.

## AWS SageMaker

SageMaker is an Amazon service that is designed to build, train and deploy machine learning models easily. For each step there are tools and functions which makes implementing the pipeline much faster. All the work is done in Jupyter Notebook, which have pre-installed Anaconda packages and libraries for for example Tensorflow. One can easily access data in their S3 buckets, too, from SageMaker notebooks. SageMaker also has multiple example notebooks so that getting started is very easy.

### Dataset
In the example snippets I use [MNIST dataset](https://www.kaggle.com/datamunge/sign-language-mnist) which contains labeled pictures of alphabets in sign language. They are 28x28 grey-scale pictures, which means each pixels contains integer between 0 and 255. Training data contains 27 455 pictures and test data 7127 pictures and they're stored in S3.

For importing and exploring the dataset I simply use pandas libraries. Pandas is able to read data straight from S3 bucket:

```python
import pandas as pd

bucket='<bucket-name>'
file_name = 'data-file.csv'

data_location = 's3://{}/{}'.format(bucket, file_name)

df=pd.read_csv(data_location)
```

From dataset I can see that first column is label for picture and remaining 784 columns are pixels. By reshaping the first row I can get the first image:

```python
from matplotlib import pyplot as plt

pic=df.head(1).values[0][1:].reshape((28,28))
plt.imshow(pic, cmap='gray')
plt.show()
```
![Image with alphabet d](/img/aws-sagemaker-example/first_img_sign.png)

### Build
Build phase in the case of AWS SageMaker means exploring and cleaning the data. If we wanted to keep data in csv format, we would have to remove the header row from the csv files in order to use SageMaker built-in algorithm and upload it back to S3. Instead, we'll convert the data into RecordIO protobuf format, which makes using built-in algorithms to train model more efficient. This can be done with following code and should be done for both training and test data:

```python
from sagemaker.amazon.common import write_numpy_to_dense_tensor
import boto3

def convert_and_upload(pixs, labels, bucket_name, data_file):
	buf = io.BytesIO()
	write_numpy_to_dense_tensor(buf, pixs, labels)
	buf.seek(0)

	boto3.resource('s3').Bucket(bucket_name).Object(data_file).upload_fileobj(buf)

pixels_train=df.drop('label', axis=1).values
labels_train=df['label'].values

convert_and_upload(pixels_train, labels_train, bucket, 'sign_mnist_train_rec')  
```
Of course, in this case the data is very clean already and usually a lot more work needs to be done in order to explore and clean the data properly, before uploading it back to for example S3. This is part where SageMaker doesn't really provide additional help, which is very understandable.

### Train
Now that data is cleaned and we can either use SageMaker built-in algorithms or use our own, maybe provided by sklearn or some other library. For yusing other than SageMaker built-in algorithms you would have to provide a Docker container for the training and validation tasks. More information about it can be found in [SageMaker documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-training-algo.html). In this case as we want to recognise alphabets from the pictures we use k-Nearest Neighbors -algorithm. It is one of the built-in algorithms in SageMaker, and can be used with very few lines of code:

```python
knn=sagemaker.estimator.Estimator(get_image_uri(boto3.Session().region_name, "knn"),
								 get_execution_role(),
								 train_instance_count=1,
								 train_instance_type='ml.m4.xlarge',
								 output_path='s3://{}/output'.format(bucket),
								 sagemaker_session=sagemaker.Session())
knn.set_hyperparameters(**{
	'k': 10,
	'predictor_type': 'classifier',
	'feature_dim': 784,
	'sample_size': 27455
})

in_config_test = sagemaker.s3_input(
	   s3_data='s3://{}/{}'.format(bucket,'sign_mnist_test_rec'))

in_config_train = sagemaker.s3_input(
	   s3_data='s3://{}/{}'.format(bucket,'sign_mnist_train_rec'))

knn.fit({'train':in_config_train, 'test': in_config_test})
```

So let's get into what happens there. Estimator is an interface for creating training task in SageMaker. We simply tell it which algorithm we want to use, how many EC2 instances we want for training, which type of instances they should be and where trained model should be stored.

Next we define hyperparameters for used algorithm, in this case k-nearest neighbor classifier. Given four parameters are mandatory, and the training job will fail without them. Instead of classifier we could have regressor for some other type of machine learning task. The algorithm has other, optional hyperparameters, for example for dimensionality reduction which the algorithm can perform as well. The hyperparameters are one of the factors when trying to make accurate model. 

Finally we need to define the path of training data. We do it by using [Channels](https://docs.aws.amazon.com/sagemaker/latest/dg/API_Channel.html) which are just named input sources for training algorithms. In this case as our data is in S3, we use s3_input class. Only train channel is required, but if test channel is given, too, training job also measures the accuracy of the resulting model. In this case I provide both.

For knn-algorithm only allowed datatypes are RecordIO protobuf and CSV formats. If we were to use CSV format, we would need to define it in configuration by defining named parameter content_type and assigning 'text/csv;label_size=0' as value. Otherwise only s3_data parameter is mandatory but there are optional parameters for example for shuffling data and to define whether the whole dataset should be replicated on every instance as a whole. When fit-function is called, SageMaker creates new training job and logs training process and duration into notebook. Past training jobs and information about them can be found by selecting 'Training jobs' in SageMaker side panel. There you can see given training/test data location and find information about model accuracy and logs of training job.

![Screenshot of training job dashboard](/img/aws-sagemaker-example/training_job.png)

### Deploy
The last step in order to provide predictions from the trained model is to set up an endpoint for it. This means that we automatically set up endpoint for real-time predictions and deploy trained model for it. This will create new EC2 instance which will take data as input and provide prediction as response. Following code is all needed for creating endpoint as deploying model to it:

```python
import time

def predictor_from_estimator(knn_estimator, estimator_name, instance_type, endpoint_name=None): 
    knn_predictor = knn_estimator.deploy(initial_instance_count=1, instance_type=instance_type,
                                        endpoint_name=endpoint_name)
    knn_predictor.content_type = 'text/csv'
    return knn_predictor


instance_type = 'ml.m5.xlarge'
model_name = 'knn_%s'% instance_type
endpoint_name = 'knn-ml-m5-xlarge-%s'% (str(time.time()).replace('.','-'))
print('setting up the endpoint..')
predictor = predictor_from_estimator(knn, model_name, instance_type, endpoint_name=endpoint_name)
```

and it can be called for example in following way:

```python
file = open("path_to_test_file.csv","rb")

predictor.predict(file)
```

which would return following response:

```python
b'{"predictions": [{"predicted_label": 6.0}, {"predicted_label": 3.0}, {"predicted_label": 21.0}, {"predicted_label": 0.0}, {"predicted_label": 3.0}]}'
```

because the file contained five pictures. In real life case if we wanted to provide interface for making real-time predictions we could use API Gateway and Lambda function. Lambda function can use boto3 library to connect to the created endpoint and get prediction. In API gateway we can setup API that calls lambda function once it gets a POST request and returns the prediction in response.
