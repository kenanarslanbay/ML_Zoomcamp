#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install grpcio==1.42.0 tensorflow-serving-api==2.7.0')


# In[2]:


get_ipython().system('pip install keras-image-helper')


# In[2]:


import grpc

import tensorflow as tf

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc


# In[4]:


host = 'localhost:8500'

channel = grpc.insecure_channel(host)

stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)


# In[7]:


from keras_image_helper import create_preprocessor


# In[8]:


preprocessor = create_preprocessor('xception', target_size=(299, 299))


# In[10]:


url = 'http://bit.ly/mlbookcamp-pants'

X = preprocessor.from_url(url)


# In[12]:


# Let's turn X array to proto format:

def np_to_protobuf(data):
    return tf.make_tensor_proto(data, shape=data.shape)


# In[15]:


pb_request = predict_pb2.PredictRequest()

pb_request.model_spec.name = 'clothing-model'
pb_request.model_spec.signature_name = 'serving_default'

pb_request.inputs['input_8'].CopyFrom(np_to_protobuf(X))


# In[16]:


pb_request


# In[17]:


pb_response = stub.Predict(pb_request, timeout=20.0)


# In[21]:


preds = pb_response.outputs['dense_7'].float_val


# In[20]:


classes = [
    'dress',
    'hat',
    'longsleeve',
    'outwear',
    'pants',
    'shirt',
    'shoes',
    'shorts',
    'skirts',
    't-shirt'
]


# In[23]:


dict(zip(classes, preds))


# In[ ]:




