"""
PyAudio Example: Make a wire between input and output (i.e., record a
few samples and play them back immediately).

This is the callback (non-blocking) version.
"""

import pyaudio
import time

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
from tensorflow.python.framework import tensor_util
import tensorflow as tf
import grpc

WIDTH = 2
CHANNELS = 2
RATE = 22050
CHUNK = 4096
FORMAT = pyaudio.paInt16

visualize_channel = grpc.insecure_channel("0.0.0.0:50051")
visualize_stub = prediction_service_pb2_grpc.PredictionServiceStub(visualize_channel)


p = pyaudio.PyAudio()

stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

while True:
    data = stream.read(CHUNK)
    request = predict_pb2.PredictRequest()
    request.inputs['audio'].CopyFrom(
        tf.contrib.util.make_tensor_proto(data))
    result = visualize_stub.Predict(request, 10.0)


stream.stop_stream()
stream.close()

p.terminate()