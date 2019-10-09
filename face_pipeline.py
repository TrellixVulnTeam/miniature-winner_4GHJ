import cv2
import grpc
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
from tensorflow.python.framework import tensor_util
from modules.prnet import PRNet
from modules.prnet_image_cropper import PRNetImageCropper
from modules.face_detector import FaceDetector

import tensorflow as tf

# For plotting
from PRNet.utils.cv_plot import plot_kpt, plot_vertices

import time
import numpy as np
import pymesh
import threading
from Queue import Queue


visualize_channel = grpc.insecure_channel("0.0.0.0:50051")
visualize_stub = prediction_service_pb2_grpc.PredictionServiceStub(visualize_channel)


q = Queue()
def worker():
    while True:
        item = q.get()
        result = visualize_stub.Predict(item, 10.0)

        q.task_done()

t = threading.Thread(target=worker)
t.daemon = True
t.start()



# channel = grpc.insecure_channel('0.0.0.0:8500')
channel = grpc.insecure_channel('204.57.7.38:8500')

stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

# Setup
PRNet.Setup()
FaceDetector.Setup()
PRNetImageCropper.Setup()


# cap = cv2.VideoCapture("./tensorflow_face_detection/media/test.mp4")
cap = cv2.VideoCapture(0)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))


out = None
depth_out = None
frame_num = 1490
ind = 0
read_time = -1
while True:
    frame_num -= 1
    ret, image = cap.read()
    if time.time() - read_time < 1.0/24:
        continue
    read_time = time.time()
    if ret == 0:
        break
    if out is None:
        out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(
            'M', 'J', 'P', 'G'), 24, (frame_width, frame_height))
        # depth_out = cv2.VideoWriter('output_depth.avi', cv2.VideoWriter_fourcc(
        #     'M', 'J', 'P', 'G'), 24, (frame_width, frame_height))
    print("frame_width {}, frame_height {}".format(frame_width, frame_height))
    pp_start_time = time.time()
    start_time = time.time()
    next_request = predict_pb2.PredictRequest()
    next_request.inputs['input_image'].CopyFrom(
      tf.make_tensor_proto(image))
    elapsed_time = time.time() - start_time
    print('serialization time cost: {}'.format(elapsed_time))

    start_time = time.time()
    face_detector = FaceDetector()
    face_detector.PreProcess(next_request, stub)
    face_detector.Apply()
    prnet_request = face_detector.PostProcess()
    elapsed_time = time.time() - start_time
    print('face_detector time cost: {}'.format(elapsed_time))

    if "bounding_box" in prnet_request.inputs:
        start_time = time.time()
    	prnet_image_cropper = PRNetImageCropper()
    	prnet_image_cropper.PreProcess(prnet_request, stub)
    	prnet_image_cropper.Apply()
    	next_request = prnet_image_cropper.PostProcess()
        elapsed_time = time.time() - start_time
        print('prnet_image_cropper time cost: {}'.format(elapsed_time))

        start_time = time.time()
        prn = PRNet()
        prn.PreProcess(next_request, stub)
        prn.Apply()
        final_request = prn.PostProcess();
        elapsed_time = time.time() - start_time
        print('prnet time cost: {}'.format(elapsed_time))

        start_time = time.time()
        kpt = tensor_util.MakeNdarray(final_request.inputs["prnet_output"])
        vertices = tensor_util.MakeNdarray(final_request.inputs["vertices"])
        print(vertices.shape)

        q.put(final_request)

#        show_img = plot_vertices(np.zeros_like(image), vertices)

#        show_img = image
        elapsed_time = time.time() - start_time
        print('plot vertices time cost: {}'.format(elapsed_time))
        # Display the resulting frame    
#    cv2.imshow('frame',show_img)
         
        # Press Q on keyboard to stop recording
#    if cv2.waitKey(1) & 0xFF == ord('q'):
#        break

    pp_elapse = time.time() - pp_start_time
    print('fps = {}'.format(1/pp_elapse))
        #start_time = time.time()
        #out.write(plot_vertices(np.zeros_like(image), vertices))
        #elapsed_time = time.time() - start_time

    #else:
        #out.write(image)

q.join()       # block until all tasks are done
