import os
import pickle
import numpy as np
import pandas as pd
from sklearn import datasets
from django.conf import settings
from rest_framework import views
from rest_framework import status
from rest_framework.response import Response
from sklearn.ensemble import RandomForestClassifier
from scipy.fftpack import rfft
from sklearn.tree import DecisionTreeClassifier


class Train(views.APIView):
    def post(self, request):

        return Response(status=status.HTTP_200_OK)


class Predict(views.APIView):
    def post(self, request):
        predictions1 = []
        predictions2 = []
        predictions3 = []
        predictions4 = []

        columns_to_be_read = ['nose_x', 'nose_y', 'leftEye_x', 'leftEye_y', 'rightEye_x', 'rightEye_y',
                              'leftEar_x', 'leftEar_y', 'rightEar_x', 'rightEar_y', 'leftShoulder_x', 'leftShoulder_y',
                              'rightShoulder_x', 'rightShoulder_y', 'leftElbow_x', 'leftElbow_y', 'rightElbow_x',
                              'rightElbow_y', 'leftWrist_x', 'leftWrist_y', 'rightWrist_x', 'rightWrist_y', 'leftHip_x',
                              'leftHip_y', 'rightHip_x', 'rightHip_y', 'leftKnee_x', 'leftKnee_y', 'rightKnee_x',
                              'rightKnee_y', 'leftAnkle_x', 'leftAnkle_y', 'rightAnkle_x', 'rightAnkle_y']
        #mapping =  {"1":"Buy","2":"Communicate","3":"Fun","4":"Hope","5":"Mother","6.0":"Really"}
        mapping = {1.0: "Buy", 2.0: "Communicate", 3.0: "Fun", 4.0: "Hope", 5.0: "Mother", 6.0: "Really"}
        positions = []
        i =0 
        for entry in request.data:
            if i==120:
               break
            i += 1
            try:
                for key, val in entry.items():
                    if key == "score":
                        frameno = []
                        frameno.append(val)
                    else:
                        for subframe in val:
                            for key1, val1 in subframe.items():
                                if key1 == "score":
                                    frameno.append(val1)

                                elif key1 == 'position':
                                    frameno.extend(list(val1.values()))
                        positions.append(frameno)

            except Exception as err:
                return Response(str(err), status=status.HTTP_400_BAD_REQUEST)
        positions = rfft(positions)
        model_name1 = 'Random_forest.pkl'
        model_name2 = 'KNN.pkl'
        model_name3 = 'logistic_regresion.pkl'
        model_name4 = 'decision_tree.pkl'

        path1 = os.path.join(settings.MODEL_ROOT, model_name1)
        path2 = os.path.join(settings.MODEL_ROOT, model_name2)
        path3 = os.path.join(settings.MODEL_ROOT, model_name3)
        path4 = os.path.join(settings.MODEL_ROOT, model_name4)
        with open(path1, 'rb') as file1:
            model1 = pickle.load(file1)
        with open(path2, 'rb') as file2:
            model2 = pickle.load(file2)
        with open(path3, 'rb') as file3:
            model3 = pickle.load(file3)
        with open(path4, 'rb') as file4:
            model4 = pickle.load(file4)
        # for entry in positions:
        #     #model_name = entry.pop('model_name')
        #newData = positions
        
        predictions1 = list(model1.predict(positions))
        predictions2 = list(model2.predict(positions))
        
        predictions3 = list(model3.predict(positions))
        
        predictions4 = list(model4.predict(positions))
        


        res = {}
        res[1] = mapping[max(predictions1,key=predictions1.count)]
        res[2] = mapping[max(predictions2,key=predictions2.count)]
        res[3] = mapping[max(predictions3,key=predictions3.count)]
        res[4] = mapping[max(predictions4,key=predictions4.count)]
        #res[1] = str(positions)
        #res[2] = str(predictions2)
        #res[3] = str(predictions3)
        #res[4] = str(predictions4)

        #val = max(predictions,key=predictions.count)
        #res = mapping[val[0]]

        return Response(res, status=status.HTTP_200_OK)