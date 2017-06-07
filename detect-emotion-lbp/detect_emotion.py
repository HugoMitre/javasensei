#!/usr/bin/env python
# -*- coding: utf-8 -*

import cPickle # nativa, objetos
import glob # nativa, buscar archivos en carpetas.
import cv2 # Libreria de OpenCV para hacer compiler vision: procesamiento de imagenes (clasificador de cascada y otros filtros). 
import math 
import dlib # Libreria para el calculo de los 68 facial points (landmarks).
from sklearn.model_selection import cross_val_score # SciKit para realizar validaciones cruzadas (cross validation).
from sklearn.metrics import confusion_matrix # Scikt para obtener la matriz de confusión para mostrar la precisión por cada emoción.
from sklearn.svm import LinearSVC # SciKit para crear el objeto clasificador con SVM con un kernel tipo lineal
from sklearn.svm import SVC # Scikit para crear una estancia de un objeto LibSVM 
from util import * # Funciones propios de Francisco de procesamiento de imagenes contenido en /detect-emotion-lbp/util.py
import random


class detect_emotion(object):
    model = None #clasificador para ser creada con el clasificador de Scikit 
    td = None # td, training data
    tl = None # tl, training label
    pd = None # pd, prueba de datos
    pl = None # pl, prueba label
    emociones = ("Boredom", "Engagement", "Excitement", "Frustration")
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) # Filtro clahe -> binarisa la imagen

    detector = dlib.get_frontal_face_detector() # Detector de rostros frontales e.g. mirar la pantalla
    predictor = dlib.shape_predictor("data\shape_predictor_68_face_landmarks.dat") # Predictor de los 68 puntos del rostro, este es un predictor muy bueno

    def __init__(self, modelPath=None, XPath=None, yPath=None, loadFiles=True): # constructor de una clase que va hacer predicciones de una IMAGEN con los puntos X,Y del rostro
        if loadFiles == True:
            self.model = cPickle.load(open(modelPath, "rb"))
            if XPath is not None:
                self.td = cPickle.load(open(XPath, "rb"))
            if yPath is not None:
                self.tl = cPickle.load(open(yPath, "rb"))

    def predict(self, image): # Detecta la emoción a partir de una IMAGEN DE ROSTRO
        returnValue = (False, "Rostro no encontrado")

        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            clahe_image = self.clahe.apply(gray)
            result, img = self.__get_image__(clahe_image)
            if result:
                y = self.model.predict(np.array(img))
                returnValue = (True, self.emociones[y[0]])
        except:
            print("Ha ocurrido un error en la detección de rostros")

        return returnValue

    def __get_image__(self, image): #get_landmark
        landmarks_vectorised = [] # VECTOR DE CARACTERISTICAS 
        result = True

        detections = self.detector(image, 1) # DETECTA LAS COORDENADAS DEL ÁREA DEL RECTANGULO DEL ROSTRO DE FRENTE
        for d in detections:  # 
            shape = self.predictor(image, d)  # PREDICTOR DEFINE LAS COORDENADAS DE LOS 68 PUNTOS 
            xlist = []
            ylist = []
            for i in range(1, 68):  # GUARDA LAS 68 COORDENADAS EN xlist y y list de la imagen 
                xlist.append(float(shape.part(i).x))
                ylist.append(float(shape.part(i).y))

            xmean = np.mean(xlist)  # determinación del centro de gravedad (centro de la imagen del rostro en X) Get the mean of both axes to determine centre of gravity
            ymean = np.mean(ylist)  # determinación del centro de gravedad (centro de la imagen del rostro en Y)
            xcentral = [(x - xmean) for x in
                        xlist]  # get distance between each point and the central point in both axes
            ycentral = [(y - ymean) for y in ylist] # OBTENER TODAS LAS DISTANCIAS DEL CENTRO DE GRAVEDAD (NARIZ) AL EJE Y

            if xlist[26] == xlist[29]:  # If x-coordinates of the set are the same, the angle is 0, catch to prevent 'divide by 0' error in function
                anglenose = 0
            else:
                anglenose = int(math.atan((ylist[26] - ylist[29]) / (xlist[26] - xlist[29])) * 180 / math.pi)

            if anglenose < 0:
                anglenose += 90
            else:
                anglenose -= 90

            for x, y, w, z in zip(xcentral, ycentral, xlist, ylist):
                landmarks_vectorised.append(x) # 
                landmarks_vectorised.append(y)
                meannp = np.asarray((ymean, xmean))
                coornp = np.asarray((z, w))
                dist = np.linalg.norm(coornp - meannp) #distancia entre el punto central y coordenadas x,y de unos de los 68 puntos
                anglerelative = (math.atan((z - ymean) / (w - xmean)) * 180 / math.pi) - anglenose # ANGULO RELATIVO ARCOTANGENTE X,Y MENOS EL ARCOTANGENTE DE X,Y MEDIA (diferencia del rostro girado a uno recto)
                landmarks_vectorised.append(dist) # CARACTERISTICA ES LA DISTANCIA ENTRE X,Y Media y X,Y 
                landmarks_vectorised.append(anglerelative) # CARACTERISTICA DEL ANGULO RELATIVO DE ROTACIÓN 

        if len(detections) < 1:
            result = False
        return (result, landmarks_vectorised)

    @staticmethod
    def create_model_training(savePath=None, XPath=None, yPath=None, path=None, rostrosPath=None): # MODELO DE ENTRENAMIENTO 
        detect = detect_emotion(loadFiles=False) # CREA INSTANCIA DE LA CLASE CON LOS ATRIBUTOS VACIOS 
        training_data = []
        training_labels = []
        prediction_data = []
        prediction_labels = []
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) # FILTRO DE EQUALIZACIÓN DE HISTOGRAMAS  

        for emotion in detect.emotions:
            training, prediction = detect_emotion.get_files(emotion) # SE OBTIENE LA RUTA LOS ARCHIVOS DE LA EMOCIÓN QUE SE VA A EXTRATER CARACTERISTICAS
            # Append data to training and prediction list, and generate labels 0-7
            for item in training:
                image = cv2.imread(item)  # open image
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # convert to grayscale
                clahe_image = clahe.apply(gray) # APLICACIÓN DEL FILTRO DE EQUALIZACIÓN DE HISTOGRAMAS A IMAGEN A ESCALA DE GRISES.
                result, landmarks_vectorised = detect.__get_image__(clahe_image) # EXTRAE CARACTERISTICAS DE LA IMAGEN A ESCALA DE GRISES.
                if result:
                    training_data.append(landmarks_vectorised)  # AGREGA LAS CARACTERISTICAS AL CONJUNTO DE ENTRENAMIENTO
                    training_labels.append(detect.emotions.index(emotion)) # AGREGA LA ETIQUEDA DE LA EMOCIÓN -- 0..N donde 0 es Boredom... 

            for item in prediction: 
                image = cv2.imread(item)
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                clahe_image = clahe.apply(gray)
                result, landmarks_vectorised = detect.__get_image__(clahe_image)
                if result:
                    prediction_data.append(landmarks_vectorised)
                    prediction_labels.append(detect.emotions.index(emotion))
                    
        svm = SVC(kernel='linear', probability=True,tol=1e-3) # CREA INSTANCIA SVM
        svm.fit(training_data, training_labels) # ENTRENA EL SVM CON 
        detect.model = svm 
        detect.td = training_data
        detect.tl = training_labels
        detect.pd = prediction_data
        detect.pl = prediction_labels
        # GUARDA LOS DATOS
        cPickle.dump(training_data, open("data/td.x", "wb"))
        cPickle.dump(training_labels, open("data/tl.y", "wb"))
        cPickle.dump(prediction_data, open("data/pd.x", "wb"))
        cPickle.dump(prediction_labels, open("data/pl.y", "wb"))
        cPickle.dump(svm, open("data/modelo.m","wb"))
        # GUARDA EN ARCHIVO 
        return detect

    @staticmethod
    def get_files(emotion):  # Define function to get file list, randomly shuffle it and split 80/20
        files = glob.glob("D:\Subcorpus\%s\*.png" % emotion)
        random.shuffle(files)
        # top 100
        files = files[:120]
        training = files[:int(len(files) * 0.9)]  # get first 80% of file list
        prediction = files[-int(len(files) * 0.1):]  # get last 20% of file list
        return training, prediction

# detector = detect_emotion.create_model_training()
