from PyQt5 import QtCore, QtGui, QtWidgets
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import threading

data = []
labels = []
classes = 51
cur_path = os.getcwd() #To get current directory


classs = { 1:"Speed limit (5km/h)",
    2:"Speed limit (15km/h)",
    3:"Speed limit (30km/h)",
    4:"Speed limit (40km/h)",
    5:"Speed limit (50km/h)",
    6:"Speed limit (60 km/h)",
    7:"End of speed limit (70 km/h)",
    8:"Speed limit (80km/h)",
    9:"No straight or left turn",
    10:"No straight or right turn",
    11:"Don't go straight ",
    12:"No left turn    ",
    13:"No left or right turn ",
    14:"No right turn ",
    15:"No overtake ",
    16:"No U Turn ",
    17:"No Cars allowed ",
    18:"No Honk ",
    19:"End Speed limit 40 ",
    20:"End Speed limit 50 ",
    21:"Presence straight left",
    22: "Presence straight ",
    23:"Turn Left ahead   ",
    24:"Turn Left or rigth ahead ",
    25:"Turn right ahead",
    26:"Pass left side  ",
    27:"Pass right side ",
    28:"Roundable ",
    29:"Taxi Only ",
    30: "Compulsory sound Zone ",
    31: "Compulsory Cycle Zone ",
    32: "Compulsory U Turn  ",
    33:"Road division   ",
    34:"Traffic Light ahead ",
    35:"Be careful  ",
    36:"Pedestrians ",
    37:"Bicycle zone ",
    38:"children crossing  ",
    39:"Road right curve ahead ",
    40:"Road left curve ahead ",
    41:"Steep Hill Down ",
    42: "Steep Hill up  ",
    43:"Side road right ",
    44:"Side road left ",
    45:"zig zaq road ",
    46:"Road work ongoing",
    47:"zig zag road warning",
    48:"Railway crossing ",
    49:"No traffic road ",
    50:"No stopping standing",
    51:"Do not enter",
    }


#Retrieving the images and their labels
print ("Obtaining Images & its Labels.......")
for i in range (classes):
    path = os.path.join (cur_path, 'dataset/DATA/', str(i))
    images = os.listdir (path)
    for a in images:
            try:
                image = Image.open(path + '\\'+ a)
                image = image.resize((30,30))
                image = np.array(image)
                data.append(image)
                labels.append(i)
                #print("{0}) Loaded".format(a))
            except:
                print("Error loading image")
print("Dataset Loaded")


#Converting lists into numpy arrays
data = np.array(data)
labels = np.array (labels)

print(data.shape, labels.shape)
#Splitting training and testing dataset

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=50)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
                                                    
#Converting the labels into one hot encoding
y_train = to_categorical (y_train, 51)
y_test = to_categorical (y_test, 51)

class Ui_MainWindow(object):
    # Your UI code...
    def setupUi(self, Mainwindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize (800, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.BrowseImage = QtWidgets.QPushButton(self.centralwidget)
        self.BrowseImage.setGeometry (QtCore.QRect (160, 370, 151, 51))
        self.BrowseImage.setObjectName ("BrowseImage")
        self.imageLbl = QtWidgets.QLabel(self.centralwidget)
        self.imageLbl.setGeometry(QtCore.QRect(200, 80, 361, 261))
        self.imageLbl.setFrameShape(QtWidgets.QFrame.Box)
        self.imageLbl.setText("")
        self.imageLbl.setObjectName("image[b]")
        self.label_2 = QtWidgets.QLabel (self.centralwidget)
        self.label_2.setGeometry (QtCore.QRect(110, 20, 621, 20))
        font = QtGui.QFont()
        font.setFamily ("Courier New")
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.Classify = QtWidgets.QPushButton(self.centralwidget)
        self.Classify.setGeometry(QtCore.QRect(160, 450, 151, 51))
        self.Classify.setObjectName("Classify")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(430, 370, 111, 16))
        self.label.setObjectName("label")
        self.Training = QtWidgets.QPushButton(self.centralwidget)
        self.Training.setGeometry(QtCore.QRect(400, 450, 151, 51))
        self.Training.setObjectName("Training")
        self.textEdit = QtWidgets.QTextEdit(self.centralwidget)
        self.textEdit.setGeometry(QtCore.QRect (400, 390, 211, 51))
        self.textEdit.setObjectName("textEdit")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar= QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0,0,800,26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        
        self.retranslateUi (MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        
        self.BrowseImage.clicked.connect(self.loadImage)
        
        self.Classify.clicked.connect(self.classifyFunction)
        
        self.Training.clicked.connect(self.trainingFunction)

    def retranslateUi (self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.BrowseImage.setText(_translate("MainWindow", "Browse Image"))
        self.label_2.setText(_translate("MainWindow","ROAD SIGN RECOGNITION"))
        self.Classify.setText(_translate ("MainWindow", "Classify"))
        self.label.setText(_translate ("MainWindow", "Recognized Class"))
        self.Training.setText(_translate("MainWindow", "Training"))


    def loadImage(self):
      
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName (None, "Select Image","", "Image Files (*.png *.jpg *.jpeg *.bmp);;All Files (*)") #Ask for file
        if fileName: # If the user gives a file
            print(fileName)
            self.file=fileName
            pixmap = QtGui.QPixmap(fileName) # Setup pixmap with the provided image
            pixmap = pixmap.scaled(self.imageLbl.width(), self.imageLbl.height(), QtCore.Qt.KeepAspectRatio) #Scale pixmap
            self.imageLbl.setPixmap(pixmap) # Set the pixmap onto the label
            self.imageLbl.setAlignment(QtCore.Qt.AlignCenter) # Align the label to center

    def classifyFunction(self):
        try:
            model = load_model('test.keras')
            print("Loaded model from disk")

            path2 = self.file
            print(path2)

            test_image = Image.open(path2)
            test_image = test_image.resize((30, 30))
            test_image = np.expand_dims(test_image, axis=0)
            test_image = np.array(test_image)

            probabilities = model.predict(test_image)[0]  # Get the probabilities for each class
            predicted_class_index = np.argmax(probabilities)  # Get the index of the class with the highest probability
            print("Predicted class index:", predicted_class_index)

            if predicted_class_index + 1 in classs:
                sign = classs[predicted_class_index + 1]
                print("Recognized class:", sign)
                self.textEdit.setText(sign)
            else:
                print("Class label not found in dictionary.")
                self.textEdit.setText("Unknown Class")

        except Exception as e:
            print("An error occurred during classification:", str(e))
            self.textEdit.setText("Error: " + str(e))
        
    def trainingFunction(self):
        try:
            print("Training started...")
            self.textEdit.setText("Training under process...")

            # Your model initialization and training code...
            self.textEdit.setText("Training under process...")
            model = Sequential()
            model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu', input_shape=(30,30,3)))
            model.add (Conv2D (filters=32, kernel_size=(5, 5), activation='relu'))
            model.add(MaxPool2D(pool_size=(2, 2)))
            model.add(Dropout(rate=0.25))
            model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
            model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
            model.add (MaxPool2D(pool_size=(2, 2)))
            model.add(Dropout(rate=0.25))
            model.add(Flatten())
            model.add(Dense(256, activation='relu'))
            model.add(Dropout(rate=0.5))
            model.add(Dense(51, activation='softmax'))
            print ("Initialized model")

            # Compilation of the model
            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

            history = model.fit(X_train, y_train, batch_size=32, epochs=5, validation_data=(X_test, y_test)) 
            model.save("test.keras")
                  
            plt.figure(0)
            plt.plot(history.history['accuracy'], label='training accuracy')
            plt.plot(history.history['val_accuracy'], label='validation accuracy')
            plt.title('Accuracy')
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.savefig('Accuracy.png')                   

            plt.figure(1)
            plt.plot(history.history['loss'], label='training loss')
            plt.plot(history.history['val_loss'], label='validation loss')
            plt.title('Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend ()
            plt.savefig('Loss.png')
            self.textEdit.setText("Saved Model & Graph to disk")
            print("Training completed successfully.")
            self.textEdit.setText("Training completed")
        except Exception as e:
            print("An error occurred during training:", e)
            self.textEdit.setText("Error occurred during training")

    
        

    

if __name__ == "__main__":
    # Your main application code...
        import sys
        app = QtWidgets.QApplication(sys.argv)
        MainWindow = QtWidgets.QMainWindow()
        ui = Ui_MainWindow()
        ui.setupUi(MainWindow)
        MainWindow.show()
        sys.exit(app.exec_())
