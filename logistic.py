import numpy as np
import pandas as pd
import seaborn as sns
import tkinter as tk
from tkinter import *
from tkinter import ttk
import matplotlib.pyplot as plt
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import speech_recognition as sr
import playsound # to play saved mp3 file
from gtts import gTTS # google text to speech
import os # to save/open files
import wikipedia# to calculate strings into formula
# to calculate strings into formula # to control browser operations
num = 1
def assistant_speaks(output):
    global num

    # num to rename every audio file
    # with different name to remove ambiguity
    num += 1
    print("PerSon : ", output)

    toSpeak = gTTS(text = output, lang ='en', slow = False)
    # saving the audio file given by google text to speech
    file = str(num)+".mp3 "
    toSpeak.save(file)

    # playsound package is used to play the same file.
    playsound.playsound(file, True)
    os.remove(file)

def get_audio():

    rObject = sr.Recognizer()
    audio = ''

    with sr.Microphone() as source:
        print("Speak...")

        # recording the audio using speech recognition
        audio = rObject.listen(source, phrase_time_limit = 5)
    print("Stop.") # limit 5 secs

    try:

        text = rObject.recognize_google(audio, language ='en-US')
        print("You : ", text)
        return text

    except:

        assistant_speaks("Could not understand your audio, PLease try again !")
        return 0



#load data
assistant_speaks("Hi i am tumor predictor AI")
data = pd.read_csv('cancer.csv')
#drop column
data=data.drop(['Unnamed: 32','id'],axis=1)
data.diagnosis = [1 if each == "M" else 0 for each in data.diagnosis]

#X and Y
y=data.iloc[:,0].values
#x=data.iloc[:,1:31].values
x=data[['radius_mean','texture_mean','perimeter_mean','smoothness_mean','concavity_se','texture_se']]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.8,random_state=0)
#I can scale data using stander scaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.fit_transform(x_test)

#train
reg=LogisticRegression(random_state=0)
reg.fit(x_train,y_train)
#GUI
root=Tk()
label1=Label(root,text="Cancer Prediction",font=('arial',20,'bold'),bg="black",fg="white")
label1.pack(side=TOP,fill=X)
label2=Label(root,text=" ",font=('arial',15,'bold'),bg="black",fg="white")
label2.pack(side=BOTTOM,fill=X)

label1=Label(root,text="Enter Radius_mean report:",font=('arial',9,'bold'))
label1.place(x=30,y=60)
name_entry=StringVar()
name_entry=ttk.Entry(root,textvariable=name_entry)
name_entry.place(x=210,y=60)

label2=Label(root,text="Enter Texture_mean report:",font=('arial',9,'bold'))
label2.place(x=30,y=100)
name_entry2=StringVar()
name_entry2=ttk.Entry(root,textvariable=name_entry2)
name_entry2.place(x=210,y=100)

label3=Label(root,text="Enter Perimeter_mean:",font=('arial',9,'bold'))
label3.place(x=400,y=60)
name_entry3=StringVar()
name_entry3=ttk.Entry(root,textvariable=name_entry3)
name_entry3.place(x=550,y=60)

label4=Label(root,text="Enter Smothness_Mean:",font=('arial',9,'bold'))
label4.place(x=400,y=100)
name_entry4=StringVar()
name_entry4=ttk.Entry(root,textvariable=name_entry4)
name_entry4.place(x=550,y=100)

label5=Label(root,text="Enter Concavity_se:",font=('arial',9,'bold'))
label5.place(x=30,y=150)
name_entry5=StringVar()
name_entry5=ttk.Entry(root,textvariable=name_entry5)
name_entry5.place(x=210,y=150)

label6=Label(root,text="Enter Texture_se:",font=('arial',9,'bold'))
label6.place(x=400,y=150)
name_entry6=StringVar()
name_entry6=ttk.Entry(root,textvariable=name_entry6)
name_entry6.place(x=550,y=150)

output=Text(root,width=20,height=5,background="light gray")
output.place(x=270,y=175)

def gett():

     a=float(name_entry.get())
     b=float(name_entry2.get())
     c=float(name_entry3.get())
     d=float(name_entry4.get())
     e=float(name_entry5.get())
     f=float(name_entry6.get())
     
     x_t=[[a,b,c,d,e,f]]
     pr = reg.predict(x_t)
     for i in range(len(pr)):
           if(pr[i]==0):
              
               assistant_speaks("I think, it is Benign tumar")
             
           else:
                assistant_speaks("I think , it is malignant tumor")
             
        
     



btn=ttk.Button(root,text="Click Here to make a predict",command=gett)
btn.place(x=250,y=300,width=200,height=50)

root.geometry('800x800')
root.mainloop()

#pr=reg.predict([[17.99,10.38,122.8,0.1184,0.05373,0.9053]])



        
ac1=reg.score(x_train,y_train)
assistant_speaks("Prediction accuracy is")
print(ac1*100)


