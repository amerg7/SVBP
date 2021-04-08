from tkinter import *
import os
import pickle

window = Tk()
window.title("Smart Vision")
window.geometry("200x400")

name = []
Id = []


def takePicture():
    os.system("python FaceTrainer.py")


def startProgram():
    os.system("python SmartVision.py")


def endProgram():
    exit()


def Save():
    name.append(enterName.get())
    Id.append(enterId.get())
    pickle.dump(name, open("names.dat", "wb"))
    pickle.dump(Id, open("id.dat", "wb"))
    print(name, Id)


lableName = Label(window, text="Enter your name")
lableName.grid()

enterName = Entry(window)
enterName.grid()

labelId = Label(window, text="Enter an id")
labelId.grid()

enterId = Entry(window)
enterId.grid(padx=10)

confirm = Button(window, text="Save", command=Save)
confirm.grid(pady=10)

takePicture = Button(window, text="Take Picture", command=takePicture)
takePicture.grid(pady=10)

Startprogram = Button(window, text="Start Smart Vision", command=startProgram)
Startprogram.grid(pady=10)

endprogram = Button(window, text="Exit", command=endProgram)
endprogram.grid(pady=10)

window.mainloop()
