from os import stat
import tkinter as tk
from tkinter import *
from tkinter import scrolledtext
from chat import Bot_responses
class My_GUI():
    def __init__(self,Window):
        self.W = Window
        self.B1 = Button(master=self.W, text='SEND', font=('arial', 13), width=10, command=self.send)
        self.B1.place(x=270, y=430)

        self.L1 = Label(master=self.W, text='BOT', font=('arial', 15))
        self.L1.place(x = 30, y = 10)

        self.SCr_T = scrolledtext.ScrolledText(master=self.W,  wrap = tk.WORD,  width = 30, height = 16, font = ("arial", 15),state='disabled')
        self.SCr_T.place(x=30,y=50)

        self.T = Text(master=self.W, font=('arial', 15), width=20, height=1)
        self.T.place(x=30, y=430)
    def send(self):
        people = self.T.get('1.0','end')
        people_insert = 'You: '+people 

        self.T.delete('1.0','end')
        self.SCr_T.config(state='normal')
        self.SCr_T.insert('end',people_insert)
        Bot = 'Bot: '+Bot_responses(people=people)+'\n'
        self.SCr_T.insert('end',Bot)
        self.SCr_T.config(state='disabled')
    def responese(self,input):
        pass

W = Tk()
W.geometry('400x500+100+200')
W.title('CHAT')
My_GUI(W)
W.mainloop()