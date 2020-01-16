from tkinter import *
from tkinter import ttk

# Create an instance
win = Tk()
win.title("Learn Combobox")

# create a Label
lb1 = Label(win, text="Below is a combobox 1", font="tahoma 12 normal")
lb1.grid(column=0, row=0, padx=8, pady=4)


def show_select_1():
 print("post_command: show_select")
 print(value.get())


# Define tkinter data type
data = ["a", "b", "c"]
value = StringVar()

# Create a combobox, and tighter it to value
cbx_1 = ttk.Combobox(win, width=12, height=8, textvariable=value, postcommand=show_select_1)
cbx_1.grid(column=0, row=1)

# add data to combobox
cbx_1["values"] = data

# # ======================================================================================================
# # create a Label
# lb2 = Label(win, text="Below is a combobox 2", font="tahoma 12 normal")
# lb2.grid(column=0, row=4, padx=8, pady=4)
#
#
# def show_data_2(*args):
#  print("Event: ComboboxSelected")
#  print(cbx_2.get())
#
#
# # Define tkinter data type
# data2 = ["a2", "b2", "c2", "d2", "e2"]
#
# # Create a combobox, and tighter it to value
# cbx_2 = ttk.Combobox(win, width=12, height=8)
# cbx_2.grid(column=0, row=5)
#
# # set cbx_2 as readonly
# cbx_2.configure(state="readonly")
#
# # add data to combobox
# cbx_2["values"] = data2
# # set the initial data [index =2] to shows up when win generated
# cbx_2.current(2)
#
# # bind a event
# cbx_2.bind("<<ComboboxSelected>>", show_data_2)

win.mainloop()