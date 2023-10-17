import requests
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk

def send_sms():
    number = phone_no.get()
    messages = message.get("1.0", "end-1c")
    url = "https://www.fast2sms.com/dev/bulk"
    api = "put your api here"  # Substitua pelo seu código de API
    querystring = {"authorization": api, "sender_id": "FSTSMS", "message": messages,
                  "language": "english", "route": "p", "numbers": number}
    headers = {'cache-control': "no-cache"}
    requests.request("GET", url, headers=headers, params=querystring)
    messagebox.showinfo("Send SMS", "SMS has been sent successfully")

root = tk.Tk()
root.title("Send SMS")

img = Image.open('Backgound.jpg')
img = ImageTk.PhotoImage(img)
panel = tk.Label(root, image=img)
panel.pack(side="bottom", fill="both", expand="yes")

label = tk.Label(root, text="Send SMS", font=('verdana', 10, 'bold'))
label.place(x=210, y=10)

phone_no = tk.Entry(root, width=20, borderwidth=0, font=('verdana', 10, 'bold'))
phone_no.place(x=220, y=115)
phone_no.insert('end', 'phone number')

message = tk.Text(root, height=5, width=25, borderwidth=0, font=('verdana', 10, 'bold'))
message.place(x=190, y=140)
message.insert('end', 'Message')

send = tk.Button(root, text="Send Message", font=('verdana', 10, 'bold'), relief=tk.RIDGE, cursor='hand2', borderwidth=0, command=send_sms)
send.place(x=260, y=235)

root.mainloop()