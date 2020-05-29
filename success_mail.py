import smtplib
s = smtplib.SMTP('smtp.gmail.com', 587)
s.starttls()
s.login("vinujakhatode2050@gmail.com", "yourpassword")

message = "Subject: Testing" +"\n" + "Hey developer, the model is trained and giving good accuracy!"

s.sendmail("vinujakhatode2050@gmail.com", "vinujakhatode@gmail.com", message)

print("Mail has been sent to developer.")
s.quit() 
