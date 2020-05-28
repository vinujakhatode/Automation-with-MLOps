import smtplib
s = smtplib.SMTP('smtp.gmail.com', 587)
s.starttls()
s.login("vinujakhatode2050@gmail.com", "8888391845")

message = "Subject: Testing" +"\n" + "Hey developer, the the accuracy of the your model is not so good! So, model needs some tweaking, which will be done automatically to get better accuracy."

s.sendmail("vinujakhatode2050@gmail.com", "vinujakhatode@gmail.com", message)
print("Mail has been sent to developer.")
s.quit()