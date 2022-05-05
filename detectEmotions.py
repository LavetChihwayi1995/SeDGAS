# import the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
#from pygame import mixer
import numpy as np
import imutils
import time
import cv2
import os
import math

#system libraries
import os
import sys
from threading import Timer
import shutil
import time

# ------------------------------------------------------------------------------------------------
from flask import Flask, render_template, Response,request,jsonify,redirect,url_for
from flask_mysqldb import MySQL
#---------------------------------------------------------------------------------------------------
# Import modules
import smtplib, ssl
## email.mime subclasses
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
### Add new subclass for adding attachments
##############################################################
from email.mime.application import MIMEApplication
##############################################################
## The pandas library is only for generating the current date, which is not necessary for sending emails
import pandas as pd
#-------------------------------------
import pyqrcode

from pyzbar.pyzbar import decode

app = Flask(__name__)

# ------------------------------------------------------------------------------------------------
# MySQL configurations
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'SeGas'
app.config['MYSQL_CURSORCLASS'] = 'DictCursor'
mysql=MySQL(app)
# -----------------------------------------------------------------------------------------------------
encryptedDetails =""


# --------------------------------------------------------------------------------------------------------
#Routes
# -------------------------------------------------------------------------------------
#Login 
@app.route("/login", methods=['GET','POST'])     
def login():
    if request.method=="POST":
        username=request.form["username"]
        password=request.form["password"]
        cur = mysql.connect.cursor()
        cur.execute("SELECT username,password,Grade FROM login ln JOIN loginDetails ld ON ln.loginId = ld.loginId")
        users = cur.fetchall()
        for user in users:
            if user["username"] ==username and user["password"]==password and user["Grade"]=="Resident":
                print("in resident")
                cursor = mysql.connection.cursor()
                query = "select ln.loginId, trackComm from login ln join logindetails ld on ln.loginId=ld.id where Username='{}' and Password='{}' ".format(username,password)
                cursor.execute(query)

                result = cursor.fetchone()
                result=(result['trackComm']) #Tracking community Id
                print(result)
                return redirect(url_for('resident', result=result)) 

            elif user["username"] ==username and user["password"]==password and user["Grade"]=="Admin":
                print("in Admin")
                cursor = mysql.connection.cursor()
                query = "select ln.loginId, trackComm from login ln join logindetails ld on ln.loginId=ld.id where Username='{}' and Password='{}' ".format(username,password)
                cursor.execute(query)

                result = cursor.fetchone()
                result=(result['trackComm']) #Tracking community Id
                print(result)

                return redirect(url_for('video', result=result))

            elif user["username"] ==username and user["password"]==password and user["Grade"]=="Guard":
                return redirect(url_for('guard'))
            else:
                print("In else")
    return render_template("login.html")
#Signup-member
@app.route("/signupmember", methods=['GET','POST'])
def signupmember():
    return render_template("member_signup.html")   

#Signup
@app.route("/signup", methods=['GET','POST'])
def signup():
    return render_template("signup.html")

#Community guards -----------------------------------------------------------------
@app.route("/guard", methods=['GET','POST'])
def guard():
    return render_template("guard.html")

@app.route('/scanVideo')
def scanVideo():
    return Response(detectQR(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/livesearch2",methods=["POST","GET"])
def livesearch2():
    searchbox = request.form.get("text")
    cursor = mysql.connection.cursor()
    query = "select Happy, Nutral, Sad from totalemotions"
    cursor.execute(query)
    result = cursor.fetchall()
    #print(result)     
    return jsonify(result)
@app.route("/livesearch3",methods=["POST","GET"])
def livesearch3():
    decisionData = request.form.get("text")
    decisionData=(decisionData.split("."))
    decision=decisionData[2]
    RId=decisionData[1]
    location=str(decisionData[3])
    locationData=(location.split("/"))
    print(locationData)
    residentId=(locationData[len(locationData)-1],)
    residentId=(residentId[0])
    print(residentId)
    print("ID:{} and DECISION:{}".format(RId,decision))

    cursor = mysql.connection.cursor()
    query = "select VId from visitor where trackIndex={} and RId={}".format(RId,residentId)
    cursor.execute(query)
    result = cursor.fetchone()
    print(result)
    VId=int(result['VId'])
    print("VId::::::{}".format(VId))

    cursor2 = mysql.connection.cursor()
    query2 = " select name, surname, Email, Sex  from visitor where VId={}".format(VId);
    cursor2.execute(query2)
    result2 = cursor2.fetchone()
    # print(result2)
    # print("RESULT:::{}".format(result2['name']))
    # print("This is visitor :{} {} with visitor ID:{}".format(result2['name'],result2['surname'],VId))
    encryptedDetails=str(result2['name'])# +" "+ str(result2['surname']) + " visiting "+ str(VId)
    emailTo=str(result2['Email'])
    gender=str(result2['Sex'])
    # select VId from visitor where trackIndex=1 and RId=4
    if decision=="accepted":
        cur=mysql.connection.cursor()
        cur.execute("UPDATE RequestStatus set RequestAccepted='Accepted' WHERE VId={}".format(VId))
        print("Updated at visitor NO:{VId}")
        # cur.execute("UPDATE RequestStatus rs join visitor v ON v.VId=rs.VId  set RequestAccepted='Accepted' WHERE  rs.VId=5 and name='lavet' and Surname='chihwayi' and RequestedOnDay='Null'");
        # Commit should be after the qrcode generation and email seding...............?????
        
        
        #Sending Mail
        try:
            genarateQrCode(encryptedDetails)
            email=emailTo;
            sendEmail(email,encryptedDetails, gender)
            mysql.connection.commit()
            print("commited  db")
        except Exception as e:
            print ("Connection problem")
        
    elif decision=="rejected":
        print("decision")
        cur=mysql.connection.cursor()
        print("connection")
        cur.execute("UPDATE RequestStatus set RequestRejected='Rejected' WHERE VId={}".format(VId));
        print("cur")
        mysql.connection.commit()
        print("Commited reject")
    #update RequestStatus rs join visitor v ON v.VId=rs.VId  set RequestAccepted="Accepted" WHERE  v.RId=1 and name="Victor" and Surname="Mapupu" and RequestedOnDay="2022-04-19"
    return jsonify("{result:kkk}")

@app.route("/livesearch4",methods=["POST","GET"])
def livesearch4():
    searchbox = request.form.get("text")
    print("This is location:{}".format(searchbox))
    return jsonify("{result:kkk}")


def detectQR():
    #img = cv2.imread('1.png')
    vs = cv2.VideoCapture(1)
    vs.set(3,640)
    vs.set(4,480)

    while True:
        success, frame = vs.read()
        #frame = imutils.resize(frame, width=400)
        #original_frame = frame.copy()
        #frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        #frame = cv2.cvtColor(frame,cv2.COLOR_GRAY2BGR)

        #success, img = cap.read()
        for barcode in decode(frame):
            myData = barcode.data.decode('utf-8')
            print(myData)
            pts = np.array([barcode.polygon],np.int32)
            pts = pts.reshape((-1,1,2))
            cv2.polylines(frame,[pts],True,(255,0,255),5)
            pts2 = barcode.rect
            cv2.putText(frame,myData,(pts2[0],pts2[1]),cv2.FONT_HERSHEY_SIMPLEX,
            0.9,(255,0,255),2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n') 
#---------------------------------------------------------------------------------------------------------------------------------------------
#Community residents
@app.route("/resident/<result>", methods=['GET','POST'])
def resident(result):
    RId=(result)
    cur = mysql.connect.cursor()
    cur.execute("SELECT  name, surname, RequestedOnDay, RequestPending,RequestAccepted,RequestRejected FROM  visitor v LEFT JOIN RequestStatus rs ON v.VId=rs.VId  WHERE  v.RId={}".format(RId))
    data = cur.fetchall()

    cur = mysql.connect.cursor()
    cur.execute("select count(name) from visitor v join requeststatus rs on v.VId=rs.VId where rs.RequestAccepted='' and RequestRejected='' and v.RId={}".format(RId))
    data2 = cur.fetchall()
    return render_template("resident.html",data=data,data2=data2)
#---------------------------------------------------------------------------------------------------------------
#Community visitors
@app.route("/visitor", methods=['GET','POST'])      
def visitor():
    cur = mysql.connect.cursor()
    cur.execute("SELECT Name, Surname, Address FROM Residents")
    data = cur.fetchall()
    return render_template("visitor.html",data=data)
#---------------------------------------------------------------------------------------------------------------
#Streamings
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


# ---------------------------------------------------------------------------------------------------------
#---Emotions
def callHappy():
    with app.app_context():           
        cur=mysql.connection.cursor()
        cur.execute("UPDATE totalemotions SET Happy = Happy + 1 WHERE GId = 1")
        mysql.connection.commit()
        print("Saved")
def callNutral():
    with app.app_context():           
        cur=mysql.connection.cursor()
        cur.execute("UPDATE totalemotions SET Nutral = Nutral + 1 WHERE GId = 1")
        mysql.connection.commit()
        print("Saved")
@app.route("/callSad")
def callSad():
    with app.app_context():           
        cur=mysql.connection.cursor()
        cur.execute("UPDATE totalemotions SET Sad = Sad + 1 WHERE GId = 1")
        mysql.connection.commit()
        print("Saved")

#--------------------------------------------------------------------------------------------------------------
######Video
######Video
@app.route("/video/<result>", methods=['GET','POST'])
def video(result):
    print("This is result:{}".format(result))
    try:
        cur = mysql.connect.cursor()
        cur.execute("SELECT * FROM community WHERE commId={}".format(result))
        data = cur.fetchall()
        
        cur2 = mysql.connect.cursor()
        cur2.execute("SELECT COUNT(Status) FROM visitorEmotions WHERE Status='Happy'")
        data2 = cur2.fetchall()
        
        
        cur3 = mysql.connect.cursor()
        cur3.execute("SELECT COUNT(Status) FROM visitorEmotions WHERE Status='Sad'")
        data3 = cur3.fetchall()
        
        cur4 = mysql.connect.cursor()
        cur4.execute("SELECT COUNT(Status) FROM visitorEmotions WHERE Status='Nutral'")
        data4 = cur4.fetchall()
        
        cur5 = mysql.connect.cursor()
        cur5.execute("select name, happy, Nutral, Sad from guards g right join totalemotions te on g.GId=te.GId order by happy DESC")
        data5 = cur5.fetchall()

        cur = mysql.connection.cursor()
        #con = mysql.connect()
        cur.execute("UPDATE totalemotions SET Happy = Happy + 1 WHERE GId = 1")
        mysql.connection.commit()
        print("Saved")
        
    except Exception:
        # print("Something wrong")
        return("Failed to connect to Database")
    else:
        return render_template("video.html", data=data, data5=data5, result=result)

@app.route("/admin")
def admin():
    return render_template("admin.html")
@app.route("/index")
def index():
    return render_template("index.html")


@app.route("/visitation")
def visitation():
    return render_template("index.html")
    
@app.route("/livesearch",methods=["POST","GET"])
def livesearch():
    searchbox = request.form.get("text")
    cursor = mysql.connection.cursor()
    query = "select * from residents where Address LIKE '{}%' ".format(searchbox)
    cursor.execute(query)
    result = cursor.fetchall()
    #print(result)     
    return jsonify(result)

@app.route("/postToDB",methods=["GET","POST"])
def postToDB():
    if request.method=="POST":
        text=str(request.form["text"])
        x=text.split()#Spliting the data from frontend



        firstname=request.form["firstname"]
        lastname=request.form["lastname"]
        sex=request.form["sex"]
        phonenumber=request.form["phonenumber"]
        nationalId=request.form["nationalId"]
        RId=x[0]
        print("{}:{}".format(x[0],x[1]))
        address=request.form["address"]
        email=request.form["email"]
        cur=mysql.connection.cursor()
        cur.execute("select RId from residents where Address like '%{}%'".format(RId))
        rId=cur.fetchone()
        RId=rId.get("RId")
        print("We are in posrToDB. rId:{} and RId:{}".format(rId,RId))
        # selecting the count of visitors per specific resident
        cur2 = mysql.connect.cursor()
        cur2.execute("SELECT count(RId) from visitor where RId={}".format(RId))
        result = cur2.fetchone()
        print(result['count(RId)'])

        cur.execute("INSERT into visitor(name,surname,Sex,NationalId, RId, Address, PhoneNumber,Email,trackIndex) values ('{}','{}','{}','{}','{}','{}','{}','{}',{})".format(firstname,lastname,sex,nationalId,RId,address,phonenumber,email,int(result['count(RId)'] +1 )))
        mysql.connection.commit()

        cursor = mysql.connection.cursor()
        query = " select max(VId) from visitor"
        cursor.execute(query)
        result = cursor.fetchone()
        visitorId=int(result['max(VId)'])

        cur.execute("insert into requeststatus ( RequestPending,RequestAccepted,RequestRejected, VId) values ('Pending','','',{})".format(visitorId))
        mysql.connection.commit()


        #Gerating QR Code
        # encryptedDetails=str(firstname) +" "+ str(lastname) + " visiting "+ str(RId)
        # genarateQrCode(encryptedDetails)
        # #Sending Mail
        # sendEmail(email,encryptedDetails)
        # #-------------
        print(rId.get("RId"))
        return text

@app.route("/postToDB2",methods=["GET","POST"])
def postToDB2():
    if request.method=="POST":
        
        membername=request.form["member_name"]
        membersurname=request.form["member_surname"]
        memberpassword=request.form["member_password"]
        memberpasswordconfirm=request.form["member_password_confirm"]
        gender=request.form["gender"]
        #username=request.form["user"]
        commname=request.form["comm_name"]
        commaddr=request.form["comm_addr"]
        commdescr=request.form["comm_descr"]
        memberemail=request.form["member_email"]
        membernationalid=request.form["member_national_id"]
        memberaddress=request.form["member_address"]
        memberanswerqn=request.form["member_answer_qn"]

        cur=mysql.connection.cursor()
       # 
        cur.execute("insert into Administrators(membername,membersurname,gender,memberemail,membernationalid,memberaddress,memberanswerqn) values ('{}','{}','{}','{}','{}','{}','{}')".format(membername,membersurname,gender, memberemail,membernationalid, memberaddress,memberanswerqn))
       # 
        mysql.connection.commit()

        cur2=mysql.connection.cursor()
        cur2.execute("insert into community(commName,commLocation) values ('{}','{}')".format(commname,commaddr))
        mysql.connection.commit()

        # ----------------------------------------------
        # Trackig commnity id and index
        # ------------------------------------------------
        cursor = mysql.connection.cursor()
        query = "SELECT max(commId) FROM community"
        cursor.execute(query)
        result = cursor.fetchone()
        trackComm=int(result['max(commId)'])

        cursor = mysql.connection.cursor()
        query = "SELECT count(commId) FROM community WHERE commId={}".format(trackComm)
        cursor.execute(query)
        result = cursor.fetchone()
        print(result)
        trackIndex=int(result['count(commId)'])

        # --------------------------------------------------

        cur2=mysql.connection.cursor()
        cur2.execute("insert into login(Username, Password) values ('{}','{}')".format(membername  ,memberpassword))
        mysql.connection.commit()

        cur = mysql.connect.cursor()
        cur.execute("select max(loginId) from login")
        data = cur.fetchone()
        data=(data['max(loginId)'])

        cur2=mysql.connection.cursor()
        cur2.execute("insert into logindetails(loginId, Grade, Email,  Address, trackIndex, trackComm) values ({},'{}','{}','{}',{},{})".format(str(data),"Admin", memberemail,memberaddress,trackIndex, trackComm))
        mysql.connection.commit()
        #------------
        return  commname


@app.route("/postToDB3",methods=["GET","POST"])
def postToDB3():
    # text=str(request.form["text"])
    # print("This is id from save:{}".format(text))
    # if request.method=="POST":
    #     text=str(request.form["text"])
    #     print("This is id from save:{}".format(text))


    #     membername=request.form["member_name"]
    #     membersurname=request.form["member_surname"]
    #     memberpassword=request.form["member_password"]
    #     memberpasswordconfirm=request.form["member_password_confirm"]
    #     gender=request.form["gender"]
    #     #username=request.form["user"]
    #     # commname=request.form["comm_name"]
    #     commaddr=request.form["comm_addr"]
    #     commdescr=request.form["comm_descr"]
    #     memberemail=request.form["member_email"]
    #     membernationalid=request.form["member_national_id"]
    #     memberaddress=request.form["member_address"]
    #     memberanswerqn=request.form["member_answer_qn"]

        
    #     cur=mysql.connection.cursor()
    #     cur.execute("insert into residents( Name,Surname,Sex, NationalId,CommId,Address,Email) values ('{}','{}','{}','{}',1,'{}','{}')".format(membername,membersurname,gender,membernationalid, memberaddress,memberemail)) 
    #     mysql.connection.commit()

    #     cur2=mysql.connection.cursor()
    #     cur2.execute("insert into login(Username, Password) values ('{}','{}')".format(membername  ,memberpassword))
    #     mysql.connection.commit()

    #     # cur2=mysql.connection.cursor()
    #     # cur2.execute("insert into logindetails(loginId, Grade, Email,  Address, trackIndex, trackComm) values ({},'{}','{}','{}',{},{})".format(str(data),"Resident", memberemail,memberaddress,trackIndex, trackComm))
    #     # mysql.connection.commit()

    return "Done r"

# --------------------------------------------------------------------------------------
# End OF Routes
# -----------------------------------------------------------------------------------------------

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# ----------------------------------------------------------------------------------------
# Methods
# ------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------
detections = None #Initializing detections

def detect_and_predict_mask(frame, faceNet, maskNet,threshold):
    # grab the dimensions of the frame and then construct a blob
    # from it
    global detections 
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),(104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the face detections
    faceNet.setInput(blob)
    detections = faceNet.forward()

    # initialize our list of faces, their corresponding locations,
    # and the list of predictions from our face mask network
    faces = []
    locs = []
    preds = []
    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the confidence is
        # greater than the minimum confidence
        try:
            if confidence >threshold:
                # compute the (x, y)-coordinates of the bounding box for
                # the object
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # ensure the bounding boxes fall within the dimensions of
                # the frame
                (startX, startY) = (max(0, startX), max(0, startY))
                (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
                # extract the face ROI, convert it from BGR to RGB channel
                # ordering, resize it to 224x224, and preprocess it
                face = frame[startY:endY, startX:endX]
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = cv2.resize(face, (224, 224))
                face = img_to_array(face)
                face = preprocess_input(face)
                face = np.expand_dims(face, axis=0)
                
                # add the face and bounding boxes to their respective
                # lists
                locs.append((startX, startY, endX, endY))
                #print(maskNet.predict(face)[0].tolist())
                preds.append(maskNet.predict(face)[0].tolist())
        except Exception:
            pass
            # print("Something wrong")
    return (locs, preds)
# --------------------------------------------------------------------------------------------------------------
# SETTINGS
MASK_MODEL_PATH=os.getcwd()+"\\model\\emotion_model.h5"
FACE_MODEL_PATH=os.getcwd()+"\\face_detector" 
SOUND_PATH=os.getcwd()+"\\sounds\\alarm.wav" 
THRESHOLD = 0.5



def gen_frames(): 
    # Load Sounds
    #mixer.init()
    #sound = mixer.Sound(SOUND_PATH)

    # load our serialized face detector model from disk
    print("[INFO] loading face detector model...")
    prototxtPath = os.path.sep.join([FACE_MODEL_PATH, "deploy.prototxt"])
    weightsPath = os.path.sep.join([FACE_MODEL_PATH,"res10_300x300_ssd_iter_140000.caffemodel"])
    faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

    # load the face mask detector model from disk
    print("[INFO] loading emotion detector model...")
    maskNet = load_model(MASK_MODEL_PATH)

    # initialize the video stream and allow the camera sensor to warm up
    print("[INFO] starting video stream...")
    vs = VideoStream(0 + cv2.CAP_DSHOW ).start()
    time.sleep(2.0)

    labels=["happy","neutral","sad"]
    happy_count=0#Counting happines
    happy_len=[]#Counting happines len
    nutral_count=0#Counting nutrality
    nutral_len=[]#Counting nutrality len
    sad_count =0#Counting sadness
    sad_len =[]#Counting sadness len

    # loop over the frames from the video stream
    while True:
        # grab the frame from the threaded video stream and resize it
        # to have a maximum width of 400 pixels
        frame = vs.read()
        frame = imutils.resize(frame, width=400)
        original_frame = frame.copy()
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        frame = cv2.cvtColor(frame,cv2.COLOR_GRAY2BGR)
        
        # detect faces in the frame and determine if they are wearing a
        # face mask or not
        (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet,THRESHOLD)

        # loop over the detected face locations and their corresponding
        # locations
        
        for (box, pred) in zip(locs, preds):
            # unpack the bounding box and predictions
            (startX, startY, endX, endY) = box
            # include the probability in the label
            label = str(labels[np.argmax(pred)])
            # display the label and bounding box rectangle on the output
            # frame
            if label == "happy":
                happy_count=happy_count+1
                print("Happy count #"+ str(happy_count))
                happy_len.append(happy_count)
                with app.app_context():
                    callHappy()
                #cur = mysql.connect.cursor()
                #cur.execute("UPDATE totalemotions SET Happy = Happy + 1 WHERE GId = 1")
                #print("successfully added")
                # data = cur.fetchall()
                cv2.putText(original_frame, label, (startX, startY - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.45,(0,200,50), 2)
                cv2.rectangle(original_frame, (startX, startY), (endX, endY),(0,200,50), 2)
            elif label == "neutral":
                nutral_count=nutral_count+1
                print("Nutral count #"+ str(nutral_count))
                nutral_len.append(nutral_count)
                with app.app_context():
                    callNutral()

                cv2.putText(original_frame, label, (startX, startY - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.45,(255,255,255), 2)
                cv2.rectangle(original_frame, (startX, startY), (endX, endY),(255,255,255), 2)
            elif label == "sad":
                sad_count=sad_count+1
                print("Sad count #"+ str(sad_count))
                sad_len.append(sad_count)
                with app.app_context():
                    callSad() 

                cv2.putText(original_frame, label, (startX, startY - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.45,(0,50,200), 2)
                cv2.rectangle(original_frame, (startX, startY), (endX, endY),(0,50,200), 2)
        
        ret, buffer = cv2.imencode('.jpg', original_frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result
# ---------------------------------------------------------------------------------------------------
###Sending message to the email
#---------------------------------------------------------------------------------
# Define the HTML document
html = '''
    <html>
    <body>
        <h1>Authentication Key</h1>
        <p>Hello, Your request to visit is accepted.Please provide the QR Code below at the gate.</p>
        <!-- <iframe src="Curthbert.svg"  height="600px" width="500px"></iframe> -->
        <embed src='Cuthbert.svg'  height='600px' width='500px' />
        <!-- <img src = "cup.jpg" alt="My Happy SVG"/> -->


    </body>
    </html>
    '''

# Define a function to attach files as MIMEApplication to the email
##############################################################
def attach_file_to_email(email_message, filename):
    # Open the attachment file for reading in binary mode, and make it a MIMEApplication class
    with open(filename, "rb") as f:
        file_attachment = MIMEApplication(f.read())
    # Add header/name to the attachments    
    file_attachment.add_header(
        "Content-Disposition",
        f"attachment; filename= {filename}",
    )
    # Attach the file to the message
    email_message.attach(file_attachment)
##############################################################    
def sendEmail(x,y,z):
    # Set up the email addresses and password. Please replace below with your email address and password
    email_from = 'chihwayil95@gmail.com'
    password = 'mukanya071995.'
    email_to = x

    # Generate today's date to be included in the email Subject
    date_str = pd.Timestamp.today().strftime('%Y-%m-%d')

    # Create a MIMEMultipart class, and set up the From, To, Subject fields
    email_message = MIMEMultipart()
    email_message['From'] = email_from
    email_message['To'] = email_to
    title=""
    if z=="Male":
        title="Mr"
    else:
        title="Miss"
    email_message['Subject'] = f'Welcome Alington {title} {y} - {date_str}'

    # Attach the html doc defined earlier, as a MIMEText html content type to the MIME message
    email_message.attach(MIMEText(html, "html"))

    # Attach more (documents)
    ##############################################################
    attach_file_to_email(email_message, '{}.svg'.format(y))
    #attach_file_to_email(email_message, 'tybercon.jpg')
    #attach_file_to_email(email_message, 'tybercon.jpg')
    ##############################################################
    # Convert it as a string
    email_string = email_message.as_string()

    # Connect to the Gmail SMTP server and Send Email
    context = ssl.create_default_context()
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
        server.login(email_from, password)
        server.sendmail(email_from, email_to, email_string)
        print("Send")
#--------------------------------------------------------------------------------------
#Generating the qrcode 
def genarateQrCode(x):
    name= x
    # Declare a variable called 'text'
    text = name + "REQUEST ACCEPTED"
    # Pass the variable "text" into the parenthesis as the parameter:
    image = pyqrcode.create(text)

    # scale represents the size of the image so that you can tweak this parameter until 
    # you have the ideal size for the image. 

    image.svg("{}.svg".format(name), scale="5")


#-----------------------------------------------------------------------------------------------------
    
if __name__ == "__main__":
    app.run(debug=True)