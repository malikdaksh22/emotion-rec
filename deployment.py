from flask import Flask, url_for,render_template,Response
app=Flask(__name__)

@app.route("/")
def index():
    return render_template("fun.html")
@app.route("/video_feed")
def video_feed():
    return Response(genfunc(),mimetype='multipart/x-mixed-replace; boundary=frame')

def genfunc():
    import cv2
    from keras.models import load_model
    import numpy as np
    model = load_model("emotion_custom_vgg.h5")
    class_labels = {0: "Angry",
                    1: "Disgust",
                    2: "Fear",
                    3: "Happy",
                    4: "Sad",
                    5: "Surprise",
                    6: "Neutral"}
    vid = cv2.VideoCapture(0)
    fonty = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX
    face = cv2.CascadeClassifier(
        "cv2/data/haarcascade_frontalface_default.XML")
    while (1):
        res, frame = vid.read()

        faces = face.detectMultiScale(frame, 1.1, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
            newimg = frame[x:x + w, y:y + h]
            gray = cv2.cvtColor(newimg, cv2.COLOR_BGR2GRAY)
            gray = gray / 255
            gray = cv2.resize(gray, (48, 48), interpolation=cv2.INTER_AREA)
            text = class_labels[np.argmax(model.predict(gray[np.newaxis, :, :, np.newaxis])[0])]
            cv2.putText(frame, text, (x, y), fonty, 4, 3)

        cv2.imshow("img", frame)

        if (cv2.waitKey(1) == ord('q')):
            return;
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

if __name__ == '__main__':
    app.run()