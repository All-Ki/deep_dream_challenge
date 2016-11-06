# main.py

from flask import Flask, render_template, Response
from camera import VideoCamera
import deep_dream

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

def gen(camera):
    sess, graph, t_input = deep_dream.setup()
    t = deep_dream.T(graph, 'mixed4c')
    while True:
        frame = camera.get_frame(sess, t_input, t)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
