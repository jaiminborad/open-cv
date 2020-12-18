from flask import Flask,g,redirect,render_template,request,session,Response,url_for
from main_script_opencv import social_distance_detector, motion_detection, people_counter

class User:
    def __init__(self, id, username, password):
        self.id = id
        self.username = username
        self.password = password

    def __repr__(self):
        return f'<User: {self.username}>'

users = []
users.append(User(id=1, username='venom', password='password'))
users.append(User(id=2, username='user2', password='user2'))
users.append(User(id=3, username='user3', password='user3'))


app = Flask(__name__)
app.secret_key = 'secretkey'

@app.before_request
def before_request():
    g.user = None

    if 'user_id' in session:
        user = [x for x in users if x.id == session['user_id']][0]
        g.user = user


@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        session.pop('user_id', None)

        username = request.form['username']
        password = request.form['password']

        user = [x for x in users if x.username == username][0]
        if user and user.password == password:
            session['user_id'] = user.id
            return redirect(url_for('profile'))

        return redirect(url_for('login'))

    return render_template('login.html')

@app.route('/profile')
def profile():
    if not g.user:
        return redirect(url_for('login'))

    return render_template('home.html')


@app.route('/')
def home():
	"""Video streaming home page."""
	return render_template('home.html')

@app.route('/motion_detection.html')
def motion():
	return render_template('motion_detection.html')

@app.route('/social_detection_detector.html')
def social_distance():
	return render_template('social_detection_detector.html')

@app.route('/crowd_counter.html')
def crowd():
	return render_template('crowd_counter.html')

@app.route('/motion_feed')
def motion_feed():
	"""Video streaming route. Put this in the src attribute of an img tag."""
	return Response(motion_detection(),
					mimetype='multipart/x-mixed-replace; boundary=frame')
 
@app.route('/social_feed')
def social_feed():
	"""Video streaming route. Put this in the src attribute of an img tag."""
	return Response(social_distance_detector(),
					mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/crowd_feed')
def crowd_feed():
	"""Video streaming route. Put this in the src attribute of an img tag."""
	return Response(people_counter(),
					mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
	app.run(debug=True)