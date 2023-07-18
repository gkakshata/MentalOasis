from flask import Flask, render_template, request
import text2emotion

app = Flask(__name__, template_folder='templates')

@app.route('/')
def home():
    return render_template('index.html')



@app.route('/analyze', methods=['POST'])
def analyze():
    text = request.form['text']
    emotions = te.get_emotion(text)
    return render_template('result.html', emotions=emotions)


if __name__ == '__main__':
    app.run(debug=True)





