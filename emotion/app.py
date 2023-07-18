from flask import Flask, render_template, request, redirect
import text2emotion as te

app = Flask(__name__, template_folder='templates')

@app.route('/')
def home():
    return render_template('index.html')

# Define the analyze route
@app.route('/analyze', methods=['POST'])
def analyze():
    """
    This function analyzes the text entered by the user and returns the results.
    """
    # Get the text from the request
    text = request.form['text']

    # Get the emotions from the text
    emotions = te.get_emotion(text)

    # Render the result.html template with the emotions passed as a variable
    return render_template('result.html', emotions=emotions)

@app.route('/emotion/<emotion>')
def show_emotion(emotion):
    emotion = emotion.lower() # Convert the emotion name to lowercase
    return render_template(f'{emotion}.html') # Route to the corresponding emotion.html template

# Add new routes to redirect to different emotions
@app.route('/redirect/<emotion>')
def redirect_emotion(emotion):
    emotion = emotion.lower() # Convert the emotion name to lowercase
    
    # Redirect to the corresponding emotion URL
    if emotion == 'happy':
        return redirect('/emotion/happy')
    elif emotion == 'sad':
        return redirect('/emotion/sad')
    elif emotion == 'angry':
        return redirect('/emotion/angry')
    elif emotion == 'fear':
        return redirect('/emotion/fear')
    elif emotion == 'surprise':
        return redirect('/emotion/surprise')

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8000)
