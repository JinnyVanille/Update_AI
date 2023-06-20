from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

@app.route('/')
def index():
    messages = [
        {'sender': 'You', 'content': 'Hello!'},
        {'sender': 'ChatGPT', 'content': 'Hi there! How can I assist you?'},
        {'sender': 'You', 'content': 'I have a question about your product.'},
        {'sender': 'ChatGPT', 'content': 'Sure, go ahead and ask your question.'}
    ]
    return render_template('index.html', messages=messages)

@app.route('/chat', methods=['POST'])
def chat():
    message = request.form['message']
    response = 'ChatGPT: ' + message
    return jsonify({'message': response})


if __name__ == '__main__':
    app.run(debug=True)
