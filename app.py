from flask import Flask, request, jsonify
import json
from generate_questions import generate_question
from livereload import Server

app = Flask(__name__)

@app.route('/')
def index():
    return app.send_static_file('index.html')

@app.route('/generate-questions', methods=['POST'])
def generate_questions_endpoint():
    data = request.get_json()
    context = data['context']
    question_count = int(data['questionCount'])
    incorrect_options_count = int(data['incorrectOptionsCount'])
    questions = generate_question(context, question_count, incorrect_options_count)
    return jsonify({'questions': json.dumps(questions)})

# if __name__ == '__main__':
#     app.run(debug=False)
if __name__ == '__main__':
    server = Server(app.wsgi_app)
    server.serve()