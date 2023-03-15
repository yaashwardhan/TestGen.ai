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
    questions, correctAnswers, distractor1, distractor2, distractor3= generate_question(context, question_count)
    response = {
        'questions': questions,
        'correctAnswers': correctAnswers,
        'distractor1': distractor1,
        'distractor2': distractor2,
        'distractor3': distractor3
    }
    return jsonify(response)

# if __name__ == '__main__':
#     app.run(debug=False)
if __name__ == '__main__':
    server = Server(app.wsgi_app)
    server.serve()