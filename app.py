from flask import Flask, request, jsonify
import json
from generate_questions import generate_question
from generate_questions import generate_FIB_question
from livereload import Server

app = Flask(__name__)

# Add this function to your app
def convert_input_json(input_json):
    output_json = [
        {"question": sentence, "answer": key}
        for sentence, key in zip(input_json["sentences"], input_json["keys"])
    ]
    return output_json

@app.route('/')
def index():
    return app.send_static_file('index.html')

@app.route('/generate-questions', methods=['POST'])
def generate_questions_endpoint():
    data = request.get_json()
    context = data['context']
    question_count = int(data['questionCount'])
    summarized_context, questions, correctAnswers, distractor1, distractor2, distractor3 = generate_question(context, question_count)
    response = {
        'summarized_context':summarized_context,
        'questions': questions,
        'correctAnswers': correctAnswers,
        'distractor1': distractor1,
        'distractor2': distractor2,
        'distractor3': distractor3
    }
    return jsonify(response)

# Add this new endpoint to your app
@app.route('/generate-FIB', methods=['POST'])
def convert_json_endpoint():
    data = request.get_json()
    context = data['context']
    question_count = int(data['questionCount'])
    output_json = generate_FIB_question(context,question_count)
    return output_json

if __name__ == '__main__':
    server = Server(app.wsgi_app)
    server.serve()
