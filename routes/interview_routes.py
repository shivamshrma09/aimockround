from flask import Blueprint, request, jsonify
from controllers.interview_controller import InterviewController

interview_bp = Blueprint('interview', __name__)
controller = InterviewController()

@interview_bp.route('/hr', methods=['POST'])
def hr_interview():
    data = request.get_json()
    result = controller.hr_interview(data)
    return jsonify(result)

@interview_bp.route('/technical', methods=['POST'])
def technical_interview():
    data = request.get_json()
    result = controller.technical_interview(data)
    return jsonify(result)

@interview_bp.route('/coding', methods=['POST'])
def coding_interview():
    data = request.get_json()
    result = controller.coding_interview(data)
    return jsonify(result)

@interview_bp.route('/behavioral', methods=['POST'])
def behavioral_interview():
    data = request.get_json()
    result = controller.behavioral_interview(data)
    return jsonify(result)