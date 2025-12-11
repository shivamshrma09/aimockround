from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import re
import random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import json
import os

app = Flask(__name__)

class InterviewController:
    def __init__(self):
        self.load_datasets()
        self.setup_model()
    
    def load_datasets(self):
        try:
            self.df_behavioral = pd.read_csv('behavioral_round_questions.csv')
            self.df_interview = pd.read_csv('interview_data.csv')
            self.df_technical = pd.read_csv('technical_interview_questions.csv')
            self.df_hr = pd.read_csv('hr_round_questions.csv')
        except Exception as e:
            print(f"Error loading datasets: {e}")
            self.df_behavioral = pd.DataFrame()
            self.df_interview = pd.DataFrame()
            self.df_technical = pd.DataFrame()
            self.df_hr = pd.DataFrame()
    
    def setup_model(self):
        try:
            torch.random.manual_seed(0)
            self.model = AutoModelForCausalLM.from_pretrained(
                "microsoft/Phi-3.5-mini-instruct",
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True,
            )
            self.tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3.5-mini-instruct")
            self.pipe = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer)
        except Exception as e:
            print(f"Error setting up model: {e}")
            self.model = None
            self.tokenizer = None
            self.pipe = None

    def generate_response(self, messages):
        if not self.pipe:
            return [{"generated_text": "Model not available"}]
        
        try:
            generation_args = {
                "max_new_tokens": 60,
                "return_full_text": False,
                "temperature": 0.0,
                "do_sample": False,
            }
            return self.pipe(messages, **generation_args)
        except Exception as e:
            print(f"Error generating response: {e}")
            return [{"generated_text": "Error generating response"}]

    def extract_score(self, text):
        score_match = re.search(r'Score:\s*(\d+)', text)
        if score_match:
            score = int(score_match.group(1))
            return max(0, min(100, score))
        return 50

    def hr_interview(self, data):
        try:
            company_name = data.get('company_Name', 'Company')
            role = data.get('role', 'Role')
            start = data.get('start', 0)
            maxfollowup = data.get('maxfollowup', 3)
            experience = data.get('exprences', 0)
            package = data.get('package', 0)
            prev_questions = data.get('previous_questions', '')
            prev_answer = data.get('previous_answer', '')

            # Filter by role, experience, and salary range
            filtered_hr_df = self.df_hr.copy()
            if not filtered_hr_df.empty:
                if role:
                    filtered_df = filtered_hr_df[filtered_hr_df.get('role', '') == role]
                    filtered_hr_df = filtered_df if not filtered_df.empty else filtered_hr_df
                if experience:
                    filtered_df = filtered_hr_df[filtered_hr_df.get('experience', 0) == experience]
                    filtered_hr_df = filtered_df if not filtered_df.empty else filtered_hr_df
                if package:
                    # Filter by salary range
                    def salary_match(salary_range, target_package):
                        if pd.isna(salary_range): return True
                        try:
                            min_sal, max_sal = map(int, salary_range.split('-'))
                            return min_sal <= target_package <= max_sal
                        except: return True
                    
                    filtered_df = filtered_hr_df[filtered_hr_df.get('salary_range', '').apply(lambda x: salary_match(x, package))]
                    filtered_hr_df = filtered_df if not filtered_df.empty else filtered_hr_df
            
            if start == 0:
                examples_df = filtered_hr_df.sample(min(len(filtered_hr_df), 3), random_state=42) if not filtered_hr_df.empty else pd.DataFrame()
                
                prompt_examples = "Generate a professional HR question.\n"
                for i, row_data in examples_df.iterrows():
                    if 'question' in row_data and pd.notna(row_data['question']):
                        prompt_examples += f"Example: {row_data['question']}\n"

                messages = [
                    {"role": "system", "content": f"You are an experienced HR recruiter at {company_name}. Generate ONE professional HR question focusing on cultural fit, motivation, career goals, or soft skills for a {role} position. DO NOT ask technical questions."},
                    {"role": "user", "content": f"{prompt_examples}Generate one HR question about company culture, motivation, or career aspirations - NOT technical skills."}
                ]
                output = self.generate_response(messages)
                return {"success": True, "question": output[0]['generated_text']}
            else:
                if not prev_questions.strip() or not prev_answer.strip():
                    return {"success": False, "error": "Previous questions and answers required"}

                messages = [
                    {"role": "system", "content": "You are an experienced HR. Analyze the candidate's answer and provide feedback with score out of 100."},
                    {"role": "user", "content": f"QUESTION: {prev_questions}\n\nANSWER: {prev_answer}\n\nProvide detailed feedback and score (format: 'Score: XX')."}
                ]
                feedback_output = self.generate_response(messages)
                generated_text = feedback_output[0]['generated_text']
                score = self.extract_score(generated_text)

                result = {"success": True, "feedback": generated_text, "score": score}
                
                if maxfollowup > 0 and score < 80:
                    # Generate follow-up question for poor performance
                    followup_messages = [
                        {"role": "system", "content": "Generate a targeted follow-up HR question."},
                        {"role": "user", "content": f"Previous Q: {prev_questions}\nAnswer: {prev_answer}\nScore: {score}\n\nGenerate one follow-up question."}
                    ]
                    followup_output = self.generate_response(followup_messages)
                    result["followup_question"] = followup_output[0]['generated_text']
                    result["max_followup"] = maxfollowup - 1
                else:
                    # Generate new question for good performance or no follow-ups left
                    new_examples_df = filtered_hr_df.sample(min(len(filtered_hr_df), 3), random_state=random.randint(0, 1000)) if not filtered_hr_df.empty else pd.DataFrame()
                    
                    new_prompt_examples = "Generate a professional HR question.\n"
                    for i, row_data in new_examples_df.iterrows():
                        if 'question' in row_data and pd.notna(row_data['question']):
                            new_prompt_examples += f"Example: {row_data['question']}\n"
                    
                    new_messages = [
                        {"role": "system", "content": f"You are an experienced HR of {company_name}. Generate a new HR question for {role} to continue the interview."},
                        {"role": "user", "content": f"{new_prompt_examples}Generate one new HR question for the {role} position."}
                    ]
                    new_output = self.generate_response(new_messages)
                    result["new_question"] = new_output[0]['generated_text']
                    result["interview_continues"] = True

                return result
        except Exception as e:
            return {"success": False, "error": str(e)}

    def technical_interview(self, data):
        try:
            company_name = data.get('company_Name', 'Company')
            role = data.get('role', 'Role')
            start = data.get('start', 0)
            maxfollowup = data.get('maxfollowup', 3)
            experience = data.get('exprences', 0)
            package = data.get('package', 0)
            prev_questions = data.get('previous_questions', '')
            prev_answer = data.get('previous_answer', '')

            # Filter by role, experience, and salary range
            source_df = self.df_technical.copy()
            if not source_df.empty:
                if role:
                    filtered_df = source_df[source_df.get('role', '') == role]
                    source_df = filtered_df if not filtered_df.empty else source_df
                if experience:
                    filtered_df = source_df[source_df.get('experience', 0) == experience]
                    source_df = filtered_df if not filtered_df.empty else source_df
                if package:
                    # Filter by salary range
                    def salary_match(salary_range, target_package):
                        if pd.isna(salary_range): return True
                        try:
                            min_sal, max_sal = map(int, salary_range.split('-'))
                            return min_sal <= target_package <= max_sal
                        except: return True
                    
                    filtered_df = source_df[source_df.get('salary_range', '').apply(lambda x: salary_match(x, package))]
                    source_df = filtered_df if not filtered_df.empty else source_df

            # Extract technical topics from interview_data.csv
            technical_topics = []
            if not self.df_interview.empty:
                filtered_interview = self.df_interview[
                    (self.df_interview.get('Company Name', '') == company_name) &
                    (self.df_interview.get('Role', '') == role) &
                    (self.df_interview.get('Experience', 0) == experience)
                ]
                
                if not filtered_interview.empty:
                    topics_str = filtered_interview.iloc[0].get('technical_topics', '')
                    if pd.notna(topics_str):
                        technical_topics = [topic.strip() for topic in topics_str.split(',')]

            if start == 0:
                examples_df = source_df.sample(min(len(source_df), 3), random_state=42) if not source_df.empty else pd.DataFrame()
                
                prompt_examples = "Generate a technical question.\n"
                for i, row_data in examples_df.iterrows():
                    if 'question' in row_data and pd.notna(row_data['question']):
                        prompt_examples += f"Example: {row_data['question']}\n"

                # Include company-specific technical topics
                topics_context = f"Focus on these technical areas: {', '.join(technical_topics)}" if technical_topics else ""
                
                messages = [
                    {"role": "system", "content": f"You are a technical interviewer for {company_name}. Generate ONE focused technical question for {role} about a SINGLE topic only. Keep it specific and concise."},
                    {"role": "user", "content": f"{prompt_examples}\nFocus on ONE topic from: {', '.join(technical_topics) if technical_topics else 'general technical skills'}\n\nGenerate one specific question about ONE technology/concept only."}
                ]
                output = self.generate_response(messages)
                return {"success": True, "question": output[0]['generated_text'], "topics": technical_topics}
            else:
                if not prev_questions.strip() or not prev_answer.strip():
                    return {"success": False, "error": "Previous questions and answers required"}

                # Include feedback points in evaluation
                examples_df = source_df.sample(min(len(source_df), 3), random_state=42) if not source_df.empty else pd.DataFrame()
                
                feedback_examples = "Here are evaluation criteria:\n"
                for i, row_data in examples_df.iterrows():
                    feedback_points = []
                    for j in range(1, 4):  # feedback_point_1, feedback_point_2, feedback_point_3
                        point = row_data.get(f'feedback_point_{j}', '')
                        if pd.notna(point) and point.strip():
                            feedback_points.append(point)
                    if feedback_points:
                        feedback_examples += f"Example criteria: {'; '.join(feedback_points)}\n"

                messages = [
                    {"role": "system", "content": f"You are a technical interviewer for {company_name}. Analyze the technical answer and provide detailed feedback with score out of 100. Consider technical accuracy, depth of knowledge, and communication clarity."},
                    {"role": "user", "content": f"QUESTION: {prev_questions}\n\nANSWER: {prev_answer}\n\n{feedback_examples}\n\nProvide detailed technical feedback and score (format: 'Score: XX')."}
                ]
                feedback_output = self.generate_response(messages)
                generated_text = feedback_output[0]['generated_text']
                score = self.extract_score(generated_text)

                result = {"success": True, "feedback": generated_text, "score": score}
                
                # Follow-up logic for technical questions
                if maxfollowup > 0 and score < 80:
                    # Generate easier follow-up question
                    easy_df = source_df[source_df.get('difficulty_level', '') == 'Easy']
                    followup_source = easy_df if not easy_df.empty else source_df
                    
                    followup_examples = followup_source.sample(min(len(followup_source), 2), random_state=42) if not followup_source.empty else pd.DataFrame()
                    
                    followup_prompt = "Generate a simpler follow-up question.\n"
                    for i, row_data in followup_examples.iterrows():
                        if 'question' in row_data and pd.notna(row_data['question']):
                            followup_prompt += f"Example: {row_data['question']}\n"
                    
                    followup_messages = [
                        {"role": "system", "content": f"Generate a simpler technical follow-up question for {role} to help assess basic understanding."},
                        {"role": "user", "content": f"Previous Q: {prev_questions}\nAnswer: {prev_answer}\nScore: {score}\n\n{followup_prompt}\n\nGenerate one easier follow-up question."}
                    ]
                    followup_output = self.generate_response(followup_messages)
                    result["followup_question"] = followup_output[0]['generated_text']
                    result["max_followup"] = maxfollowup - 1
                else:
                    # Generate new question for good performance or no follow-ups left
                    new_examples_df = source_df.sample(min(len(source_df), 3), random_state=random.randint(0, 1000)) if not source_df.empty else pd.DataFrame()
                    
                    new_prompt_examples = "Generate a technical question.\n"
                    for i, row_data in new_examples_df.iterrows():
                        if 'question' in row_data and pd.notna(row_data['question']):
                            new_prompt_examples += f"Example: {row_data['question']}\n"
                    
                    topics_context = f"Focus on these technical areas: {', '.join(technical_topics)}" if technical_topics else ""
                    
                    new_messages = [
                        {"role": "system", "content": f"You are a technical interviewer for {company_name}. Generate ONE focused technical question for {role} about a SINGLE topic only."},
                        {"role": "user", "content": f"{new_prompt_examples}Pick ONE topic from: {', '.join(technical_topics) if technical_topics else 'technical skills'}\n\nGenerate one specific question about ONE concept only."}
                    ]
                    new_output = self.generate_response(new_messages)
                    result["new_question"] = new_output[0]['generated_text']
                    result["interview_continues"] = True
                    result["topics"] = technical_topics

                return result
        except Exception as e:
            return {"success": False, "error": str(e)}

    def behavioral_interview(self, data):
        try:
            company_name = data.get('company_Name', 'Company')
            role = data.get('role', 'Role')
            start = data.get('start', 0)
            maxfollowup = data.get('maxfollowup', 3)
            prev_questions = data.get('previous_questions', '')
            prev_answer = data.get('previous_answer', '')

            filtered_behavioral_df = self.df_behavioral.copy()
            if not filtered_behavioral_df.empty and role:
                filtered_df = filtered_behavioral_df[filtered_behavioral_df.get('role', '') == role]
                filtered_behavioral_df = filtered_df if not filtered_df.empty else filtered_behavioral_df
            
            if start == 0:
                examples_df = filtered_behavioral_df.sample(min(len(filtered_behavioral_df), 3), random_state=42) if not filtered_behavioral_df.empty else pd.DataFrame()

                prompt_examples = "Generate a behavioral question using STAR method.\n"
                for i, row_data in examples_df.iterrows():
                    if 'question' in row_data and pd.notna(row_data['question']):
                        prompt_examples += f"Example: {row_data['question']}\n"

                messages = [
                    {"role": "system", "content": f"You are an experienced behavioral interviewer for {company_name}. Generate one behavioral question for a {role}."},
                    {"role": "user", "content": f"{prompt_examples}Generate one new behavioral question."}
                ]
                output = self.generate_response(messages)
                return {"success": True, "question": output[0]['generated_text']}
            else:
                if not prev_questions.strip() or not prev_answer.strip():
                    return {"success": False, "error": "Previous questions and answers required"}

                messages = [
                    {"role": "system", "content": "Analyze using STAR method and provide feedback with score out of 100."},
                    {"role": "user", "content": f"QUESTION: {prev_questions}\n\nANSWER: {prev_answer}\n\nAnalyze using STAR method and provide feedback with score (format: 'Score: XX')."}
                ]
                feedback_output = self.generate_response(messages)
                generated_text = feedback_output[0]['generated_text']
                score = self.extract_score(generated_text)

                result = {"success": True, "feedback": generated_text, "score": score}
                
                if maxfollowup > 0 and score < 80:
                    # Generate follow-up question for poor performance
                    followup_messages = [
                        {"role": "system", "content": "Generate a targeted follow-up behavioral question."},
                        {"role": "user", "content": f"Previous Q: {prev_questions}\nAnswer: {prev_answer}\nScore: {score}\n\nGenerate one follow-up behavioral question."}
                    ]
                    followup_output = self.generate_response(followup_messages)
                    result["followup_question"] = followup_output[0]['generated_text']
                    result["max_followup"] = maxfollowup - 1
                else:
                    # Generate new question for good performance or no follow-ups left
                    new_examples_df = filtered_behavioral_df.sample(min(len(filtered_behavioral_df), 3), random_state=random.randint(0, 1000)) if not filtered_behavioral_df.empty else pd.DataFrame()
                    
                    new_prompt_examples = "Generate a behavioral question using STAR method.\n"
                    for i, row_data in new_examples_df.iterrows():
                        if 'question' in row_data and pd.notna(row_data['question']):
                            new_prompt_examples += f"Example: {row_data['question']}\n"
                    
                    new_messages = [
                        {"role": "system", "content": f"You are an experienced behavioral interviewer for {company_name}. Generate a new behavioral question for {role} to continue the interview."},
                        {"role": "user", "content": f"{new_prompt_examples}Generate one new behavioral question for the {role} position."}
                    ]
                    new_output = self.generate_response(new_messages)
                    result["new_question"] = new_output[0]['generated_text']
                    result["interview_continues"] = True

                return result
        except Exception as e:
            return {"success": False, "error": str(e)}

    def coding_interview(self, data):
        try:
            company_name = data.get('company_Name', 'Company')
            role = data.get('role', 'Role')
            start = data.get('start', 0)
            maxfollowup = data.get('maxfollowup', 3)
            experience = data.get('exprences', 0)
            package = data.get('package', 0)
            prev_questions = data.get('previous_questions', '')
            prev_answer = data.get('previous_answer', '')

            # Extract coding topics from interview_data.csv
            coding_topics = []
            if not self.df_interview.empty:
                filtered_interview = self.df_interview[
                    (self.df_interview.get('Company Name', '') == company_name) &
                    (self.df_interview.get('Role', '') == role) &
                    (self.df_interview.get('Experience', 0) == experience)
                ]
                
                if not filtered_interview.empty:
                    topics_str = filtered_interview.iloc[0].get('coding_topics', '')
                    if pd.notna(topics_str):
                        coding_topics = [topic.strip() for topic in topics_str.split(',')]

            if start == 0:
                # Generate coding question based on company-specific topics
                topics_context = f"Focus on these coding topics: {', '.join(coding_topics)}" if coding_topics else "Generate a general coding problem"
                
                messages = [
                    {"role": "system", "content": f"You are a technical interviewer for {company_name}. Generate one coding question for {role} with {experience} years experience. {topics_context}."},
                    {"role": "user", "content": f"Generate a coding problem relevant to {company_name} focusing on: {', '.join(coding_topics) if coding_topics else 'general programming concepts'}. Include clear requirements and expected approach."}
                ]
                output = self.generate_response(messages)
                return {"success": True, "question": output[0]['generated_text'], "topics": coding_topics}
            else:
                if not prev_questions.strip() or not prev_answer.strip():
                    return {"success": False, "error": "Previous questions and answers required"}

                # Include company-specific evaluation criteria
                evaluation_context = f"Evaluate based on {company_name} standards for {role}" if coding_topics else "Evaluate general coding skills"
                topics_context = f"Consider expertise in: {', '.join(coding_topics)}" if coding_topics else ""
                
                messages = [
                    {"role": "system", "content": f"You are a technical interviewer for {company_name}. {evaluation_context}. Analyze code quality, logic, efficiency, and provide score out of 100."},
                    {"role": "user", "content": f"PROBLEM: {prev_questions}\n\nSOLUTION: {prev_answer}\n\n{topics_context}\n\nProvide detailed feedback analyzing code quality, algorithm efficiency, and adherence to best practices. Give score (format: 'Score: XX')."}
                ]
                feedback_output = self.generate_response(messages)
                generated_text = feedback_output[0]['generated_text']
                score = self.extract_score(generated_text)

                result = {"success": True, "feedback": generated_text, "score": score, "topics": coding_topics}
                
                # Follow-up logic for coding questions
                if maxfollowup > 0 and score < 80:
                    # Generate easier follow-up based on topics
                    easier_topics = coding_topics[:2] if len(coding_topics) > 2 else coding_topics  # Use fewer topics for easier question
                    
                    followup_messages = [
                        {"role": "system", "content": f"Generate a simpler coding follow-up question for {company_name} {role} to assess basic programming skills."},
                        {"role": "user", "content": f"Previous Q: {prev_questions}\nAnswer: {prev_answer}\nScore: {score}\n\nFocus on: {', '.join(easier_topics) if easier_topics else 'basic programming'}\n\nGenerate one easier coding question."}
                    ]
                    followup_output = self.generate_response(followup_messages)
                    result["followup_question"] = followup_output[0]['generated_text']
                    result["max_followup"] = maxfollowup - 1
                else:
                    # Generate new question for good performance or no follow-ups left
                    topics_context = f"Focus on these coding topics: {', '.join(coding_topics)}" if coding_topics else "Generate a general coding problem"
                    
                    new_messages = [
                        {"role": "system", "content": f"You are a technical interviewer for {company_name}. Generate a new coding question for {role} to continue the interview. {topics_context}."},
                        {"role": "user", "content": f"Generate a new coding problem relevant to {company_name} focusing on: {', '.join(coding_topics) if coding_topics else 'general programming concepts'}. Include clear requirements."}
                    ]
                    new_output = self.generate_response(new_messages)
                    result["new_question"] = new_output[0]['generated_text']
                    result["interview_continues"] = True

                return result
        except Exception as e:
            return {"success": False, "error": str(e)}

# Initialize controller
controller = InterviewController()

# API Routes
@app.route('/')
def home():
    return {"message": "AI Interview System API", "status": "running", "endpoints": ["/api/hr", "/api/technical", "/api/behavioral", "/api/coding"]}

@app.route('/api/hr', methods=['POST'])
def hr_interview():
    data = request.get_json()
    result = controller.hr_interview(data)
    return jsonify(result)

@app.route('/api/technical', methods=['POST'])
def technical_interview():
    data = request.get_json()
    result = controller.technical_interview(data)
    return jsonify(result)

@app.route('/api/behavioral', methods=['POST'])
def behavioral_interview():
    data = request.get_json()
    result = controller.behavioral_interview(data)
    return jsonify(result)

@app.route('/api/coding', methods=['POST'])
def coding_interview():
    data = request.get_json()
    result = controller.coding_interview(data)
    return jsonify(result)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=7860, debug=False)