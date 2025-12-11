import pandas as pd
import numpy as np
import re
import random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from sentence_transformers import SentenceTransformer
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

class InterviewController:
    def __init__(self):
        self.load_datasets()
        self.setup_vectorstores()
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
    
    def setup_vectorstores(self):
        try:
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            
            hr_docs = [Document(page_content=str(row)) for _, row in self.df_hr.iterrows()]
            behavioral_docs = [Document(page_content=str(row)) for _, row in self.df_behavioral.iterrows()]
            interview_docs = [Document(page_content=str(row)) for _, row in self.df_interview.iterrows()]
            
            self.hr_vectorstore = Chroma.from_documents(hr_docs, embeddings) if hr_docs else None
            self.behavioral_vectorstore = Chroma.from_documents(behavioral_docs, embeddings) if behavioral_docs else None
            self.interview_vectorstore = Chroma.from_documents(interview_docs, embeddings) if interview_docs else None
        except Exception as e:
            print(f"Error setting up vectorstores: {e}")
            self.hr_vectorstore = None
            self.behavioral_vectorstore = None
            self.interview_vectorstore = None
    
    def setup_model(self):
        try:
            torch.random.manual_seed(0)
            self.model = AutoModelForCausalLM.from_pretrained(
                "microsoft/Phi-3.5-mini-instruct",
                device_map="cpu",
                torch_dtype=torch.float16,
                trust_remote_code=True,
                attn_implementation="eager",
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
                "max_new_tokens": 500,
                "return_full_text": False,
                "temperature": 0.0,
                "do_sample": False,
            }
            return self.pipe(messages, **generation_args)
        except Exception as e:
            print(f"Error generating response: {e}")
            return [{"generated_text": "Error generating response"}]

    def extract_score(self, text):
        """Extract and validate score from generated text"""
        score_match = re.search(r'Score:\s*(\d+)', text)
        if score_match:
            score = int(score_match.group(1))
            return max(0, min(100, score))  # Enre score is between 0-100
        return 50  # Default score if extraction fails

    def validate_inputs(self, data):
        """Validate input data"""
        required_fields = ['company_Name', 'role', 'start']
        for field in required_fields:
            if field not in data:
                return False, f"Missing required field: {field}"
        
        if data.get('start') not in [0, 1]:
            return False, "Start must be 0 or 1"
        
        if data.get('start') == 1 and not data.get('previous_questions'):
            return False, "Previous questions required when start=1"
            
        return True, "Valid"
    
    def hr_interview(self, data):
        # Validate inputs
        is_valid, message = self.validate_inputs(data)
        if not is_valid:
            return {"success": False, "error": message}

        company_name = data.get('company_Name', 'Company')
        role = data.get('role', 'Role')
        round_type = data.get('round_type', 'HR Round')
        start = data.get('start', 0)
        max_followup = max(0, data.get('maxfollowup', 3))  # Ensure non-negative
        experience = data.get('exprences', 0)
        package = data.get('package', 0)
        previous_questions = data.get('previous_questions', '')
        previous_answer = data.get('previous_answer', '')
        
        try:
            # Filter HR CSV data
            filtered_hr_df = self.df_hr.copy()
            if not filtered_hr_df.empty:
                if 'role' in filtered_hr_df.columns and role:
                    filtered_hr_df = filtered_hr_df[filtered_hr_df['role'] == role]
                if 'experience' in filtered_hr_df.columns and experience:
                    filtered_hr_df = filtered_hr_df[filtered_hr_df['experience'] == experience]
            
            # Fallback to full dataset if filtered is empty
            if filtered_hr_df.empty:
                filtered_hr_df = self.df_hr.copy()
            
            if start == 0:
                # Initial question generation
                examples_df = filtered_hr_df.sample(min(len(filtered_hr_df), 5), random_state=42) if not filtered_hr_df.empty else pd.DataFrame()

                prompt_examples = "Here are some examples of HR questions:\n"
                for i, row_data in examples_df.iterrows():
                    prompt_examples += f"Example {i+1}:\n"
                    if 'question' in row_data and pd.notna(row_data['question']):
                        prompt_examples += f"Question: {row_data['question']}\n\n"

                # Fallback if no examples
                if not prompt_examples.strip() or prompt_examples == "Here are some examples of HR questions:\n":
                    prompt_examples = "Generate a professional HR interview question.\n"

                messages = [
                    {"role": "system", "content": f"You are an experienced HR of {company_name}. Generate one professional HR question for a {role} with {experience} years of experience."},
                    {"role": "user", "content": f"{prompt_examples}Generate one new HR question for the {role} position."}
                ]
                output = self.generate_response(messages)
                return {"success": True, "question": output[0]['generated_text']}
            else:
                # Feedback generation
                if not previous_questions.strip() or not previous_answer.strip():
                    return {"success": False, "error": "Previous questions and answers are required for feedback"}

                examples_df = filtered_hr_df.sample(min(len(filtered_hr_df), 3), random_state=random.randint(0, 1000)) if not filtered_hr_df.empty else pd.DataFrame()

                feedback_examples = "Here are examples of HR evaluation:\n"
                for i, row_data in examples_df.iterrows():
                    if 'question' in row_data and pd.notna(row_data['question']):
                        feedback_examples += f"Example {i+1}: {row_data['question']}\n"

                messages = [
                    {"role": "system", "content": f"You are an experienced HR of {company_name}. Analyze the candidate's answer and provide constructive feedback with a score out of 100."},
                    {"role": "user", "content": f"QUESTION: {previous_questions}\n\nANSWER: {previous_answer}\n\n{feedback_examples}\n\nProvide detailed feedback and score (format: 'Score: XX')."}
                ]
                feedback_output = self.generate_response(messages)
                generated_text = feedback_output[0]['generated_text']
                score = self.extract_score(generated_text)

                # Follow-up logic with validation
                if max_followup > 0 and score < 80:
                    followup_messages = [
                        {"role": "system", "content": f"Generate a targeted follow-up HR question based on the candidate's previous response."},
                        {"role": "user", "content": f"Previous Q: {previous_questions}\nAnswer: {previous_answer}\nScore: {score}\n\nGenerate one follow-up question."}
                    ]
                    followup_output = self.generate_response(followup_messages)
                    return {
                        "success": True, 
                        "feedback": generated_text, 
                        "score": score,
                        "followup_question": followup_output[0]['generated_text'], 
                        "max_followup": max_followup - 1
                    }
                else:
                    return {
                        "success": True, 
                        "feedback": generated_text, 
                        "score": score,
                        "new_question": "Continue with next HR question or end interview."
                    }
        except Exception as e:
            return {"success": False, "error": f"HR interview error: {str(e)}"}

    def behavioral_interview(self, data):
        # Validate inputs
        is_valid, message = self.validate_inputs(data)
        if not is_valid:
            return {"success": False, "error": message}

        company_name = data.get('company_Name', 'Company')
        role = data.get('role', 'Role')
        start = data.get('start', 0)
        max_followup = max(0, data.get('maxfollowup', 3))
        experience = data.get('exprences', 0)
        previous_questions = data.get('previous_questions', '')
        previous_answer = data.get('previous_answer', '')
        
        try:
            # Filter behavioral CSV data
            filtered_behavioral_df = self.df_behavioral.copy()
            if not filtered_behavioral_df.empty:
                if 'role' in filtered_behavioral_df.columns and role:
                    filtered_behavioral_df = filtered_behavioral_df[filtered_behavioral_df['role'] == role]
                if 'experience' in filtered_behavioral_df.columns and experience:
                    filtered_behavioral_df = filtered_behavioral_df[filtered_behavioral_df['experience'] == experience]
            
            # Fallback to full dataset if filtered is empty
            if filtered_behavioral_df.empty:
                filtered_behavioral_df = self.df_behavioral.copy()
            
            if start == 0:
                # Initial question generation
                examples_df = filtered_behavioral_df.sample(min(len(filtered_behavioral_df), 5), random_state=42) if not filtered_behavioral_df.empty else pd.DataFrame()

                prompt_examples = "Here are examples of behavioral questions:\n"
                for i, row_data in examples_df.iterrows():
                    if 'question' in row_data and pd.notna(row_data['question']):
                        prompt_examples += f"Example {i+1}: {row_data['question']}\n"

                # Fallback if no examples
                if not prompt_examples.strip() or prompt_examples == "Here are examples of behavioral questions:\n":
                    prompt_examples = "Generate a behavioral interview question using STAR method.\n"

                messages = [
                    {"role": "system", "content": f"You are an experienced behavioral interviewer for {company_name}. Generate one behavioral question for a {role}."},
                    {"role": "user", "content": f"{prompt_examples}Generate one new behavioral question."}
                ]
                output = self.generate_response(messages)
                return {"success": True, "question": output[0]['generated_text']}
            else:
                # Feedback generation
                if not previous_questions.strip() or not previous_answer.strip():
                    return {"success": False, "error": "Previous questions and answers are required for feedback"}

                messages = [
                    {"role": "system", "content": f"You are an experienced behavioral interviewer. Analyze using STAR method and provide feedback with score out of 100."},
                    {"role": "user", "content": f"QUESTION: {previous_questions}\n\nANSWER: {previous_answer}\n\nAnalyze using STAR method and provide feedback with score (format: 'Score: XX')."}
                ]
                feedback_output = self.generate_response(messages)
                generated_text = feedback_output[0]['generated_text']
                score = self.extract_score(generated_text)

                # Follow-up logic
                if max_followup > 0 and score < 80:
                    followup_messages = [
                        {"role": "system", "content": f"Generate a targeted follow-up behavioral question."},
                        {"role": "user", "content": f"Previous Q: {previous_questions}\nAnswer: {previous_answer}\nScore: {score}\n\nGenerate one follow-up behavioral question."}
                    ]
                    followup_output = self.generate_response(followup_messages)
                    return {
                        "success": True, 
                        "feedback": generated_text, 
                        "score": score,
                        "followup_question": followup_output[0]['generated_text'], 
                        "max_followup": max_followup - 1
                    }
                else:
                    return {
                        "success": True, 
                        "feedback": generated_text, 
                        "score": score,
                        "new_question": "Continue with next behavioral question or end interview."
                    }
        except Exception as e:
            return {"success": False, "error": f"Behavioral interview error: {str(e)}"}

    def technical_interview(self, data):
        # Validate inputs
        is_valid, message = self.validate_inputs(data)
        if not is_valid:
            return {"success": False, "error": message}

        try:
            company_name = data.get('company_Name', 'Company')
            role = data.get('role', 'Role')
            start = data.get('start', 0)
            max_followup = max(0, data.get('maxfollowup', 3))
            experience = data.get('exprences', 0)
            previous_questions = data.get('previous_questions', '')
            previous_answer = data.get('previous_answer', '')
            
            # Use technical CSV data with fallback
            source_df = self.df_technical.copy()
            if not source_df.empty and role:
                filtered_df = source_df[source_df.get('role', '') == role]
                source_df = filtered_df if not filtered_df.empty else source_df

            if start == 0:
                # Generate initial question
                examples_df = source_df.sample(min(len(source_df), 3), random_state=42) if not source_df.empty else pd.DataFrame()
                
                prompt_examples = "Generate a technical question.\n"
                for i, row_data in examples_df.iterrows():
                    if 'question' in row_data and pd.notna(row_data['question']):
                        prompt_examples += f"Example: {row_data['question']}\n"

                messages = [
                    {"role": "system", "content": f"You are a technical interviewer for {company_name}. Generate one technical question for {role}."},
                    {"role": "user", "content": f"{prompt_examples}Generate one new technical question."}
                ]
                output = self.generate_response(messages)
                return {"success": True, "question": output[0]['generated_text']}
            else:
                # Generate feedback
                if not previous_questions.strip() or not previous_answer.strip():
                    return {"success": False, "error": "Previous questions and answers required"}

                messages = [
                    {"role": "system", "content": "Analyze the technical answer and provide feedback with score out of 100."},
                    {"role": "user", "content": f"QUESTION: {previous_questions}\n\nANSWER: {previous_answer}\n\nProvide technical feedback and score (format: 'Score: XX')."}
                ]
                feedback_output = self.generate_response(messages)
                generated_text = feedback_output[0]['generated_text']
                score = self.extract_score(generated_text)

                return {"success": True, "feedback": generated_text, "score": score}
        except Exception as e:
            return {"success": False, "error": f"Technical interview error: {str(e)}"}

    def coding_interview(self, data):
        # Validate inputs
        is_valid, message = self.validate_inputs(data)
        if not is_valid:
            return {"success": False, "error": message}

        try:
            company_name = data.get('company_Name', 'Company')
            role = data.get('role', 'Role')
            start = data.get('start', 0)
            previous_questions = data.get('previous_questions', '')
            previous_answer = data.get('previous_answer', '')

            if start == 0:
                # Generate coding question
                messages = [
                    {"role": "system", "content": f"You are a technical interviewer for {company_name}. Generate one coding question for {role}."},
                    {"role": "user", "content": "Generate a coding problem with clear requirements and expected solution approach."}
                ]
                output = self.generate_response(messages)
                return {"success": True, "question": output[0]['generated_text']}
            else:
                # Generate feedback
                if not previous_questions.strip() or not previous_answer.strip():
                    return {"success": False, "error": "Previous questions and answers required"}

                messages = [
                    {"role": "system", "content": "Analyze the coding solution and provide feedback with score out of 100."},
                    {"role": "user", "content": f"PROBLEM: {previous_questions}\n\nSOLUTION: {previous_answer}\n\nAnalyze code quality, logic, efficiency and provide score (format: 'Score: XX')."}
                ]
                feedback_output = self.generate_response(messages)
                generated_text = feedback_output[0]['generated_text']
                score = self.extract_score(generated_text)

                return {"success": True, "feedback": generated_text, "score": score}
        except Exception as e:
            return {"success": False, "error": f"Coding interview error: {str(e)}"}