# 🤖 AI MockRound - Interview Platform

Advanced AI-powered interview platform with real-time feedback and scoring.

## 🚀 Features

- **HR Round**: Cultural fit and motivation questions
- **Technical Round**: Role-specific technical questions  
- **Behavioral Round**: STAR method behavioral questions
- **Coding Round**: Programming challenges with feedback

## 📡 API Endpoints

```
POST /api/hr
POST /api/technical
POST /api/behavioral
POST /api/coding
```

## 🔧 Request Format

```json
{
  "company_Name": "TechCorp",
  "role": "Software Engineer",
  "exprences": 3,
  "package": 100000,
  "start": 0,
  "previous_questions": "",
  "previous_answer": "",
  "maxfollowup": 3
}
```

## 🌐 Live Demo

**API URL**: `https://aimockround.onrender.com`

## 🛠️ Local Setup

```bash
pip install -r requirements.txt
python app.py
```

## 📊 Response Format

**Question Generation (start=0):**
```json
{
  "success": true,
  "question": "Generated interview question"
}
```

**Feedback Generation (start=1):**
```json
{
  "success": true,
  "feedback": "Detailed feedback",
  "score": 85
}
```

## 🎯 Supported Roles

- Software Engineer
- Frontend Engineer
- Backend Engineer
- DevOps Engineer
- Mobile Developer
- Full Stack Engineer
- Data Engineer
- ML Engineer
- QA Engineer
- Security Engineer
- Product Manager

## 🏢 Supported Companies

- Google
- Microsoft
- Amazon
- Meta
- Apple
- Netflix
- Uber
- Airbnb
- Spotify
- OpenAI
- CrowdStrike

## 🚀 Deployment

Deployed on Render for fast, reliable API responses.

## 📁 Project Structure

```
mockround.ai/
├── app.py                              # Main Flask application
├── requirements.txt                    # Dependencies
├── interview_data.csv                  # Company-role mappings
├── technical_interview_questions.csv   # Technical questions dataset
├── hr_round_questions.csv             # HR questions dataset
├── behavioral_round_questions.csv     # Behavioral questions dataset
└── README.md                          # Project documentation
```

## 🤖 AI Model

- **Model**: Microsoft Phi-3.5-mini-instruct
- **Features**: Role-specific question generation
- **Scoring**: Intelligent feedback with 0-100 scoring
- **Follow-ups**: Adaptive questioning based on performance

## 💡 College Startup

Built as a college-level startup project for revolutionizing interview preparation and assessment.

---
**Built with ❤️ for better interviews**