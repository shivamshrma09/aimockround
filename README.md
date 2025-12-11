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
  "package": 100,
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
pip install -r requirements_render.txt
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

## 🚀 Deployment

Deployed on Render for fast, reliable API responses.

---
**Built with ❤️ for better interviews**