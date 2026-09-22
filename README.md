# 🍽️ Smart Food Waste Reduction Advisor

## 📌 Project Description

Smart Food Waste Reduction Advisor is an AI and Fuzzy Logic based web application designed to help users reduce unnecessary food waste.

The application allows users to describe their food situation using natural language. A Large Language Model (LLM), integrated through LangChain, extracts important information from the user's description. This structured information is then processed using a Fuzzy Inference System to calculate a food waste risk score.

The application provides a final risk level and an understandable recommendation to help users make better food-management decisions.

---

## 🎯 Objectives

- Understand food-related information provided in natural language.
- Use LangChain and an LLM to extract structured information.
- Apply fuzzy logic to evaluate food waste risk.
- Use membership functions and fuzzy rules for inference.
- Defuzzify the fuzzy output into a numerical risk score.
- Provide useful recommendations through a simple web interface.
- Demonstrate the integration of AI and Fuzzy Logic in a real-world application.

---

## ✨ Main Features

- Natural-language food input
- AI-powered food information extraction
- LangChain-based LLM processing
- Fuzzy inference system
- Membership functions
- Fuzzy rule evaluation
- Defuzzification
- Food waste risk score
- Personalized recommendation
- Interactive Streamlit interface

---

## 🛠️ Technologies Used

- Python
- Streamlit
- LangChain
- Large Language Model (LLM)
- scikit-fuzzy
- NumPy
- Pandas
- Matplotlib
- python-dotenv

---

## 🧠 System Workflow

The application follows this workflow:

User Input
↓
Streamlit Interface
↓
LangChain + LLM
↓
Food Information Extraction
↓
Fuzzification
↓
Fuzzy Rule Evaluation
↓
Fuzzy Inference
↓
Defuzzification
↓
Waste Risk Score
↓
AI Recommendation

---

## 🤖 AI / LangChain Component

The user provides a natural-language description of their food situation.

For example:

"I have 3 cups of cooked rice that has been refrigerated for two days. There are two people at home but we may eat outside tonight."

The LangChain/LLM component processes the text and extracts relevant information such as:

- Food type
- Quantity
- Food age
- Storage condition
- Expected consumption

The extracted information is then passed to the fuzzy logic system.

---

## 🧠 Fuzzy Logic Component

The application uses a genuine Fuzzy Inference System rather than simple if-else conditions.

The system includes:

### Input Variables

- Food age
- Quantity
- Consumption likelihood
- Storage condition

### Membership Functions

The input variables are represented using fuzzy membership functions such as:

- Low
- Medium
- High

or appropriate linguistic categories depending on the variable.

### Fuzzy Rules

Example:

IF food age is OLD
AND consumption likelihood is LOW
THEN waste risk is HIGH.

IF quantity is LARGE
AND consumption likelihood is LOW
THEN waste risk is HIGH.

IF food age is FRESH
AND consumption likelihood is HIGH
THEN waste risk is LOW.

### Defuzzification

The fuzzy output is converted into a numerical waste-risk score.

Example:

Waste Risk Score: 72/100

---

## 🌐 Running the Project Locally

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>