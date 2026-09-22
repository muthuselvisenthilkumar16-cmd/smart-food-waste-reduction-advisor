import os
import json
import re

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()


# ============================================================
# CONNECT LANGCHAIN TO GROQ
# ============================================================

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)


# ============================================================
# FOOD INFORMATION EXTRACTION PROMPT
# ============================================================

extraction_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are an intelligent food waste analysis assistant.

The user may enter information about ANY type of food.

Examples include:
- rice
- curry
- vegetables
- fruits
- bread
- pasta
- noodles
- chicken
- fish
- meat
- milk
- dairy products
- snacks
- cooked food
- desserts
- or any other food

Your job is to identify the food and extract the numerical information.

Extract exactly these four values:

1. food_type
   - The actual food mentioned by the user.
   - Do NOT automatically assume rice.

2. leftover_quantity
   - Percentage of food remaining.
   - Must be between 0 and 100.

3. consumption_demand
   - Expected percentage of the remaining food that is likely to be consumed.
   - Must be between 0 and 100.

4. storage_time
   - Number of days the food has been stored.
   - Must be between 0 and 7.

Return ONLY valid JSON.

Use exactly this format:

{{
    "food_type": "food name",
    "leftover_quantity": 0,
    "consumption_demand": 0,
    "storage_time": 0
}}

Important rules:

- Identify the actual food mentioned.
- Never automatically change the food to rice.
- If the user gives a percentage, use that percentage.
- If the user gives a number of days, use that number.
- If a numerical value is not mentioned, make a reasonable estimate.
- Do not include explanations outside the JSON.
"""
    ),
    (
        "human",
        "{user_input}"
    )
])


# ============================================================
# EXTRACT FOOD INFORMATION
# ============================================================

def extract_food_information(user_input):

    chain = extraction_prompt | llm

    response = chain.invoke({
        "user_input": user_input
    })

    content = response.content

    # Handle possible Markdown code fences from the model
    content = content.replace("```json", "")
    content = content.replace("```", "")
    content = content.strip()

    # Extract the JSON object if extra text accidentally appears
    match = re.search(
        r"\{.*\}",
        content,
        re.DOTALL
    )

    if match:
        content = match.group(0)

    data = json.loads(content)

    # --------------------------------------------------------
    # Validate required fields
    # --------------------------------------------------------

    required_fields = [
        "food_type",
        "leftover_quantity",
        "consumption_demand",
        "storage_time"
    ]

    for field in required_fields:

        if field not in data:
            raise ValueError(
                f"Missing field from AI response: {field}"
            )

    # --------------------------------------------------------
    # Clean values
    # --------------------------------------------------------

    food_type = str(
        data["food_type"]
    ).strip()

    leftover_quantity = float(
        data["leftover_quantity"]
    )

    consumption_demand = float(
        data["consumption_demand"]
    )

    storage_time = float(
        data["storage_time"]
    )

    # --------------------------------------------------------
    # Keep values inside valid ranges
    # --------------------------------------------------------

    leftover_quantity = max(
        0,
        min(100, leftover_quantity)
    )

    consumption_demand = max(
        0,
        min(100, consumption_demand)
    )

    storage_time = max(
        0,
        min(7, storage_time)
    )

    return {
        "food_type": food_type,
        "leftover_quantity": leftover_quantity,
        "consumption_demand": consumption_demand,
        "storage_time": storage_time
    }


# ============================================================
# FOOD WASTE ADVICE PROMPT
# ============================================================

advice_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are a food waste reduction advisor.

Give a short, practical and easy-to-understand recommendation.

Use the food type and fuzzy risk assessment provided.

Your response should contain:

1. Why the food received this risk level.
2. What the user should do with the leftover food.
3. How the user can reduce similar food waste in the future.

Important:

- Do not invent facts.
- Do not claim that the food is definitely safe or unsafe.
- If storage time may raise a food-safety concern, advise the user to follow official food-safety guidance and use their judgment.
- Keep the answer concise.
- Do not return JSON.
- Do not return HTML.
- Do not return Python code.
- Write normal human-readable text.
"""
    ),
    (
        "human",
        """
Food type: {food_type}

Leftover quantity: {leftover_quantity}%

Expected consumption demand: {consumption_demand}%

Storage time: {storage_time} days

Fuzzy waste risk score: {risk_score}/100

Risk level: {risk_level}

Give a personalized food-waste recommendation.
"""
    )
])


# ============================================================
# GENERATE AI FOOD WASTE ADVICE
# ============================================================

def generate_advice(
    food_type,
    leftover_quantity,
    consumption_demand,
    storage_time,
    risk_score,
    risk_level
):

    chain = advice_prompt | llm

    response = chain.invoke({
        "food_type": food_type,
        "leftover_quantity": leftover_quantity,
        "consumption_demand": consumption_demand,
        "storage_time": storage_time,
        "risk_score": risk_score,
        "risk_level": risk_level
    })

    return response.content.strip()