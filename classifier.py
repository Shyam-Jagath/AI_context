from langchain_groq import ChatGroq 
from langchain_core.messages import SystemMessage , HumanMessage , AIMessage
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic

load_dotenv() 

llm = ChatGroq(model="llama-3.3-70b-versatile") 


def Classifier(prompt):
    messages = [
        SystemMessage("""
You are a domain classifier. Your task is to analyze a user prompt and classify it into one of the following domains:

['medical', 'engineering', 'finance', 'legal', 'technology', 'business', 'coding', 'entertainment', 'environment sciences']

Instructions:
- Respond with ONLY one domain name from the list above.
- Use EXACTLY the same spelling as shown.
- Do NOT return anything else—no explanations, punctuation, or additional words or no special characters. 
- If the prompt does not fit any of these domains, return: general
- Output must be a single word or phrase exactly as listed above, in lowercase.

Only return the domain name. Nothing else.
"""),
        HumanMessage(prompt)
    ]
    
    result = llm.invoke(messages)
    domain = result.content.strip().lower()  # Clean whitespace + normalize casing

    # Validate against allowed domains
    valid_domains = [
        "medical", "engineering", "finance", "legal", "technology",
        "business", "coding", "entertainment", "environment sciences", "general"
    ]
    
    if domain in valid_domains:
        return domain
    else:
        print(f"[WARN] Unexpected classifier output: '{domain}'")
        return None
