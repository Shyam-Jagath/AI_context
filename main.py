from classifier import Classifier
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq 
from langchain_anthropic import ChatAnthropic


load_dotenv()



prompt = input()

def Result(prompt):
    domain = Classifier(prompt)
    return domain

def Model(prompt):
    domain = Result(prompt)

    if(domain == "medical"):

        llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro-exp-03-25")
        response = llm.invoke(prompt)
        return response.content

    elif(domain == "engineering"):

        llm = ChatGroq(model="llama-3.3-70b-versatile")
        response = llm.invoke(prompt)
        return response.content
    
    elif(domain == "finance"):

        llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro-exp-03-25")
        response = llm.invoke(prompt)
        return response.content
    
    elif(domain == "legal"):

        llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro-exp-03-25")
        response = llm.invoke(prompt)
        return response.content
    
    elif(domain == "technology"):

        llm = ChatGroq(model="llama-3.3-70b-versatile")
        response = llm.invoke(prompt)
        return response.content
    
    elif(domain == "business"):

        llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro-exp-03-25")
        response = llm.invoke(prompt)
        return response.content
            
    elif(domain == "coding"):

        llm = ChatGroq(model="llama-3.3-70b-versatile")
        response = llm.invoke(prompt)
        return response.content
    
    elif(domain == "entertainment"):

        llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro-exp-03-25")
        response = llm.invoke(prompt)
        return response.content
    
    elif(domain == "environment sciences"):

        llm = ChatGroq(model="llama-3.3-70b-versatile")
        response = llm.invoke(prompt)
        return response.content
    
    else:

        llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro-exp-03-25")
        response = llm.invoke(prompt)
        return response.content
        
Final=Model(prompt)
print(Final)
    