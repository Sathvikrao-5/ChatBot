from langchain.chains import RetrievalQA
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import logging


def extract_string_after_substring(response):
    substring = "<|assistant|>"
    try:
        substring_index = response.find(substring)
        if substring_index != -1:
            string_after_substring = response[substring_index + len(substring):].strip()
            return string_after_substring
    except Exception as e:
        print(f"An error occurred: {e}")
    return None
    
class LocalLLM:

    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name,device_map="auto")

    def generate_response(self, messages, max_length=600, temperature=0.8, do_sample=True):

        prompt = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        )
        with torch.no_grad():
            outputs = self.model.generate(
                prompt.to(self.model.device),
                max_length=max_length,  
                temperature=temperature,
                max_new_tokens=1024,
                do_sample=do_sample,      
                num_beams=5,              
                early_stopping=True 
            )
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            final_response=extract_string_after_substring(response)
            return final_response


def create_retrieval_chain(vectorstore, model_name, SYSTEM_PROMPT, max_context_tokens=500):
    llm = LocalLLM(model_name)
    def truncate_context(context, max_tokens):
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)
        logger.info(f"Context size: {len(context.split())} tokens")

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokens = tokenizer.encode(context, truncation=True)
        if len(tokens) > max_tokens:
            truncated = tokenizer.decode(tokens[:max_tokens], skip_special_tokens=True)
            return truncated
        return context
    def query_chain(user_query):
        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={'k': 4}
        )

        docs = retriever.invoke(user_query)
        if not docs:
            return "I cannot find relevant information in my resources regarding your query. Please provide more specific details."
        context = "\n".join([doc.page_content for doc in docs])
        context = truncate_context(context, max_tokens=max_context_tokens)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "context", "content": context},
            {"role": "user", "content": user_query}
        ]
        return llm.generate_response(messages)

    return query_chain
