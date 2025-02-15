from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate

def create_pipeline():
    # Load model and tokenizer
    model_name = "gpt2-medium"  # Using medium model for better accuracy
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Configure tokenizer
    tokenizer.pad_token = tokenizer.eos_token
    
    # Create generation pipeline with carefully tuned parameters
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=30,        # Shorter response for more focus
        temperature=0.1,          # Very low temperature for more deterministic output
        top_k=50,                # Limit vocabulary choices
        top_p=0.92,              # Nucleus sampling
        repetition_penalty=1.2,   # Prevent repetition
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        device="cpu"
    )
    
    return HuggingFacePipeline(pipeline=pipe)

def get_answer(llm, question):
    # Carefully crafted prompt template to encourage factual responses
    template = """Please provide a short, accurate, factual answer to this question.
Question: {question}
Factual answer:"""
    
    prompt = PromptTemplate.from_template(template)
    formatted_prompt = prompt.format(question=question)
    
    # Get response and clean it
    try:
        response = llm.invoke(formatted_prompt)
        
        # Clean up response
        answer = response.split("Factual answer:")[-1].strip()
        # Take only the first sentence
        answer = answer.split('.')[0].strip() + '.'
        
        return answer
    except Exception as e:
        return f"Error: {str(e)}"

def main():
    print("Loading GPT-2 model (this may take a moment)...")
    llm = create_pipeline()
    
    while True:
        question = input("\nEnter your question (or 'quit' to exit): ")
        if question.lower() == 'quit':
            break
            
        print("\nThinking...")
        answer = get_answer(llm, question)
        print(f"Answer: {answer}")

if __name__ == "__main__":
    main()