from langchain_community.llms import HuggingFaceEndpoint
from langchain.chains import LLMChain
from langchain.prompts.prompt import PromptTemplate
from langchain.vectorstores import FAISS
from utils.combined import make_combined_store
from langchain.schema import Document
import os

llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",  # ✅ Hosted
    task="text-generation",
    temperature=0.4,
    max_new_tokens=150,
    huggingfacehub_api_token=os.getenv("HF_TOKEN")
)

prompt = PromptTemplate(
    input_variables=["summary"],
    template="Explain this EV route in plain English: {summary}"
)

chain = LLMChain(llm=llm, prompt=prompt)

def advise_user(input_data):
    summary = f"{input_data['route_info']['distance_km']}km, {input_data['energy_needed_kWh']}kWh, charging stop: {input_data['charging_stop']}, ETA: {input_data['final_eta_mins']}min"
    explanation = chain.run(summary=summary)
    input_data["user_explanation"] = explanation
    doc = make_combined_store()
    doc.add_documents([Document(page_content=summary, metadata={"origin": input_data['origin'], "destination": input_data['destination']})])
    return input_data
