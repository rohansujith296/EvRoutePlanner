import requests
import os
from utils.combined import make_combined_store
from langchain_community.llms import HuggingFaceEndpoint
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def get_nearby_stations(lat, lon):
    url = f"https://api.openchargemap.io/v3/poi/?output=json&latitude={lat}&longitude={lon}&distance=25&distanceunit=KM&maxresults=1"
    headers = {"X-API-Key": os.getenv("OPENCHARGEMAP_API_KEY")}
    return requests.get(url, headers=headers).json()

def advise_charging(input_data):
    if input_data["current_charge_kWh"] < input_data["energy_needed_kWh"]:
        lat, lon = input_data["origin"]

        # Retrieve documents relevant to charging
        query = f"charging stations for {input_data['ev_model']} near lat {lat}, lon {lon}"
        doc = make_combined_store()
        res = doc.similarity_search(query, k=3)
        knowledge = "\n".join([d.page_content for d in res])

        prompt = PromptTemplate(
            input_variables=["model", "lat", "lon", "docs"],
            template="""
You are a charging advisor. Based on the EV model: {model}, and nearby coordinates: ({lat}, {lon}),
and the following documents:

{docs}

Suggest the most suitable charging station name and estimated charging time (in minutes). Return JSON with 'name', 'latitude', 'longitude', and 'charging_time_mins'.
"""
        )

        llm = HuggingFaceEndpoint(
            repo_id="HuggingFaceH4/zephyr-7b-beta",  # ✅ Hosted
            task="text-generation",
            temperature=0.4,
            max_new_tokens=150,
            huggingfacehub_api_token=os.getenv("HF_TOKEN")
        )

        chain = LLMChain(llm=llm, prompt=prompt)
        llm_result = chain.run(model=input_data['ev_model'], lat=str(lat), lon=str(lon), docs=knowledge)

        try:
            import json
            suggestion = json.loads(llm_result.strip())
            input_data["charging_stop"] = {
                "name": suggestion.get("name", "LLM Charging Point"),
                "location": (suggestion.get("latitude", lat), suggestion.get("longitude", lon)),
                "charging_time_mins": suggestion.get("charging_time_mins", 25)
            }
        except:
            # fallback to OpenChargeMap
            stations = get_nearby_stations(lat, lon)
            if stations:
                s = stations[0]["AddressInfo"]
                input_data["charging_stop"] = {
                    "name": s["Title"],
                    "location": (s["Latitude"], s["Longitude"]),
                    "charging_time_mins": 25
                }
    else:
        input_data["charging_stop"] = None
    return input_data