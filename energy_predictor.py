def estimate_energy(input_data):
    from utils.combined import make_combined_store
    from langchain.llms import HuggingFaceEndpoint
    from langchain.chains.llm import LLMChain
    from langchain.prompts.prompt import PromptTemplate
    from langchain.vectorstores import FAISS
    import os
    from langchain_community.embeddings import HuggingFaceEmbeddings

    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    ev_model = input_data["ev_model"]
    route = input_data["route_info"]
    distance_km = route['distance_km']
    elevation = 200  # Simulated elevation gain since real data is unavailable

    # Search for relevant documents
    query = f"energy consumption {ev_model} for {distance_km}km with elevation {elevation}m"
    doc = make_combined_store()
    res = doc.similarity_search(query, k=3)
    knowledge = "\n".join([d.page_content for d in res])

    # Setup LLM for energy estimation
    prompt = PromptTemplate(
        input_variables=["model", "distance", "elevation", "docs"],
        template="""
You are an EV energy estimator. Based on the EV model: {model}, distance to travel: {distance} km,
elevation gain: {elevation} m, and the following documents:

{docs}

Estimate the energy required in kWh for this trip. Respond with a single float.
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
    result = chain.run(model=ev_model, distance=str(distance_km), elevation=str(elevation), docs=knowledge)

    try:
        kWh = float(result.strip().split()[0])
    except:
        kWh = (distance_km * 160 + 0.1 * elevation) / 1000

    input_data["energy_needed_kWh"] = round(kWh, 2)
    return input_data