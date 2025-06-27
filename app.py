import streamlit as st
import os
import pandas as pd
from dotenv import load_dotenv
from langgraph_flow import EVROUTE_APP
from langchain.vectorstores import FAISS
import folium
from streamlit_folium import folium_static

# Load environment variables
load_dotenv()
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


# Streamlit setup
st.set_page_config(page_title="EVRoute Planner ", layout="centered")
st.title("🔋 EVRoutePlanner: Smart EV Route Planner")

# Inputs
origin = st.text_input("Origin (lat, lon)", placeholder="e.g. 12.9716,77.5946")
destination = st.text_input("Destination (lat, lon)", placeholder="e.g. 13.0827,80.2707")
ev_model = st.text_input("Your EV Model", placeholder="e.g. Tesla Model 3")
charge = st.slider("Current Charge (kWh)", 0.0, 100.0, 60.0)

# Plan route button
if st.button("Plan Route"):

    # Validate input format
    try:
        origin_coords = tuple(map(float, origin.split(",")))
        destination_coords = tuple(map(float, destination.split(",")))
    except ValueError:
        st.error("⚠️ Please enter both latitude and longitude for origin and destination.")
        st.stop()

    input_data = {
        "origin": origin_coords,
        "destination": destination_coords,
        "ev_model": ev_model,
        "current_charge_kWh": charge
    }

    try:
        result = EVROUTE_APP.invoke(input_data)

        st.subheader("Route Summary")
        st.write("**Energy Needed:**", result.get("energy_needed_kWh", "N/A"), "kWh")
        st.write("**Charging Stop:**", result.get("charging_stop", "N/A"))
        st.write("**ETA:**", result.get("final_eta_mins", "N/A"), "minutes")
        st.success(result.get("user_explanation", "No explanation available."))

        # Plot on Map
        coords = result.get("route_info", {}).get("coordinates", [])
        if coords:
            df = pd.DataFrame(coords, columns=["lon", "lat"])
            m = folium.Map(location=[df["lat"].mean(), df["lon"].mean()], zoom_start=6)
            folium.PolyLine(locations=df[["lat", "lon"]].values.tolist(), color="blue", weight=4).add_to(m)
            folium_static(m)
        else:
            st.warning("No route coordinates available to display on the map.")

    except Exception as e:
        st.error(f"Something went wrong while invoking the route planner: {e}")
