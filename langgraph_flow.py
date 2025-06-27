from langgraph.graph.state import StateGraph
from langgraph.constants import END
from agents.route_planner import plan_route
from agents.energy_predictor import estimate_energy
from agents.charging_advisor import advise_charging
from agents.traffic_adjuster import adjust_for_traffic
from agents.user_advisor import advise_user
from typing import TypedDict, Tuple, Optional

class EVRouteState(TypedDict):
    origin: Tuple[float, float]
    destination: Tuple[float, float]
    ev_model: str
    current_charge_kWh: float
    route_info: Optional[dict]
    energy_needed_kWh: Optional[float]
    charging_stop: Optional[dict]
    final_eta_mins: Optional[float]
    user_explanation: Optional[str]


graph = StateGraph(EVRouteState)
graph.add_node("RoutePlanner", plan_route)
graph.add_node("EnergyPredictor", estimate_energy)
graph.add_node("ChargingAdvisor", advise_charging)
graph.add_node("TrafficAdjuster", adjust_for_traffic)
graph.add_node("UserAdvisor", advise_user)

graph.set_entry_point("RoutePlanner")
graph.add_edge("RoutePlanner", "EnergyPredictor")
graph.add_edge("EnergyPredictor", "ChargingAdvisor")
graph.add_edge("ChargingAdvisor", "TrafficAdjuster")
graph.add_edge("TrafficAdjuster", "UserAdvisor")
graph.add_edge("UserAdvisor", END)

EVROUTE_APP = graph.compile()