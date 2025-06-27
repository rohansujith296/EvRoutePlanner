def adjust_for_traffic(input_data):
    time = input_data["route_info"]["base_route_time_mins"]
    input_data["final_eta_mins"] = round(time * 1.1, 2)  # 10% delay
    return input_data
