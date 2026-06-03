def no_battery_policy(state: dict) -> int:
    return 0

def self_consumption_policy(state: dict) -> int:
    if state["surplus"] > 0:
        return 1
    if state["demand"] > 0:
        return -1
    return 0


def off_peak_charging_policy(state: dict,
                            peak_rate_threshold=0.4, 
                            off_peak_rate_threshold=0.22) -> int:
    
    rate = state["rate"]
    soc  = state["battery_soc"]
    cap  = state["battery_capacity"]
    if rate >= peak_rate_threshold and soc > 0:
        return -1
    if rate <= off_peak_rate_threshold and soc < cap:
        return 1
    return 0



