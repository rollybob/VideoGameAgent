def hud_delta(prev, cur):
    changes = {}
    for key in prev:
        if prev[key] != cur[key]:
            changes[key] = [prev[key], cur[key]]
    
    events = set()
    if cur["health"] < prev["health"]:
        events.add("damage")
    if prev["health"] > 0 and cur["health"] == 0:
        events.add("death")
    item_keys = {"rupees", "keys", "bombs", "arrows", "magic"}
    for key in item_keys:
        if key in prev and cur[key] > prev[key]:
            events.add("pickup")
            break
    return {
        "changes": changes,
        "events": sorted(events)
    }
