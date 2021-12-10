def format_time(seconds):
    seconds = int(seconds)
    minutes = seconds // 60
    hours = minutes // 60
    minutes = minutes % 60
    rseconds = seconds % 60
    return f"{hours}h{minutes}m{rseconds}s"
