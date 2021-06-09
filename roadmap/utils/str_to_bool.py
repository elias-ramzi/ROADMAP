def str_to_bool(condition):
    if isinstance(condition, str):
        if condition.lower() == 'true':
            condition = True
        if condition.lower() == 'false':
            condition = False

    return condition
