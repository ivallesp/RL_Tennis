def flatten(l):
    """
    Flatten a list of lists into a plain list
    :param l: list to flatten (list)
    :return: list flattened (list)
    """
    return [item for sublist in l for item in sublist]
