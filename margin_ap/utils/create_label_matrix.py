def create_label_matrix(labels, other_labels=None):
    if other_labels is None:
        return (labels.view(-1, 1) == labels.t()).float()

    return (labels.view(-1, 1) == other_labels.t()).float()
