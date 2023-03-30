def hadamard_op(u, v):
    """Calculates hadamard edge representation between two node vectors."""
    return u * v


def calculate_edge_embedding(keyed_model, edges, op):
    """Calculates edge embedding."""
    representations = []

    for u, v in edges:
        e_u = keyed_model.get_vector(str(u))
        e_v = keyed_model.get_vector(str(v))

        representations.append(op(e_u, e_v))

    return representations
