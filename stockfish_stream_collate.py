import torch


def stockfish_stream_collate(batch):
    us_indices = []
    them_indices = []
    policy_indices = []
    policy_probs = []
    values = []

    for us, them, move_indices, move_probs, value in batch:
        us_indices.append(us)
        them_indices.append(them)
                                                                          
                                                                           
                                                                           
                                                                            
        policy_indices.append(torch.tensor(move_indices, dtype=torch.long))
        policy_probs.append(torch.tensor(move_probs, dtype=torch.float32))
        values.append(torch.tensor(value, dtype=torch.float32))

    return (
        us_indices,
        them_indices,
        policy_indices,
        policy_probs,
        torch.stack(values).unsqueeze(1)
    )