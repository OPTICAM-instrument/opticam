from typing import Any, List


def get_batches(input: List[Any]) -> List[List[Any]]:
    """
    Given an input list, split it into up to 100 evenly(ish)-sized batches.
    
    Parameters
    ----------
    input : List[Any]
        The input list.
    
    Returns
    -------
    List[List[Any]]
        The batches.
    """
    
    L = len(input)
    
    batch_size = get_batch_size(L)
    
    batches = []
    
    for i in range(0, L, batch_size):
        batches.append(input[i:i+batch_size])
    
    return batches


def get_batch_size(L: int) -> int:
    """
    Compute the batch size for a given input length.
    
    Parameters
    ----------
    L : int
        The length of the input.
    
    Returns
    -------
    int
        The batch size.
    """
    
    return max(1, L // 100)