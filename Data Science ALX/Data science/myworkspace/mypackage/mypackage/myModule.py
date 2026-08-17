def top_n (items, n):
    """Return the top n items in an array, in dec order.
    
    Args:
        items(array): A list or array like object.
        n (int): The number of items to return.

    Return:
        An array containing the top n items in desc order.

    Egs:
        >>> top_n([8,3,2,7,4], 3)
        [8, 7, 3]
    """
    for i in range(n): # Keep sorting until we have the top n items
        for j in range(len(items)-1-i):

            if items[j] > items[j+1]: # if this item is bigger than the next item..
                items[j], items[j+1] = items[j], items[j+1] # swap them!

    # get the last two items
    top_n -  items[-n:]

    #return in descending order
    return top_n[::-1]