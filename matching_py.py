import mygrad as mg
import numpy as np

def cos_distance(descriptor_a, descriptor_b):
    """Takes in two arrays of descriptors and returns their cosine distance."""
    return 1 - (descriptor_a @ descriptor_b) / (np.linalg.norm(descriptor_a) * np.linalg.norm(descriptor_b))

def matching(database, new_descriptor, threshold):
    """Compares new face descriptor to database and returns name with the lowest distance - if not under threshold, returns unknown
    Parameters:
        database -- type: Dict
            database of name : Profile
        new_descriptor -- type: np.array
            the average descriptor from input image
        threshold -- type: int
            max distance between input image and item in database to be considered a match
    Output:
        String
            name or Unknown
    """
    distances = []
    for name in database:
        #find the cosine distance between each avg and the new_descriptor, append to [distances]
        distances.append(cos_distance(np.mean(database[name]), new_descriptor))

    #find the lowest distance value
    lowest_distance = min(distances)
    #if below threshold, return name
    if lowest_distance < threshold:
        return name
    #else return unknown
    return "Unknown :("
    
