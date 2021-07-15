import numpy as np

class Profile:
    def __init__(self) -> None:
        pass

    def __call__(self, name, descriptor_vector):
        return self.add_to_database(name, descriptor_vector)

    def add_to_database(self, name, descriptor_vector):
        """ Takes in user-inputted name and descriptor vector from FaceNet and adds to pickled database.

        Parameters
        ----------
        name : String
            User-inputted label for face associated with the box produced by FaceNet
        descriptor_vector : np.ndarray, shape=(1, 128)
            The descriptor vector for [face]

        Returns
        -------
        database : dict
            Database of name to descriptor vector for [name] produced by FaceNet
        """

        if name in database:
            database[name].append(descriptor_vector)
        else:
            database[name] = [descriptor_vector]

        return database
