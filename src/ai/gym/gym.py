# encoding: UTF-8

from abc import ABCMeta, abstractmethod

########################################################################
class GymEnv(object):
    """Abstract class for an environment. Simplified OpenAI API.
    """

    def __init__(self):
        self.n_actions = None
        self.state_shape = None

    @abstractmethod
    def step(self, action):
        """Run one timestep of the environment's dynamics.
        Accepts an action and returns a tuple (observation, reward, done, info).

        Args:
            action (numpy.array): action array

        Returns:
            tuple:
                - observation (numpy.array): Agent's observation of the current environment.
                - reward (float) : Amount of reward returned after previous action.
                - done (bool): Whether the episode has ended, in which case further step() calls will return undefined results.
                - info (str): Contains auxiliary diagnostic information (helpful for debugging, and sometimes learning).
        """
        raise NotImplementedError()

    @abstractmethod
    def reset(self):
        """Reset the state of the environment and returns an initial observation.

        Returns:
            numpy.array: The initial observation of the space. Initial reward is assumed to be 0.
        """r
        raise NotImplementedError()

    @abstractmethod
    def render(self):
        """Render the environment.
        """
        raise NotImplementedError()

########################################################################
class IterableData(object):
    """Data generator from csv file.
    The csv file should no index columns.

    Args:
        filename (str): Filepath to a csv file.
        header (bool): True if the file has got a header, False otherwise
    """
    def __init__(self):
        """Initialisation function. The API (kwargs) should be defined in
        the function _generator.
        """
        super(IterableData, self).__init__()
        self._generator = self.generate()

    def __iter__(self):
        if not self._generator :
            raise NotImplementedError()
        return self

    @abstractmethod
    def generate(self):
        # dummyrow = [2.0]*5
        i=0
        while True:
            if i >10:
                raise StopIteration
            yield np.array([np.random.normal(scale=10)]*5, dtype=np.float)
            i+=1
 
    def __next__(self):
        if not self._generator :
            raise NotImplementedError()
        return next(self._generator)

    def _iterator_end(self):
        """Rewinds if end of data reached.
        """
        print("End of data reached, rewinding.")
        super(self.__class__, self).rewind()

    @abstractmethod
    def rewind(self):
        """For this generator, we want to rewind only when the end of the data is reached.
        """
        pass

########################################################################
class CSVStreamer(IterableData):
    """Data generator from csv file.
    The csv file should no index columns.

    Args:
        filename (str): Filepath to a csv file.
        header (bool): True if the file has got a header, False otherwise
    """
    def __init__(self, **kwargs):
        """Initialisation function. The API (kwargs) should be defined in
        the function _generator.
        """
        super(CSVStreamer, self).__init__()
        self._kwargs = kwargs

    def generate(self):
        filename = self._kwargs["filename"]
        header = False
        if 'header' in self._kwargs.keys() and isinstance(self._kwargs["header"], bool) :
            header = self._kwargs["header"]

        with open(filename, "r") as csvfile:
            reader = csv.reader(csvfile)
            if header:
                next(reader, None)
            for row in reader:
                assert len(row) % 2 == 0
                yield np.array(row, dtype=np.float)
