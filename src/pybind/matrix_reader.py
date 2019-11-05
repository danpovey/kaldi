import kaldi_pybind as kp

class SequentialVectorReader(kp.SequentialBaseFloatMatrixReader_Vector):
    def __init__(self, rspecifier = None):
        if not rspecifier:
            kp.SequentialBaseFloatMatrixReader_Vector.__init__(self)      
        else:
            kp.SequentialBaseFloatMatrixReader_Vector.__init__(self, rspecifier)      

    def __iter__(self):
        return self

    def __next__(self):
        if self.Done():
            raise StopIteration()
        else:
            key, value = self.Key(), self.Value()
            self.Next()
            return key, value

class SequentialMatrixReader(kp.SequentialBaseFloatMatrixReader_Matrix):
    def __init__(self, rspecifier = None):
        if not rspecifier:
            kp.SequentialBaseFloatMatrixReader_Matrix.__init__(self)      
        else:
            kp.SequentialBaseFloatMatrixReader_Matrix.__init__(self, rspecifier)      

    def __iter__(self):
        return self

    def __next__(self):
        if self.Done():
            raise StopIteration()
        else:
            key, value = self.Key(), self.Value()
            self.Next()
            return key, value

class ReaderIterator():
    def __init__(self, reader_class):
        self.iterator = reader_class

    def __iter__(self):
        return self.iterator.__iter__()

    def __next__(self):
        return self.iterator.__next__()
