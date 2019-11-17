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

class RandomAccessVectorReader(kp.RandomAccessBaseFloatMatrixReader_Vector):
    def __init__(self, rspecifier = None):
        if not rspecifier:
            kp.RandomAccessBaseFloatMatrixReader_Vector.__init__(self)      
        else:
            kp.RandomAccessBaseFloatMatrixReader_Vector.__init__(self, rspecifier)      

    def __getitem__(self, key):
        if not self.HasKey(key):
            raise KeyError("{} does not exits.".format(key))
        else:
            return self.Value(key)

    def __contains__(self, key):
        return self.HasKey(key)

class RandomAccessMatrixReader(kp.RandomAccessBaseFloatMatrixReader_Matrix):
    def __init__(self, rspecifier = None):
        if not rspecifier:
            kp.RandomAccessBaseFloatMatrixReader_Matrix.__init__(self)      
        else:
            kp.RandomAccessBaseFloatMatrixReader_Matrix.__init__(self, rspecifier)      

    def __getitem__(self, key):
        if not self.HasKey(key):
            raise KeyError("{} does not exits.".format(key))
        else:
            return self.Value(key)

    def __contains__(self, key):
        return self.HasKey(key)

class ReaderDict():
    def __init__(self, reader_class):
        self.dict = reader_class

    def __getitem__(self, key):
        return self.dict.__getitem__(key)

    def __contains__(self, key):
        return self.dict.__contains__(key)
