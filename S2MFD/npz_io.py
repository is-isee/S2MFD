import numpy as np


class NpzIO:
    """np.savez による save/load を提供するミックスイン。

    np.savez はスカラー (int, float) を0次元 ndarray として保存するため、
    load 時に .item() で Python ネイティブ型に復元する。復元しないと
    numba カーネルや文字列フォーマットに渡した際に型が崩れる恐れがある。
    """

    def save(self, filename):
        """
        Save the object data to a file.

        Parameters
        ----------
        filename : str
           File name to save the data.
        """
        np.savez(filename, **self.__dict__)

    @classmethod
    def load(cls, filename):
        """
        Load the object data from a file.

        Parameters
        ----------
        filename : str
           File name to load the data.
        """
        data = np.load(filename)
        obj = cls.__new__(cls)
        obj.__dict__.update({
            key: (data[key].item() if data[key].ndim == 0 else data[key])
            for key in data.files
        })
        return obj
