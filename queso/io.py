import pathlib
import warnings
import os
import copy
import datetime
import json
import string
import random
import yaml


def _default_data_path():
    config = pathlib.Path(__file__).parent.parent.joinpath("paths.yaml")
    # print(config)
    try:
        with open(config, "r") as fid:
            file = yaml.safe_load(fid)
            default_path = file["data_path"]
            # print(file)
    except:
        default_path = pathlib.Path(__file__).parent.parent.joinpath("data")
    return default_path


def _default_fig_path():
    config = pathlib.Path(__file__).parent.parent.joinpath("paths.yaml")
    # print(config)
    try:
        with open(config, "r") as fid:
            file = yaml.safe_load(fid)
            default_path = file["fig_path"]
    except:
        default_path = pathlib.Path(__file__).parent.parent.joinpath("data")
    return default_path


def current_time():
    """
    Returns current date and time in a consistent format, used for monitoring long-running measurements

    :return: current date and time
    :rtype: str
    """
    return datetime.datetime.now().strftime("%d/%m/%Y, %H:%M:%S")


class IO:
    r"""
    The IO class encapsulates all saving/loading features of data, figures, etc.
    This provides consistent filetypes, naming conventions, etc.

    Typical usage:
        io = IO(path=r"\path\to\data")
        io.load_txt(filename="filename.txt")

    or
        io = IO.create_new_save_folder(folder="subfolder", include_date=True, include_uuid=True)
        io.save_df(df, filename="dataframe.txt")
    """

    # default save path always points to `data/` no matter where this repository is located
    # default_path = pathlib.Path(__file__).parent.parent.joinpath("data")
    default_path = _default_data_path()

    def __init__(
        self,
        path=None,
        folder="",
        include_date=False,
        include_time=False,
        include_id=False,
        verbose=True,
    ):
        """

        :param path: The parent folder.
        :type path: str (of pathlib.Path object)
        :param folder: The main, descriptive folder name.
        :type folder: str
        :param include_date: If True, add the date to the front of the path. Otherwise, do not add the date
        :type include_date: bool
        :param include_time: If True, add the time to the front of the path. Otherwise, do not add the time
        :type include_time: bool
        :param include_id: If True, add a random string of characters to the end of the path. Otherwise, do not
        :type include_id: bool
        :param verbose: If True, will print out the path of each saved/loaded file.
        :type verbose: bool
        :return: A new IO class instance
        :rtype: IO
        """
        if path is None:
            path = self.default_path

        if type(path) is str:
            path = pathlib.Path(path)

        date = datetime.date.today().isoformat()
        time = datetime.datetime.now().strftime("%H-%M-%S")
        if not folder:  # if empty string
            warnings.warn(
                "No folder entered. Saving to a folder with a unique identifier"
            )
            include_data, include_id, verbose = True, True, True

        # build the full folder name with date, time, and uuid, if selected
        _str = ""
        if include_date:
            _str = _str + date + "_"
        if include_time:
            _str = _str + time + "_"

        _str = _str + folder

        if include_id:
            _str = (
                _str + "_" + "".join(random.choice(string.hexdigits) for _ in range(4))
            )

        self.path = path.joinpath(_str)
        self.verbose = verbose
        return

    def subpath(self, subfolder: str):
        cls = copy.deepcopy(self)
        cls.path = cls.path.joinpath(subfolder)
        return cls

    def save_json(self, variable, filename):
        """
        Save serialized python object into a json format, at filename

        :param variable: the object to save
        :type variable: serialized object
        :param filename: name of the file to which variable should be saved
        :type filename: str
        :return: the function returns nothing
        :rtype: None
        """
        full_path = self.path.joinpath(filename)
        os.makedirs(full_path.parent, exist_ok=True)
        self._save_json(variable, full_path)
        if self.verbose:
            print(f"{current_time()} | Saved to {full_path} successfully.")

    def load_json(self, filename):
        """
        Load serialized python object from json

        :param filename: name of the file from which we are loading the object
        :type filename: str
        :return: the loaded object data
        :rtype: may vary
        """
        full_path = self.path.joinpath(filename)
        file = self._load_json(full_path)
        if self.verbose:
            print(f"{current_time()} | Loaded from {full_path} successfully.")
        return file

    def save_txt(self, variable, filename):
        """
        Save serialized python object into a text format, at filename

        :param variable: the object to save
        :type variable: serialized object
        :param filename: name of the file to which variable should be saved
        :type filename: str
        :return: the function returns nothing
        :rtype: None
        """
        full_path = self.path.joinpath(filename)
        os.makedirs(full_path.parent, exist_ok=True)
        self._save_txt(variable, full_path)
        if self.verbose:
            print(f"{current_time()} | Saved to {full_path} successfully.")

    def load_txt(self, filename):
        """
        Load serialized python object from text file

        :param filename: name of the file from which we are loading the object
        :type filename: str
        :return: the loaded object data
        :rtype: may vary
        """
        full_path = self.path.joinpath(filename)
        file = self._load_txt(full_path)
        if self.verbose:
            print(f"{current_time()} | Loaded from {full_path} successfully.")
        return file

    def save_dataframe(self, df, filename):
        """
        Save a panda dataframe object to csv

        :param df: data contained in a dataframe
        :type df: panda dataframe
        :param filename: file to which data should be saved
        :return: the function returns nothing
        :rtype: None
        """
        ext = ".pkl"
        full_path = self.path.joinpath(filename + ext)
        os.makedirs(full_path.parent, exist_ok=True)
        # df.to_csv(str(full_path), sep=",", index=False, header=True)
        df.to_pickle(str(full_path))
        if self.verbose:
            print(f"{current_time()} | Saved to {full_path} successfully.")

    def load_dataframe(self, filename):
        """
        Load panda dataframe object from CSV

        :param filename: name of the file from which data should be loaded
        :type filename: str
        :return: dataframe data
        :rtype: panda dataframe
        """
        import pandas as pd

        ext = ".pkl"
        full_path = self.path.joinpath(filename + ext)
        # df = pd.read_csv(str(full_path), sep=",", header=0)
        df = pd.read_pickle(str(full_path))
        if self.verbose:
            print(f"{current_time()} | Loaded from {full_path} successfully.")
        return df

    def save_figure(self, fig, filename):
        """
        Save a figure (image datatype can be specified as part of filename)

        :param fig: the figure containing the figure to save
        :type fig: matplotlib.figure
        :param filename: the filename to which we save a figure
        :type filename: str
        :return: the function returns nothing
        :rtype: None
        """
        full_path = self.path.joinpath(filename)
        os.makedirs(full_path.parent, exist_ok=True)
        fig.savefig(full_path, dpi=300, bbox_inches="tight")
        if self.verbose:
            print(f"{current_time()} | Saved figure to {full_path} successfully.")

    def save_np_array(self, np_arr, filename):
        """
        Save numpy array to a text document

        :param np_arr: the array which we are saving
        :type np_arr: numpy.array
        :param filename: name of the text file to which we want to save the numpy array
        :type filename: str
        :return: the function returns nothing
        :rtype: None
        """
        import numpy as np

        full_path = self.path.joinpath(filename)
        os.makedirs(full_path.parent, exist_ok=True)
        np.savetxt(str(full_path), np_arr)
        if self.verbose:
            print(f"{current_time()} | Saved to {full_path} successfully.")

    def load_np_array(self, filename, complex_vals=False):
        """
        Loads numpy array from a text document

        :param filename: name of the text file from which we want to load the numpy array
        :type filename: str
        :param complex_vals: True if we expect the numpy array to be complex, False otherwise
        :type complex_vals: bool
        :return: the function returns nothing
        :rtype: None
        """
        import numpy as np

        full_path = self.path.joinpath(filename)
        file = np.loadtxt(
            str(full_path), dtype=np.complex if complex_vals else np.float
        )
        if self.verbose:
            print(f"{current_time()} | Loaded from {full_path} successfully.")
        return file

    def save_csv(self, df, filename):
        """
        Save a panda dataframe object to csv

        :param df: data contained in a dataframe
        :type df: panda dataframe
        :param filename: file to which data should be saved
        :return: the function returns nothing
        :rtype: None
        """
        ext = ".csv"
        full_path = self.path.joinpath(filename + ext)
        os.makedirs(full_path.parent, exist_ok=True)
        df.to_csv(str(full_path), sep=",", index=False, header=True)
        if self.verbose:
            print(f"{current_time()} | Saved to {full_path} successfully.")

    def load_csv(self, filename):
        import pandas as pd

        """
        Load panda dataframe object from CSV

        :param filename: name of the file from which data should be loaded
        :type filename: str
        :return: dataframe data
        :rtype: panda dataframe
        """
        full_path = self.path.joinpath(filename)
        df = pd.read_csv(str(full_path), sep=",", header=0)
        if self.verbose:
            print(f"{current_time()} | Loaded from {full_path} successfully.")
        return df

    # def new_h5_file(self, filename):
    #     """
    #     :param filename: name of the text file to which we want to save the numpy array
    #     """
    #     full_path = self.path.joinpath(filename)
    #     os.makedirs(full_path.parent, exist_ok=True)
    #     hf = h5py.File("test.h5", 'w')
    #     if self.verbose:
    #         print(f"{current_time()} | New H5 file at {full_path}.")

    @staticmethod
    def _save_json(variable, path):
        """
        Helper method for saving to json files
        """
        with open(path, "w+") as json_file:
            json.dump(variable, json_file, indent=4)

    @staticmethod
    def _load_json(path):
        """
        Helper method for loading from json files
        """
        with open(path) as json_file:
            data = json.load(json_file)
        return data

    @staticmethod
    def _save_txt(variable, path):
        """
        Helper method for saving to text files
        """
        with open(path, "w") as txt_file:
            txt_file.write(variable)

    @staticmethod
    def _load_txt(path):
        """
        Helper method for loading from text files
        """
        # with open(path) as json_file:
        #     data = json.load(json_file)
        # return data
        with open(path) as txt_file:
            txt_str = txt_file.read()
        return txt_str
