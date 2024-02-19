# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Copyright (c) 2022-2024 Benjamin MacLellan

import pathlib
import warnings
import os
import copy
import datetime
import json
import string
import random
import yaml
import h5py
from dataclasses import dataclass, field, fields, asdict


def current_time():
    """
    Returns the current date and time in a consistent format.

    This function is used for monitoring long-running measurements by providing the current date and time in the "%d/%m/%Y, %H:%M:%S" format.

    Returns:
        str: The current date and time as a string in the "%d/%m/%Y, %H:%M:%S" format.
    """
    return datetime.datetime.now().strftime("%d/%m/%Y, %H:%M:%S")


class IO:
    r"""
    The IO class encapsulates all saving/loading features of data, figures, etc.
    This provides consistent filetypes, naming conventions, etc.

    Attributes:
        default_path (pathlib.Path): The default path where the data is stored.
        path (pathlib.Path): The path where the data is stored.
        verbose (bool): A flag indicating whether to print out the path of each saved/loaded file.

    Typical usage:
        io = IO(path=r"\path\to\data")
        io.load_txt(filename="filename.txt")

    or
        io = IO.create_new_save_folder(folder="subfolder", include_date=True, include_uuid=True)
        io.save_df(df, filename="dataframe.txt")
    """
    default_path = os.getenv("DATA_PATH", pathlib.Path(__file__).parent.parent.joinpath("data"))

    def __init__(
        self,
        path=None,
        folder="",
        include_date=False,
        include_time=False,
        include_id=False,
        verbose=True,
    ):
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

        Args:
            variable: The object to save.
            filename (str): Name of the file to which variable should be saved.

        Returns:
            None
        """
        full_path = self.path.joinpath(filename)
        os.makedirs(full_path.parent, exist_ok=True)
        self._save_json(variable, full_path)
        if self.verbose:
            print(f"{current_time()} | Saved to {full_path} successfully.")

    def load_json(self, filename):
        """
        Load serialized python object from json.

        Args:
            filename (str): Name of the file from which we are loading the object.

        Returns:
            The loaded object data.
        """
        full_path = self.path.joinpath(filename)
        file = self._load_json(full_path)
        if self.verbose:
            print(f"{current_time()} | Loaded from {full_path} successfully.")
        return file

    def save_txt(self, variable, filename):
        """
        Save serialized python object into a text format, at filename.

        Args:
            variable: The object to save.
            filename (str): Name of the file to which variable should be saved.

        Returns:
            None
        """
        full_path = self.path.joinpath(filename)
        os.makedirs(full_path.parent, exist_ok=True)
        self._save_txt(variable, full_path)
        if self.verbose:
            print(f"{current_time()} | Saved to {full_path} successfully.")

    def load_txt(self, filename):
        """
        Load serialized python object from text file.

        Args:
            filename (str): Name of the file from which we are loading the object.

        Returns:
            The loaded object data.
        """
        full_path = self.path.joinpath(filename)
        file = self._load_txt(full_path)
        if self.verbose:
            print(f"{current_time()} | Loaded from {full_path} successfully.")
        return file

    def save_dataframe(self, df, filename):
        """
        Save a panda dataframe object to csv.

        Args:
            df (pandas.DataFrame): Data contained in a dataframe.
            filename (str): File to which data should be saved.

        Returns:
            None
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
        Load panda dataframe object from CSV.

        Args:
            filename (str): Name of the file from which data should be loaded.

        Returns:
            pandas.DataFrame: Dataframe data.
        """
        import pandas as pd

        ext = ".pkl"
        full_path = self.path.joinpath(filename + ext)
        # df = pd.read_csv(str(full_path), sep=",", header=0)
        df = pd.read_pickle(str(full_path))
        if self.verbose:
            print(f"{current_time()} | Loaded from {full_path} successfully.")
        return df

    def save_yaml(self, data, filename):
        """
        Save dictionary to YAML file.

        Args:
            filename (str): Name of the file from which data should be saved.

        """

        full_path = self.path.joinpath(filename)
        os.makedirs(full_path.parent, exist_ok=True)
        with open(full_path, "w") as fid:
            _data = asdict(data)
            # with open(file, "w") as fid:
            yaml.dump(_data, fid)

        if self.verbose:
            print(f"{current_time()} | Saved to {full_path} successfully.")

    def save_figure(self, fig, filename):
        """
        Save a figure (image datatype can be specified as part of filename).

        Args:
            fig (matplotlib.figure.Figure): The figure containing the figure to save.
            filename (str): The filename to which we save a figure.

        Returns:
            None
        """
        full_path = self.path.joinpath(filename)
        os.makedirs(full_path.parent, exist_ok=True)
        fig.savefig(full_path, dpi=300, bbox_inches="tight")
        if self.verbose:
            print(f"{current_time()} | Saved figure to {full_path} successfully.")

    def save_np_array(self, np_arr, filename):
        """
        Save numpy array to a text document.

        Args:
            np_arr (numpy.array): The array which we are saving.
            filename (str): Name of the text file to which we want to save the numpy array.

        Returns:
            None
        """
        import numpy as np

        full_path = self.path.joinpath(filename)
        os.makedirs(full_path.parent, exist_ok=True)
        np.savetxt(str(full_path), np_arr)
        if self.verbose:
            print(f"{current_time()} | Saved to {full_path} successfully.")

    def load_np_array(self, filename, complex_vals=False):
        """
        Loads numpy array from a text document.

        Args:
            filename (str): Name of the text file from which we want to load the numpy array.
            complex_vals (bool): True if we expect the numpy array to be complex, False otherwise.

        Returns:
            numpy.array: The loaded numpy array.
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
        Save a panda dataframe object to csv.

        Args:
            df (pandas.DataFrame): Data contained in a dataframe.
            filename (str): File to which data should be saved.

        Returns:
            None
        """
        ext = ".csv"
        full_path = self.path.joinpath(filename + ext)
        os.makedirs(full_path.parent, exist_ok=True)
        df.to_csv(str(full_path), sep=",", index=False, header=True)
        if self.verbose:
            print(f"{current_time()} | Saved to {full_path} successfully.")

    def load_csv(self, filename):
        """
        Load panda dataframe object from CSV.

        Args:
            filename (str): Name of the file from which data should be loaded.

        Returns:
            pandas.DataFrame: Dataframe data.
        """
        import pandas as pd

        full_path = self.path.joinpath(filename)
        df = pd.read_csv(str(full_path), sep=",", header=0)
        if self.verbose:
            print(f"{current_time()} | Loaded from {full_path} successfully.")
        return df

    def save_h5(self, filename):
        """
        Initialize an H5 file to save datasets into.

        Args:
            filename (str): Name of the file from which data should be saved.

        Returns:
            h5py.File: H5 file.
        """
        full_path = self.path.joinpath(filename)
        os.makedirs(full_path.parent, exist_ok=True)
        hf = h5py.File(full_path, 'w')
        if self.verbose:
            print(f"{current_time()} | Saving HDF5 file at {full_path}.")
        return hf

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
