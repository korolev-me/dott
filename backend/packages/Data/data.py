import os
import numpy as np
import pandas as pd

class Data:
    """
    Class for standart processec with data.
    """

    @staticmethod
    def read(path_folder, name_file):
        """
        Read dataframe from file or stream.

        :param path_folder: path to folder with file
        :param name_file: name_file of file
        :return:
        """
        path_file = "/".join( [path_folder, name_file] )
        data_df = pd.read_csv(path_file, sep=';')
        return data_df

    @staticmethod
    def save(data_df, path_folder, name_file):
        """
        Save dataframe to file.

        :param data_df: source dataframe
        :param path_folder:  path to folder with file
        :param name_file: name_file of file
        :return:
        """

        if not os.path.isdir( path_folder ):
            os.makedirs( path_folder )

        path_file = "/".join( [path_folder, name_file] )
        data_df.to_csv(path_file, sep=';', index=False)
        return True
