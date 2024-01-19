from dotenv import load_dotenv
import pathlib

path_env = pathlib.Path(__file__).parent.joinpath('paths.env')
load_dotenv(path_env)