from typing import NamedTuple, Dict
import json
from pathlib import Path
from collections import OrderedDict
import numpy as np
import pandas as pd
from treelib import Node, Tree
import pyarrow as pa
import pyarrow.parquet as pq

__version__ = "0.6"

DDIR_FILE = 'ddir.json'
ATTRIBUTES_FILE = 'attributes.json'
DATA_FILE = 'data.parq'


class DataDirTypes(NamedTuple):
    FILE: str = 'file'
    GROUP: str = 'group'
    DATASET: str = 'dataset'
    RAW: str = 'raw'


DATA_DIR_TYPES = DataDirTypes()


class Error(Exception):
    pass


class GroupError(Error):
    def __init__(self, message):
        self.message = message


class ElementWithAttributes(object):

    def __init__(self):
        self.attrs: Dict = {}

    def set_attrs(self, attrs: Dict):
        self.attrs = attrs


class Group(ElementWithAttributes):

    def __init__(self):
        super(Group, self).__init__()
        self.type = DATA_DIR_TYPES.GROUP

        self.path = None
        self.tree = Tree()

    def __getitem__(self, item):

        try:
            node = self.tree[item]
        except KeyError:
            raise KeyError(f'{item} is not a valid key')

        if isinstance(node.data, Group):
            # rebuild tree with reduced identifiers
            stree = self.tree.subtree(item)
            for n in stree.all_nodes_itr():
                if n.predecessor(stree.identifier) is None:
                    parent = None
                else:
                    parent = n.predecessor(stree.identifier).split(item, maxsplit=1)[1]
                node.data.tree.create_node(n.tag, n.identifier.split(item, maxsplit=1)[1], parent)

        elif isinstance(node.data, DataSet):
            if node.data.df.empty:
                if self.path is None:
                    raise GroupError(f'{item} is not loaded yet and this is not linked to a File or Group')
                node.data.df = pd.read_parquet(self.path / item / DATA_FILE)

        return node.data

    def __setitem__(self, key, value):

        if isinstance(value, Group):
            pass
        elif isinstance(value, DataSet):
            pass
        elif isinstance(value, Raw):
            pass
        elif isinstance(value, Attribute):
            pass
        else:
            raise ValueError(f'{value} is not a valid type for DataDir')

    def link(self, path):
        self.path = path


class File(Group):

    def __init__(self, path, mode='r'):
        super(File, self).__init__()
        self.path = Path(path)
        self.mode = mode
        self.type = DATA_DIR_TYPES.FILE

        if not (self.path / DDIR_FILE).exists():
            raise ValueError(f'{path} is not a DataDir!')
        else:
            d = json.load((self.path / DDIR_FILE).open())
            if d['ddir']['type'] != DATA_DIR_TYPES.FILE:
                raise ValueError(f'{path} is not the root of the DataDir')

        self.tree.create_node('.', '.', None, data=self)

        if mode in list('ar'):
            # get the tree and the attributes of the nodes
            ls = sorted([i.relative_to(p).parent for i in self.path.glob('**/ddir.json')])
            ls[0] = Path('')
            parent = None
            self.set_attrs(json.load((self.path / ATTRIBUTES_FILE).open()))

            for item in ls[1:]:
                if item.parent != parent:
                    parent = item.parent
                dtype = json.load((path / item / DDIR_FILE).open())['ddir']['type']
                data = None
                if dtype == DATA_DIR_TYPES.GROUP:
                    data = Group()
                    data.link(self.path)
                elif dtype == DATA_DIR_TYPES.DATASET:
                    data = DataSet()
                elif dtype == DATA_DIR_TYPES.RAW:
                    data = Raw()
                if isinstance(data, ElementWithAttributes):
                    data.set_attrs(json.load((path / item / ATTRIBUTES_FILE).open()))
                self.tree.create_node(item.name, str(item), str(parent), data=data)


class DataSet(ElementWithAttributes):

    def __init__(self, df: pd.DataFrame = pd.DataFrame()):
        super(DataSet, self).__init__()
        self.df = df
        self.type = DATA_DIR_TYPES.DATASET


class Raw(object):

    def __init__(self):
        self.type = DATA_DIR_TYPES.RAW


class Attribute(object):

    def __init__(self):
        self.type = None


def _write_ddir_json(path, type: DataDirTypes):
    d = {'type': type, 'version': __version__}
    json.dump(d, (path / DDIR_FILE).open('w'), indent=4)


if __name__ == '__main__':
    p = Path('../../data/example_ddir')
    dd = File(p)
    dd.tree.show()
    gr_vent = dd['vent']
    ds_vent = dd['vent/signals']
    df_vent = ds_vent.df
    print(df_vent.info())
