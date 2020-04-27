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
        self._attrs: Dict = {}

    @property
    def attrs(self):
        return self._attrs

    @attrs.setter
    def attrs(self, value: Dict):
        if isinstance(value, Dict):
            self._attrs = value
        else:
            raise ValueError(f'Type of attributes must be Dict not {type(value)}')

    def set_attrs(self, attrs: Dict):
        self.attrs = attrs


class Group(ElementWithAttributes):

    def __init__(self):
        super(Group, self).__init__()
        self.type = DATA_DIR_TYPES.GROUP

        self.path = None
        self.tree = Tree()

    def __getitem__(self, item):

        if item not in self.tree:
            rsplit = item.rsplit('/', maxsplit=1)
            if len(rsplit) == 1:
                item_0 = self.tree.root
                key = rsplit[0]
            else:
                item_0, key = rsplit
            if item_0 in self.tree:
                node = self.tree[item_0]
                if isinstance(node.data, ElementWithAttributes) and key in node.data.attrs:
                    return node.data.attrs[key]  # ### RETURN attribute value ###

            raise KeyError(f'{item} is not a valid key')

        node = self.tree[item]

        if isinstance(node.data, Group):
            # rebuild tree with reduced identifiers
            stree = self.tree.subtree(item)
            for n in stree.all_nodes_itr():
                if n.predecessor(stree.identifier) is None:
                    parent = None
                else:
                    parent = n.predecessor(stree.identifier).split(item, maxsplit=1)[1]
                node.data.tree.create_node(n.tag, n.identifier.split(item, maxsplit=1)[1], parent, data=n.data)

        elif isinstance(node.data, DataSet):
            if node.data.df.empty:
                if self.path is None:
                    raise GroupError(f'{item} is not loaded yet and this element is not linked to a File or Group')
                node.data.df = pd.read_parquet(self.path / item / DATA_FILE)

        return node.data

    def __setitem__(self, key, value):

        if key in self.tree:
            raise KeyError(f'{key} already exists')

        rsplit = key.rsplit('/', maxsplit=1)
        if len(rsplit) == 1:
            item_0 = self.tree.root
            key_1 = rsplit[0]
        else:
            item_0, key_1 = rsplit

        if item_0 is not None and item_0 not in self.tree:
            raise KeyError(f'Parent key {item_0} does not exist')

        dd_type = None
        if isinstance(value, Group):
            dd_type = value.type
            new_tree = Tree()
            for node in value.tree.all_nodes_itr():
                if node.parent is None:
                    parent = None
                else:
                    parent = key + '/' + node.parent
                new_tree.create_node(node.tag, key + '/' + node.identifier, parent=parent, data=node.data)
                value.tree = new_tree
            self.tree.create_node(tag=key_1, identifier=key, parent=item_0, data=value)
            self.tree.paste(key, new_tree)

        elif isinstance(value, DataSet):
            dd_type = DATA_DIR_TYPES.DATASET
            self.tree.create_node(tag=key_1, identifier=key, parent=item_0, data=value)
            if self.path is not None:
                value.df.to_parquet(self.path / key / DATA_FILE)

        elif isinstance(value, Raw):
            pass
        elif isinstance(value, Attribute):
            pass
        else:
            raise ValueError(f'{value} is not a valid type for DataDir')

        # write ddir and attributes file if self is linked
        if isinstance(value, ElementWithAttributes) and self.path is not None:
            (self.path / key).mkdir()
            _write_ddir_json(self.path / key, dd_type=dd_type)
            json.dump(value.attrs, (self.path / key / ATTRIBUTES_FILE).open('w'), indent=4)

    def link(self, path):
        self.path = path


class File(Group):

    def __init__(self, path, mode='r'):
        super(File, self).__init__()
        self.path = Path(path)
        self.mode = mode
        self.type = DATA_DIR_TYPES.FILE


        if mode in list('ar'):
            # some checking
            if not (self.path / DDIR_FILE).exists():
                raise ValueError(f'{path} is not a DataDir!')
            d = json.load((self.path / DDIR_FILE).open())
            if d['ddir']['type'] != DATA_DIR_TYPES.FILE:
                raise ValueError(f'{path} is not the root of the DataDir')

            # get the tree and the attributes of the nodes
            ls = sorted([i.relative_to(p).parent for i in self.path.glob('**/ddir.json')])
            parent = None
            self.attrs = json.load((self.path / ATTRIBUTES_FILE).open())

            for item in ls:
                if str(item.parent) != parent:
                    if item == Path(''):
                        parent = None
                    else:
                        parent = str(item.parent)
                dtype = json.load((path / item / DDIR_FILE).open())['ddir']['type']
                data = None
                if dtype == DATA_DIR_TYPES.FILE:
                    data = self
                elif dtype == DATA_DIR_TYPES.GROUP:
                    data = Group()
                    data.link(self.path)
                elif dtype == DATA_DIR_TYPES.DATASET:
                    data = DataSet()
                elif dtype == DATA_DIR_TYPES.RAW:
                    data = Raw()
                if isinstance(data, ElementWithAttributes):
                    data.attrs = json.load((path / item / ATTRIBUTES_FILE).open())
                self.tree.create_node(item.name, str(item), parent, data=data)

        elif mode == 'w':
            if self.path.exists():
                raise ValueError(f'{path} already exists')
            if (self.path / DDIR_FILE).exists():
                raise ValueError(f'{path} contains already a data dir')
            self['.'] = self
            # _write_ddir_json(self.path, DATA_DIR_TYPES.FILE)

    def __setitem__(self, key, value):

        if self.mode not in list('aw'):
            raise ValueError(f'DataDir is read only - no setting allowed')

        super(File, self).__setitem__(key, value)


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


def _write_ddir_json(path, dd_type: str):
    d = {'ddir': {'type': dd_type, 'version': __version__}}
    json.dump(d, (path / DDIR_FILE).open('w'), indent=4)


if __name__ == '__main__':
    # p = Path('../../data/example_ddir')
    # dd = File(p)
    # dd.tree.show()
    # gr_vent = dd['vent']
    # dt = gr_vent['dT']
    # print(f'Sample time: {dt}')
    # ds_vent = dd['vent/signals']
    # df_vent = ds_vent.df
    # print(df_vent.info())

    p = Path('../../data/temp.dd')
    dd = File(p, 'w')
    dd['test'] = Group()
    dd.tree.show()
