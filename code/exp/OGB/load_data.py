from ogb.graphproppred import DglGraphPropPredDataset, collate_dgl
from ogb.nodeproppred import DglNodePropPredDataset

from torch.utils.data import DataLoader

def load_graph_level_dataset(data_path, dataset_name, batch_size=8,):
    dataset = DglGraphPropPredDataset(name=dataset_name, root=data_path)

    split_idx = dataset.get_idx_split()
    train_loader = DataLoader(dataset[split_idx["train"]], batch_size=batch_size, shuffle=True, collate_fn=collate_dgl)
    valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=batch_size, shuffle=False, collate_fn=collate_dgl)
    test_loader = DataLoader(dataset[split_idx["test"]], batch_size=batch_size, shuffle=False, collate_fn=collate_dgl)

    return train_loader, valid_loader, test_loader

def load_node_level_dataset(data_path, dataset_name, batch_size=8,):
    dataset = DglNodePropPredDataset(name=dataset_name, root=data_path)

    # These datasets contain only a single graph, which must be split to form the train and validation graph
    # Thus, in this case the split is done at node level, and not graph level
    split_idx = dataset.get_idx_split()
    train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]

    data_loader = DataLoader([dataset[0]], batch_size=1, shuffle=False,  collate_fn=collate_dgl)

    return data_loader, train_idx, valid_idx, test_idx
