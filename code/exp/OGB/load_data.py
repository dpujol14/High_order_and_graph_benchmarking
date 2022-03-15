from ogb.graphproppred import DglGraphPropPredDataset, collate_dgl
from torch.utils.data import DataLoader

def load_dataset(data_path, dataset_name, batch_size=8):
    dataset = DglGraphPropPredDataset(name=dataset_name, root=data_path)

    split_idx = dataset.get_idx_split()
    train_loader = DataLoader(dataset[split_idx["train"]], batch_size=batch_size, shuffle=True, collate_fn=collate_dgl)
    valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=batch_size, shuffle=False, collate_fn=collate_dgl)
    test_loader = DataLoader(dataset[split_idx["test"]], batch_size=batch_size, shuffle=False, collate_fn=collate_dgl)

    return train_loader, valid_loader, test_loader