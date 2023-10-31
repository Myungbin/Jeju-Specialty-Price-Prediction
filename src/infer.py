from src.data.dataset import TestDataset

test_dataset = TestDataset(x_test)
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=64)