import torch


def inference(model, test_loader):
    model.eval()
    pred_list = []
    with torch.inference_mode():
        for data in test_loader:
            pred = model(data)
            model_pred = pred.cpu().numpy().reshape(-1).tolist()
            pred_list += model_pred

    return pred_list
