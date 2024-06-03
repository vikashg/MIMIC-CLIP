from monai.networks.nets import ResNet
from monai.transforms import (LoadImaged, AddChanneld, ScaleIntensityRanged, Compose, ToTensord, AsChannelFirstd)
from monai.data import Dataset, DataLoader
from monai.transforms import Activations, AsDiscrete
import torch
import tqdm
import json
from monai.data import decollate_batch
from monai.metrics import ROCAUCMetric



def main():
    filename = '/workspace/data/image/CXLSeg-segmented-processed.json'
    fid = open(filename, 'r')
    data_dict = json.load(fid)
    fid.close()
    train_data = data_dict['train']
    val_data = data_dict['val']
    test_data = data_dict['test']

    # Create a resnet model
    device = torch.device('cuda:0')
    model = ResNet(block='basic',
                   layers=[1, 1, 1, 1],
                   block_inplanes=[64, 128, 256, 512],
                   spatial_dims=2,
                   n_input_channels=3,
                   num_classes=27
                   ).to(device)

    # Create a dataloader
    batch_size=16
    train_transforms = Compose([LoadImaged(keys=['image']),
                                AsChannelFirstd(keys=['image']),
                                ScaleIntensityRanged(keys=['image'], a_min=-1000, a_max=1000, b_min=0.0, b_max=1.0,
                                                     clip=True),
                                ToTensord(keys=['image'])
                                ])
    train_ds = Dataset(data=train_data, transform=train_transforms)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)

    val_transforms = Compose([LoadImaged(keys=['image']),
                                AsChannelFirstd(keys=['image']),
                                ScaleIntensityRanged(keys=['image'], a_min=-1000, a_max=1000, b_min=0.0, b_max=1.0,
                                                        clip=True),
                                ToTensord(keys=['image'])
                                ])
    val_ds = Dataset(data=val_data, transform=val_transforms)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=True, num_workers=4)

    for _ in train_loader:
        print(_['image'].shape)
        a = torch.transpose(torch.stack(_['label']), 0, 1).to(device)
        print(a.shape)
        break

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 100
    epoch_loss_values = []
    metric_values = []
    best_metric_epoch = -1
    best_metric = -1
    y_pred_trans = Compose([Activations(softmax=True),])
    y_trans = Compose([AsDiscrete(to_onehot=27),])

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        model.train()
        epoch_loss = 0
        step = 0
        for batch in tqdm.tqdm(train_loader):
            image = batch['image'].to(device)
            label = torch.transpose(torch.stack(batch['label']), 0, 1).to(device)
            optimizer.zero_grad()
            output = model(image)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(output, label)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            step += 1
            epoch_len = len(train_ds) // train_loader.batch_size
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f'Epoch {epoch+1} average loss: {epoch_loss:.4f}')

        if (epoch + 1) % 1 == 0:
            model.eval()
            with torch.no_grad():
                y_pred = torch.tensor([], dtype=torch.float32, device=device)
                y = torch.tensor([], dtype=torch.float32, device=device)
                for val_batch in val_loader:
                    image = val_batch['image'].to(device)
                    label = torch.transpose(torch.stack(val_batch['label']), 0, 1).to(device)
                    output = model(image)
                    y_pred = torch.cat([y_pred, output], dim=0)
                    y = torch.cat([y, label], dim=0)
                print(y_pred.shape, y.shape)
                # y_onehot = [y_trans(i) for i in decollate_batch(y, detach=False)]
                # y_pred_act = [y_pred_trans(i) for i in decollate_batch(y_pred, detach=False)]
                # y_onehot = y_trans(y)
                y_pred_act = y_pred_trans(y_pred)

                auc_metric = ROCAUCMetric()
                auc_metric(y_pred_act, y)
                result = auc_metric.aggregate()
                auc_metric.reset()
                del y_pred_act
                metric_values.append(result)
                acc_value = torch.eq(y_pred.argmax(dim=1), y.argmax(dim=1))
                acc_metric = acc_value.sum().item() / len(acc_value)
                print(f'Validation accuracy: {acc_metric:.4f}')
                print(f'Validation AUC: {result:.4f}')
                if result > best_metric:
                    best_metric = result
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), 'best_metric_model_classification2d_array.pth')
                    print('saved new best metric model')


if __name__ == '__main__':
    main()
