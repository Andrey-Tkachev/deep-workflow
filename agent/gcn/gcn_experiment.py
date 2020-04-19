import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
import gcn


class DummyRegressor(nn.Module):

    def __init__(self, input_dim, hidden_dim=16):
        super(DummyRegressor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.network(x)


def run_epoch(model, optimizer, dags, masks, critical_values, train=False):
    sliding_loss = 0.0
    sliding_accuracy = 0.0
    for curr_iter, (dag, mask, crit_value) in enumerate(zip(dags, masks, critical_values)):
        if train:
            optimizer.zero_grad()

        embeddings = model(dag=dag, mask=mask)
        prediction = regressor(embeddings)
        crit_value = crit_value.unsqueeze(1)
        loss = F.mse_loss(prediction, crit_value) / (torch.max(crit_value) ** 2.0)
        if train:
            loss.backward()
            optimizer.step()

        true_node = torch.argmax(crit_value).item()
        pred_node = torch.argmax(prediction).item()
        sliding_accuracy += int(pred_node == true_node)
        sliding_loss += loss.item()

        print(f'{"train" if train else "val"}; '
              f'Loss: {sliding_loss / (curr_iter + 1):.3f}; '
              f'Accuracy: {100.0 * sliding_accuracy / (curr_iter + 1):.3f}\r', end='\r')

    print()


if __name__ == "__main__":
    model = gcn.GCNNetwork(input_dim=3, hidden_dim=16, emb_dim=5)
    regressor = DummyRegressor(input_dim=5)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    dags, masks, critical_values = utils.create_dag_dataset()
    val_dags, val_masks, val_critical_values = utils.create_dag_dataset()

    for epoch in range(10):
        print(f'Epoch {epoch + 1}')
        model.train()
        run_epoch(model, optimizer, dags, masks, critical_values, train=True)

        model.eval()
        run_epoch(model, optimizer, dags, masks, critical_values, train=False)
