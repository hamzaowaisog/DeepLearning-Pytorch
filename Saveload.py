import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, n_input_features):
        super(Model, self).__init__()
        self.linear = nn.Linear(n_input_features, 1)
    
    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred
    
model = Model(n_input_features=6)
# Lazy saving
FILE = "model.pth"
# torch.save(model,FILE)

# model = torch.load(FILE)
# model.eval()

# for param in model.parameters():
    # print(param)
    
## prefer method
# torch.save(model.state_dict() , FILE)
# loaded_model = Model(n_input_features=6)
# loaded_model.load_state_dict(torch.load(FILE))
# loaded_model.eval()

# for param in loaded_model.parameters():
#     print(param)

## ssaving everything
# train model

learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
print(optimizer.state_dict())

checkpoint = {
    "epoch": 90,
    "model_state": model.state_dict(),
    "optim_state": optimizer.state_dict()
}

# torch.save(checkpoint, "checkpoint.pth")
# torch.save(checkpoint, "checkpoint.pth", _use_new_zipfile_serialization=False)

loaded_checkpoint = torch.load("checkpoint.pth")
epoch = loaded_checkpoint["epoch"]
model = Model(n_input_features=6)
model.load_state_dict(loaded_checkpoint["model_state"])
optimizer = torch.optim.SGD(model.parameters(), lr=0)
optimizer.load_state_dict(loaded_checkpoint["optim_state"])

print(optimizer.state_dict())
print(epoch)

