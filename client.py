import pandas as pd
import torch
import torch.nn as nn
import torch_geometric as tg
from torch_geometric.nn import GCNConv
from sklearn.preprocessing import LabelEncoder
import pickle

#load the data
df = pd.read_csv('wireshark_data.csv')

#label encoding
le = LabelEncoder()
df['Source_encoded'] = le.fit_transform(df['Source'])
df['Destination_encoded'] = le.fit_transform(df['Destination'])
df['Protocol_encoded'] = le.fit_transform(df['Protocol'])

#defining node features
node_features = df[['Time', 'Length', 'Win Size', 'Source_encoded', 'Destination_encoded', 'Protocol_encoded']].values

#create edges
edges = []
for i, row1 in df.iterrows():
    for j, row2 in df.iterrows():
        if row1['Source'] == row2['Source'] and row1['Destination'] == row2['Destination'] and i != j:
            edges.append([i, j])

edges = torch.tensor(edges).t().contiguous()

#labels
df['Label'] = (df['Protocol_encoded'] % 2).astype(int)
labels = torch.tensor(df['Label'].values, dtype=torch.long)

#node features
x = torch.tensor(node_features, dtype=torch.float)
data = tg.data.Data(x=x, edge_index=edges, y=labels)

#GNN model
class GNNModel(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GNNModel, self).__init__()
        self.conv1 = GCNConv(in_channels, 64)
        self.conv2 = GCNConv(64, out_channels)
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x

#model initialization
model = GNNModel(in_channels=x.shape[1], out_channels=2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

#local training
def train_local(model, data, epochs=10):
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, data.y)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")  #  Print Training Loss
        for name, param in model.named_parameters():
            print(f"{name} Before:", param.data)


#run training for 10 epochs
print("Training the GNN Model on the CSV Dataset...")
train_local(model, data, epochs=10)

#test model accuracy
model.eval()
with torch.no_grad():
    output = model(data)
    predicted = output.argmax(dim=1)
    accuracy = (predicted == data.y).sum().item() / len(data.y)
    print(f"Training Completed. Model Accuracy: {accuracy:.2%}")
    print(df['Label'].value_counts())
    print("Output Shape:", output.shape)
    print("Labels Shape:", data.y.shape)
    print("Number of edges:", edges.shape)
    

filename = 'finalized_model.sav'#save model for future use
pickle.dump(model, open(filename, 'wb'))#save model for future use
