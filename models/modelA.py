class modelA (nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(8,6)
        self.hidden2 = nn.Linear(6,4)
        self.hidden3 = nn.Linear(4,4)
        self.out = nn.Linear(4,2)
        
    def forward(self,x):
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = F.relu(self.hidden3(x))
        x = self.out(x)
        return x
