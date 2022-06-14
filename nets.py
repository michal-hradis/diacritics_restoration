import torch


class Net(torch.nn.Module):
    def __init__(self, input_count, output_channels, inner_channels=64):
        super().__init__()
        self.input_embedding = torch.nn.Embedding(input_count, inner_channels)
        self.rec = torch.nn.LSTM(inner_channels, inner_channels // 2, num_layers=2, bidirectional=True, batch_first=True)
        self.output_layer = torch.nn.Conv1d(inner_channels, output_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.input_embedding(x)
        out, _ = self.rec(x)
        out = torch.permute(out, (0, 2, 1))
        out = self.output_layer(out)
        #out = torch.nn.functional.softmax(out, dim=1)
        return out