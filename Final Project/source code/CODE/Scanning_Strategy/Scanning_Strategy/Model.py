import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, num_heads, input_size):
        super(SelfAttention, self).__init__()

        self.num_heads = num_heads
        self.head_dim = input_size // num_heads

        assert input_size % num_heads == 0, "Input size must be divisible by the number of heads."

        self.query_linear = nn.Linear(input_size, input_size)
        self.key_linear = nn.Linear(input_size, input_size)
        self.value_linear = nn.Linear(input_size, input_size)

        self.output_linear = nn.Linear(input_size, input_size)

    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)

    def forward(self, x1):
        # x1.shape = (b, channels, h, w)
        batch_size, channels, h, w = x1.size()
        x1 = x1.reshape((batch_size, channels, -1))

        query = self.query_linear(x1)
        key = self.key_linear(x1)
        value = self.value_linear(x1)

        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)

        attention_weights = torch.matmul(query, key.transpose(2, 3))
        attention_weights = torch.softmax(attention_weights, dim=-1)

        output = torch.matmul(attention_weights, value)

        return output.transpose(1, 2).reshape((batch_size, channels, h, w))

class CrossAttention(nn.Module):
    def __init__(self, num_heads, input_size):
        super(CrossAttention, self).__init__()

        self.num_heads = num_heads
        self.head_dim = input_size // num_heads

        assert input_size % num_heads == 0, "Input size must be divisible by the number of heads."

        self.query_linear = nn.Linear(input_size, input_size)
        self.key_linear = nn.Linear(64, input_size)
        self.value_linear = nn.Linear(64, input_size)

        self.output_linear = nn.Linear(input_size, input_size)

    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)

    def forward(self, x1, x2):
        # x1.shape = (b, channels, 8, 8)
        # x2.shape = (b, 8, 8)
        batch_size, channels, h, w = x1.size()
        x2 = x2.unsqueeze(0).repeat(batch_size, 1, 1)

        query = self.query_linear(x1.view(batch_size, channels, -1))
        key = self.key_linear(x2.view(batch_size, -1))
        value = self.value_linear(x2.view(batch_size,  -1))

        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)
        
        attention_weights = torch.matmul(query, key.transpose(2, 3))
        attention_weights = torch.softmax(attention_weights / (self.head_dim ** 0.5), dim=-1)

        output = torch.matmul(attention_weights, value)

        return output.transpose(1, 2).reshape((batch_size, channels, h, w))

class AttentionBlock(nn.Module):
    def __init__(self, n_head: int, channel):
        super().__init__()
        
        self.conv_input = nn.Conv2d(channel, channel, kernel_size=1, padding=0)

        self.layernorm_1 = nn.LayerNorm([channel, 8, 8])
        self.attention_1 = SelfAttention(n_head, 64)
        self.layernorm_2 = nn.LayerNorm([channel, 8, 8])
        self.attention_2 = CrossAttention(n_head, 64)
        self.layernorm_3 = nn.LayerNorm([channel, 8, 8])
        self.linear_geglu_1  = nn.Linear(channel, 4 * channel * 2)
        self.linear_geglu_2 = nn.Linear(4 * channel, channel)

        self.conv_output = nn.Conv2d(channel, channel, kernel_size=1, padding=0)
    
    def forward(self, x, context):
        residue_long = x

        x = self.conv_input(x)
        
        residue_short = x
        x = self.layernorm_1(x)
        x = self.attention_1(x)
        x += residue_short

        residue_short = x
        x = self.layernorm_2(x)
        x = self.attention_2(x, context)
        x += residue_short

        residue_short = x
        x = self.layernorm_3(x)

        b, c, h, w = x.shape
        x = x.view(b, c, h*w)
        x = x.transpose(1, 2)

        x, gate = self.linear_geglu_1(x).chunk(2, dim=-1)
        x = x * F.gelu(gate)
        x = self.linear_geglu_2(x)

        x = x.transpose(1, 2)
        x = x.view(b, c, h, w)
        x += residue_short

        return self.conv_output(x) + residue_long


class SwitchSequential(nn.Sequential):
    def forward(self, x, cond):
        for layer in self:
            if isinstance(layer, AttentionBlock):
                x = layer(x, cond)
            else:
                x = layer(x)
        return x


class StrategyStepLayer(nn.Module):
    def __init__(self, n_head: int, layer_channels:list):
        super().__init__()
        self.layers = nn.ModuleList([])
        for i in range(len(layer_channels)-1):
            self.layers.append(SwitchSequential(AttentionBlock(n_head=n_head, channel=layer_channels[i])))
            self.layers.append(SwitchSequential(nn.Conv2d(layer_channels[i], layer_channels[i+1], 1)))

        self.softmax = nn.Softmax(1)
        self.flatten = nn.Flatten()

    def forward(self, x, current_step):

        for layer in self.layers:
            x = layer(x, current_step)
        
        return self.softmax(self.flatten(x))


class StrategyModel(nn.Module):
    def __init__(self, n_head:int, layer_channels:list):
        super().__init__()

        self.strategysteplayer = StrategyStepLayer(n_head=n_head, layer_channels=layer_channels)

    def forward(self, x):
        first_step = torch.zeros((8, 8), dtype=torch.float32).to(x.device)
        chosen_position = []
        for _ in range(64):
            next_step_prob = torch.mean(self.strategysteplayer(x, first_step), dim=0)
            next_step_prob[chosen_position] = -1
            next_step = torch.argmax(next_step_prob).item()
            chosen_position.append(next_step)
            i, j = (next_step+1)//8, (next_step+1)%8
            first_step[j-1, i-1] = -1
        
        return chosen_position
            


# model = StrategyModel(8, [1, 8, 16, 16, 8, 1])

# image_block = torch.rand(32, 1, 8, 8)

# print(model(image_block))
