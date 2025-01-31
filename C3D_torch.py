import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, activation='relu', padding=1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.activation = nn.ReLU() if activation == 'relu' else activation

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activation(x)
        return x

class EncoderBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, pool_size=2):
        super().__init__()
        self.conv_block = ConvBlock3D(in_channels, out_channels)
        self.pool = nn.MaxPool3d(pool_size)

    def forward(self, x):
        f = self.conv_block(x)
        p = self.pool(f)
        return f, p

class DecoderBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2):
        super().__init__()
        self.up_conv = nn.ConvTranspose3d(in_channels, out_channels, 
                                        kernel_size=kernel_size, stride=stride)
        self.conv_block = ConvBlock3D(out_channels*2, out_channels)

    def forward(self, x, skip_features):
        x = self.up_conv(x)
        x = torch.cat([x, skip_features], dim=1)
        x = self.conv_block(x)
        return x

class Coarse3DUNet(nn.Module):
    def __init__(self, in_channels=3, n_filters_base=16, n_classes=1, final_activation='sigmoid'):
        super().__init__()
        # Encoder
        self.enc1 = EncoderBlock3D(in_channels, n_filters_base)
        self.enc2 = EncoderBlock3D(n_filters_base, n_filters_base*2)
        self.enc3 = EncoderBlock3D(n_filters_base*2, n_filters_base*4)

        # Bottleneck
        self.bottleneck = ConvBlock3D(n_filters_base*4, n_filters_base*8)

        # Decoder
        self.dec3 = DecoderBlock3D(n_filters_base*8, n_filters_base*4)
        self.dec2 = DecoderBlock3D(n_filters_base*4, n_filters_base*2)
        self.dec1 = DecoderBlock3D(n_filters_base*2, n_filters_base)

        # Final output
        self.final_conv = nn.Conv3d(n_filters_base, n_classes, kernel_size=1)
        self.final_activation = nn.Sigmoid() if final_activation == 'sigmoid' else None

    def forward(self, x):
        # Encoder
        f1, p1 = self.enc1(x)
        f2, p2 = self.enc2(p1)
        f3, p3 = self.enc3(p2)

        # Bottleneck
        bn = self.bottleneck(p3)

        # Decoder
        d3 = self.dec3(bn, f3)
        d2 = self.dec2(d3, f2)
        d1 = self.dec1(d2, f1)

        # Final outputs
        coarse_seg = self.final_conv(d1)
        if self.final_activation:
            coarse_seg = self.final_activation(coarse_seg)
            
        return coarse_seg, d1

class Fine3DUNet(nn.Module):
    def __init__(self, in_channels=19, n_filters_base=32, n_classes=1, final_activation='sigmoid'):
        super().__init__()
        # Encoder
        self.enc1 = EncoderBlock3D(in_channels, n_filters_base)
        self.enc2 = EncoderBlock3D(n_filters_base, n_filters_base*2)
        self.enc3 = EncoderBlock3D(n_filters_base*2, n_filters_base*4)
        self.enc4 = EncoderBlock3D(n_filters_base*4, n_filters_base*8)

        # Bottleneck
        self.bottleneck = ConvBlock3D(n_filters_base*8, n_filters_base*16)

        # Decoder
        self.dec4 = DecoderBlock3D(n_filters_base*16, n_filters_base*8)
        self.dec3 = DecoderBlock3D(n_filters_base*8, n_filters_base*4)
        self.dec2 = DecoderBlock3D(n_filters_base*4, n_filters_base*2)
        self.dec1 = DecoderBlock3D(n_filters_base*2, n_filters_base)

        # Final output
        self.final_conv = nn.Conv3d(n_filters_base, n_classes, kernel_size=1)
        self.final_activation = nn.Sigmoid() if final_activation == 'sigmoid' else None

    def forward(self, x):
        # Encoder
        f1, p1 = self.enc1(x)
        f2, p2 = self.enc2(p1)
        f3, p3 = self.enc3(p2)
        f4, p4 = self.enc4(p3)

        # Bottleneck
        bn = self.bottleneck(p4)

        # Decoder
        d4 = self.dec4(bn, f4)
        d3 = self.dec3(d4, f3)
        d2 = self.dec2(d3, f2)
        d1 = self.dec1(d2, f1)

        # Final output
        fine_seg = self.final_conv(d1)
        if self.final_activation:
            fine_seg = self.final_activation(fine_seg)
            
        return fine_seg

class Cascade3DUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.coarse_net = Coarse3DUNet(in_channels=3, n_filters_base=16)
        self.fine_net = Fine3DUNet(in_channels=3+16, n_filters_base=32)

    def forward(self, coarse_input, fine_input):
        # Process coarse input
        coarse_seg, coarse_features = self.coarse_net(coarse_input)
        
        # Concatenate high-res input with coarse features
        combined_input = torch.cat([fine_input, coarse_features], dim=1)
        
        # Process fine input
        fine_seg = self.fine_net(combined_input)
        
        return coarse_seg, fine_seg
    

model = Cascade3DUNet()
coarse_input = torch.randn(1, 3, 192, 192, 192)  # (batch, channels, D, H, W)
fine_input = torch.randn(1, 1, 192, 192, 192)
coarse_seg, fine_seg = model(coarse_input, fine_input)