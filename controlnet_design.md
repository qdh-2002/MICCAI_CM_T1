# Converting Current UNet to ControlNet Architecture

## Current Architecture Analysis
Your current `UNetModel` has:
- **Input conditioning**: `hr_inte` (concatenated to input)
- **Cross-attention conditioning**: PSF features (from k-space mask)
- **Sequential processing**: Standard UNet encoder-decoder structure

## ControlNet Conversion Strategy

### 1. Architecture Overview
```
Original UNet (frozen)        ControlNet (trainable)
      ↓                              ↓
[Input + hr_inte] ──┐        [Input + conditions] 
      ↓             │              ↓
  Encoder blocks ───┼───────── Encoder blocks (copy)
      ↓             │              ↓
  Middle block ─────┼───────── Middle block (copy)
      ↓             │              ↓
  Decoder blocks    │         Zero convolutions
      ↓             │              ↓
    Output ←────────┴──────── Control features
```

### 2. Key Components

#### A. Base UNet (Frozen)
- Keep your existing `UNetModel` as the base
- Freeze all parameters
- Remove conditioning inputs (make unconditional)

#### B. ControlNet Branch (Trainable)
- Copy of encoder + middle block structure
- Additional input channels for conditions
- Zero convolutions at connection points
- Lightweight compared to full UNet

### 3. Implementation Plan

#### Step 1: Create Unconditional Base UNet
```python
class UnconditionalUNet(UNetModel):
    def __init__(self, *args, **kwargs):
        # Remove hr_inte concatenation requirement
        super().__init__(*args, **kwargs)
        
    def forward(self, x, timesteps, y=None):
        # No hr_inte or kspace_mask conditioning
        return super().forward(x, timesteps, hr_inte=None, kspace_mask=None, y=y)
```

#### Step 2: Create ControlNet
```python
class MRIControlNet(nn.Module):
    def __init__(self, base_unet_config):
        super().__init__()
        # Copy encoder structure from base UNet
        self.control_encoder = self._build_encoder(base_unet_config)
        self.control_middle = self._build_middle(base_unet_config)
        
        # Zero convolutions for each connection point
        self.zero_convs = nn.ModuleList([
            self._make_zero_conv(channels) 
            for channels in encoder_channel_list
        ])
        
        # Condition encoders
        self.hr_inte_encoder = nn.Conv2d(1, 64, 3, padding=1)  
        self.psf_encoder = PromptEncoder(1, 64)
        
    def forward(self, x, timesteps, hr_inte, kspace_mask):
        # Encode conditions
        hr_features = self.hr_inte_encoder(hr_inte)
        psf_features = self.psf_encoder(compute_psf_from_mask(kspace_mask))
        
        # Concatenate input with conditions
        x_control = torch.cat([x, hr_features, psf_features], dim=1)
        
        # Forward through control encoder
        control_features = []
        h = x_control
        for i, (module, zero_conv) in enumerate(zip(self.control_encoder, self.zero_convs)):
            h = module(h, timesteps)
            control_features.append(zero_conv(h))
            
        return control_features
```

#### Step 3: Combined Model
```python
class ControlledConsistencyModel(nn.Module):
    def __init__(self, base_unet, controlnet):
        super().__init__()
        self.base_unet = base_unet  # Frozen
        self.controlnet = controlnet  # Trainable
        
        # Freeze base UNet
        for param in self.base_unet.parameters():
            param.requires_grad = False
            
    def forward(self, x, timesteps, hr_inte=None, kspace_mask=None):
        if hr_inte is not None and kspace_mask is not None:
            # Get control features
            control_features = self.controlnet(x, timesteps, hr_inte, kspace_mask)
            
            # Forward through base UNet with control injection
            return self.base_unet.forward_with_control(x, timesteps, control_features)
        else:
            # Unconditional generation
            return self.base_unet(x, timesteps)
```

### 4. Training Strategy

#### Phase 1: Train Base Unconditional CM
```python
# Train base consistency model without any conditioning
base_model = UnconditionalUNet(...)
# Standard CM training loop
```

#### Phase 2: Add ControlNet
```python
# Load pretrained base model and freeze
base_model.requires_grad_(False)

# Train only ControlNet
controlnet = MRIControlNet(base_model.config)
combined_model = ControlledConsistencyModel(base_model, controlnet)

# Training loop focuses on ControlNet parameters only
```

### 5. Advantages of This Approach

1. **Modular**: Base model can be used unconditionally
2. **Efficient**: Only train smaller ControlNet branch
3. **Flexible**: Easy to swap different conditioning strategies
4. **Stable**: Proven architecture from ControlNet paper

### 6. Modifications Needed

1. **Modify current UNet**: Make it unconditional-capable
2. **Extract encoder logic**: Reuse in ControlNet
3. **Add zero convolutions**: For stable training initialization
4. **Update training script**: Handle two-stage training

### 7. File Changes Required

1. `cm/unet.py`: Add unconditional mode, ControlNet class
2. `cm/controlnet.py`: New ControlNet implementation  
3. `scripts/train_controlnet.py`: New training script
4. `cm/train_util.py`: Update for ControlNet training

This conversion maintains your current conditioning approach (hr_inte + PSF) while following the proven ControlNet parallel architecture pattern.
