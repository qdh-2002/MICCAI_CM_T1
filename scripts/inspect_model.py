import torch

# Replace with your actual path
ckpt_path = "output_logs/run_diffusion_adc/target_model000000.pt"
state_dict = torch.load(ckpt_path, map_location="cpu")

print(f"Total parameters: {len(state_dict)}")
for name, param in list(state_dict.items())[:20]:  # only first 20 for brevity
    print(f"{name}: {param.shape}")
