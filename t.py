import torch
import argparse

from models.dyconvnext import ConvNeXt_Teacher, AdaConvNeXt
from models.dylvvit import LVViTDiffPruning, LVViT_Teacher
from models.dyvit import VisionTransformerDiffPruning, VisionTransformerTeacher
from models.dyswin import AdaSwinTransformer, SwinTransformer_Teacher


parser = argparse.ArgumentParser(description="Benchmark models for Vision Transformers")

parser.add_argument("--img_sz", type=int, default=224, help="size of input")
parser.add_argument("--batch_size", type=int, default=32, help="size of the batch")
parser.add_argument("--base_rate", type=float, default=0.7, help="percentage of pruning")
parser.add_argument("--nb_classes", type=int, default=4, help="percentage of pruning")
parser.add_argument("--distill", type=bool, default=True, help="distillation flag")

args = parser.parse_args()
sparse_ratio = [args.base_rate, args.base_rate-0.2, args.base_rate-0.4]


class Args:
    def __init__(self):
        self.some_param = 0

model = torch.load('fine_tuned_model/deiT_small_patch16_224.ckpt', map_location='cpu')
pruned_model_weight = torch.load('prune_model/DeiT-s/DeiT_s_70.pth', map_location='cpu', weights_only=False)["model"]


PRUNING_LOC = [3, 6, 9]
KEEP_RATE = [sparse_ratio[0], sparse_ratio[0] ** 2, sparse_ratio[0] ** 3]
print('token_ratio =', KEEP_RATE, 'at layer', PRUNING_LOC)
pruned_model = VisionTransformerDiffPruning(
    patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
pruning_loc=PRUNING_LOC, token_ratio=KEEP_RATE, distill=args.distill, num_classes = args.nb_classes
)
base_model = VisionTransformerTeacher(
    patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True, num_classes = args.nb_classes)


base_model.load_state_dict(model, strict=False)
base_model.eval()


from calc_flops import calc_flops, throughput

print('results for base model')
img_size = args.img_sz
calc_flops(base_model, img_size=img_size, show_details=False, ratios=sparse_ratio)
x = torch.randn(args.batch_size, 3, img_size, img_size, device='cuda')
base_model = base_model.cuda()
throughput(x, base_model)


print('base model')
print(base_model)

print('pruned_model')
print(pruned_model)


print('results for pruned model')
pruned_model.load_state_dict(pruned_model_weight, strict=False)
pruned_model.eval()
# pruned_model = pruned_model.cuda()
calc_flops(pruned_model, img_size=img_size, show_details=False, ratios=sparse_ratio)
x = torch.randn(args.batch_size, 3, img_size, img_size, device='cuda')
pruned_model = pruned_model.cuda()
throughput(x, pruned_model)
print('done')
