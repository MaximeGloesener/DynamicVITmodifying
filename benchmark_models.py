# -*- coding: utf-8 -*-
"""
Created on Wed May 28 10:32:45 2025

@author: Kévin
"""

import argparse
import torch

from models.dyconvnext import ConvNeXt_Teacher, AdaConvNeXt
from models.dylvvit import LVViTDiffPruning, LVViT_Teacher
from models.dyvit import VisionTransformerDiffPruning, VisionTransformerTeacher
from models.dyswin import AdaSwinTransformer, SwinTransformer_Teacher

from pytorch_bench import benchmark
from calc_flops import calc_flops, throughput

def parse_args():

    parser = argparse.ArgumentParser(description="Benchmark models for Vision Transformers")

    parser.add_argument("model", type=str, help="Model name")
    parser.add_argument("base_model_path", type=str, help="path to the model without pruning")
    parser.add_argument("pruned_model_path", type=str, help="path to the model with pruning")
    parser.add_argument("--img_sz", type=int, default=224, help="size of input")
    parser.add_argument("--batch_size", type=int, default=32, help="size of the batch")
    parser.add_argument("--base_rate", type=float, default=0.7, help="percentage of pruning")
    parser.add_argument("--nb_classes", type=int, default=4, help="percentage of pruning")
    parser.add_argument("--distill", type=bool, default=True, help="distillation flag")
    parser.add_argument("--drop_path", type=float, default=0, help="drop path rate")

    return parser.parse_args()

class Args:
    def __init__(self):
        self.some_param = 0

def main(args):

    sparse_ratio = [args.base_rate, args.base_rate-0.2, args.base_rate-0.4]
    pruned_model_weight = torch.load(args.pruned_model_path, map_location='cpu', weights_only=False)['model']
    base_model_weight = torch.load(args.base_model_path, map_location='cpu')


    pruned_model_weight = torch.load("prune_model/DeiT-s/DeiT_s_70.pth", map_location='cpu', weights_only=False)['model']

    if args.model == 'convnext-t':

        pruned_model = AdaConvNeXt(sparse_ratio=sparse_ratio,
                pruning_loc=[1,2,3],
                drop_path_rate=args.drop_path,
                layer_scale_init_value=args.layer_scale_init_value,
                head_init_scale=args.head_init_scale,
                num_classes=args.nb_classes)
        base_model = ConvNeXt_Teacher(num_classes=args.nb_classes)

    elif args.model == 'convnext-s':
        pruned_model = AdaConvNeXt(sparse_ratio=sparse_ratio,
                            pruning_loc=[3,6,9],
                            drop_path_rate=args.drop_path,
                            layer_scale_init_value=args.layer_scale_init_value,
                            head_init_scale=args.head_init_scale,
                            num_classes=args.nb_classes,
                            depths=[3, 3, 27, 3])
        base_model = ConvNeXt_Teacher(depths=[3, 3, 27, 3], num_classes=args.nb_classes)

    elif args.model == 'convnext-b':
        pruned_model = AdaConvNeXt(sparse_ratio=sparse_ratio,
                            pruning_loc=[3,6,9],
                            drop_path_rate=args.drop_path,
                            layer_scale_init_value=args.layer_scale_init_value,
                            head_init_scale=args.head_init_scale,
                            num_classes=args.nb_classes,
                            depths=[3, 3, 27, 3],
                            dims=[128, 256, 512, 1024])
        base_model = ConvNeXt_Teacher(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], num_classes=args.nb_classes)

    elif args.model == 'lvvit-s':
        PRUNING_LOC = [4,8,12]
        KEEP_RATE = [sparse_ratio[0], sparse_ratio[0] ** 2, sparse_ratio[0] ** 3]
        print('token_ratio =', KEEP_RATE, 'at layer', PRUNING_LOC)
        pruned_model = LVViTDiffPruning(
            patch_size=16, embed_dim=384, depth=16, num_heads=6, mlp_ratio=3.,
            p_emb='4_2',skip_lam=2., return_dense=True,mix_token=True,
            pruning_loc=PRUNING_LOC, token_ratio=KEEP_RATE, distill=args.distill, num_classes=args.nb_classes
        )
        base_model = LVViT_Teacher(
            patch_size=16, embed_dim=384, depth=16, num_heads=6, mlp_ratio=3.,
            p_emb='4_2',skip_lam=2., return_dense=True,mix_token=True, num_classes=args.nb_classes
        )

    elif args.model == 'lvvit-m':
        PRUNING_LOC = [5,10,15]
        KEEP_RATE = [sparse_ratio[0], sparse_ratio[0] ** 2, sparse_ratio[0] ** 3]
        print('token_ratio =', KEEP_RATE, 'at layer', PRUNING_LOC)
        pruned_model = LVViTDiffPruning(
            patch_size=16, embed_dim=512, depth=20, num_heads=8, mlp_ratio=3.,
            p_emb='4_2',skip_lam=2., return_dense=True,mix_token=True,
            pruning_loc=PRUNING_LOC, token_ratio=KEEP_RATE, distill=args.distill, num_classes=args.nb_classes
        )
        base_model = LVViT_Teacher(
            patch_size=16, embed_dim=512, depth=20, num_heads=8, mlp_ratio=3.,
            p_emb='4_2',skip_lam=2., return_dense=True,mix_token=True, num_classes=args.nb_classes
        )

    elif args.model == 'deit-s':
        PRUNING_LOC = [3, 6, 9]
        KEEP_RATE = [sparse_ratio[0], sparse_ratio[0] ** 2, sparse_ratio[0] ** 3]
        print('token_ratio =', KEEP_RATE, 'at layer', PRUNING_LOC)
        pruned_model = VisionTransformerDiffPruning(
            patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        pruning_loc=PRUNING_LOC, token_ratio=KEEP_RATE, distill=args.distill, num_classes = args.nb_classes
        )
        base_model = VisionTransformerTeacher(
            patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True, num_classes = args.nb_classes)

    elif args.model == 'deit-b' or args.model == 'vit_b_16':
        PRUNING_LOC = [3, 6, 9]
        KEEP_RATE = [sparse_ratio[0], sparse_ratio[0] ** 2, sparse_ratio[0] ** 3]
        print('token_ratio =', KEEP_RATE, 'at layer', PRUNING_LOC)
        pruned_model = VisionTransformerDiffPruning(
            patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        pruning_loc=PRUNING_LOC, token_ratio=KEEP_RATE, distill=args.distill, drop_path_rate=args.drop_path, num_classes = args.nb_classes
        )

        base_model = VisionTransformerTeacher(
            patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True, num_classes = args.nb_classes)

    elif args.model == 'vit_b_32':
        PRUNING_LOC = [3, 6, 9]
        KEEP_RATE = [sparse_ratio[0], sparse_ratio[0] ** 2, sparse_ratio[0] ** 3]
        print('token_ratio =', KEEP_RATE, 'at layer', PRUNING_LOC)
        pruned_model = VisionTransformerDiffPruning(
            patch_size=32, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        pruning_loc=PRUNING_LOC, token_ratio=KEEP_RATE, distill=args.distill, drop_path_rate=args.drop_path, num_classes = args.nb_classes
        )
        base_model = VisionTransformerTeacher(
            patch_size=32, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True, num_classes = args.nb_classes)

    if 'convnext' in args.model or 'deit' in args.model or 'swin' in args.model or 'vit_b' in args.model:
        if 'model' in base_model_weight:
            base_model_weight = base_model_weight['model']
        elif 'state_dict' in base_model_weight:
            base_model_weight = base_model_weight['state_dict']
        base_model_weight = {k.replace('module.', '').replace('model.', ''): v for k, v in base_model_weight.items()}


    base_model.load_state_dict(base_model_weight, strict=False)
    pruned_model.load_state_dict(pruned_model_weight, strict=False)

    base_model.to('cpu')
    pruned_model.to('cpu')

    base_model.eval()
    pruned_model.eval()


    example_input = torch.randn(args.batch_size, 3, args.img_sz, args.img_sz).to("cuda")

    print(f"Benchmark on {args.model} base model\n")
    results_base = benchmark(base_model, example_input, gpu_only=True)
    """
    flops = calc_flops(base_model, img_size=args.img_sz, show_details=False)
    print(f"GFLOPs: {flops} for ratio {args.base_rate}")
    base_model.to('cuda')
    throughput(example_input, base_model)
    """

    print(20*"-")

    print(f"Benchmark on {args.model} pruned model with ratio={args.base_rate}")
    results_pruned = benchmark(pruned_model, example_input, gpu_only=True)
    """
    flops = calc_flops(pruned_model, img_size=args.img_sz, show_details=False, ratios=args.base_rate)
    print(f"GFLOPs: {flops} for ratio {args.base_rate}")
    pruned_model.to('cuda')
    throughput(example_input, pruned_model)
    """

    # save results to a file
    with open(f"results_{args.model}.txt", "w") as f:
        f.write(f"Base Model Results:\n{results_base}\n")
        f.write(f"Pruned Model Results:\n{results_pruned}\n")

if __name__ == "__main__":
    args = parse_args()
    main(args)
