"""
Convert our saved models into torchvision resnet-50 style
"""
import torch
import argparse
import torchvision


def load_base(ckpt_path, encoder):
    ckpt = torch.load(ckpt_path,
                      map_location=lambda storage, loc: storage)['state_dict']
    new_state = {}
    for k, v in ckpt.items():
        if 'feature.' in k:
            new_state[k.replace('feature.', '')] = v
    encoder.load_state_dict(new_state, strict=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",
                        "-i",
                        type=str,
                        default=None,
                        help="input ckpt")
    parser.add_argument("--output",
                        "-o",
                        type=str,
                        default="tv_resnet.ckpt",
                        help="output path")
    parser.add_argument("--backbone", type=str, default="resnet50")

    args = parser.parse_args()

    resnet_head = getattr(torchvision.models, args.backbone)()
    encoder = torch.nn.Sequential(*list(resnet_head.children())[:-1])
    load_base(args.input, encoder)

    torch.save(resnet_head, args.output)