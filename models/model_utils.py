import torch
from models.xdfm import CINModel
from models.armnet import ARMNetModel
from models.dnn import DNNModel

def create_model(args, logger):
    logger.info(f'=> creating model {args.model}')
    if args.model == 'arm':
        model = ARMNetModel(args.nfield, args.nfeat, args.nemb, args.alpha, args.nhid,
                            args.mlp_nlayer, args.mlp_nhid, args.dropout, args.ensemble,
                            args.emb_ncross)
    elif args.model == 'dnn':
        model = DNNModel(args.nfield, args.nfeat, args.nemb, args.mlp_nlayer, args.mlp_nhid, args.dropout,
                          args.emb_ncross)
    else:
        raise ValueError(f'unknown model {args.model}')

    if torch.cuda.is_available(): model = model.cuda()
    logger.info(f'model parameters: {sum([p.data.nelement() for p in model.parameters()])}')
    return model
