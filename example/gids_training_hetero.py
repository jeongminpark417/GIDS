import argparse, datetime
import dgl
import sklearn.metrics
import torch, torch.nn as nn, torch.optim as optim
import torchmetrics.functional as MF
import time, tqdm, numpy as np
from models import *
from dataloader import IGBHeteroDGLDataset, IGBHeteroDGLDatasetMassive, OGBHeteroDGLDatasetMassive

torch.manual_seed(0)
dgl.seed(0)
import warnings
warnings.filterwarnings("ignore")


def evaluate(model, dataloader):
    predictions = []
    labels = []
    with torch.no_grad():
        for input_nodes, output_nodes, blocks, ret in dataloader:
            blocks = [block.to(device) for block in blocks]
            inputs = blocks[0].srcdata['feat']
            labels.append(blocks[-1].dstdata['label']['paper'].cpu().numpy())
            predictions.append(model(blocks, inputs).argmax(1).cpu().numpy())

        predictions = np.concatenate(predictions)
        labels = np.concatenate(labels)

        acc = sklearn.metrics.accuracy_score(labels, predictions)
        return acc


def track_acc(g, category, args, device):

    #sampler = dgl.dataloading.NeighborSampler([int(fanout) for fanout in args.fan_out.split(',')],
    #        prefetch_node_feats={k: ['feat'] for k in g.ntypes},
    #        prefetch_labels={category: ['label']})
    sampler = dgl.dataloading.NeighborSampler([int(fanout) for fanout in args.fan_out.split(',')])


    train_nid = torch.nonzero(g.nodes[category].data['train_mask'], as_tuple=True)[0]
    val_nid = torch.nonzero(g.nodes[category].data['val_mask'], as_tuple=True)[0]
    test_nid = torch.nonzero(g.nodes[category].data['test_mask'], as_tuple=True)[0]
    
    bam_flag = False
    pin_flag = False
    cpu_cache_flag = False
    uva_flag = False
    wb_flag = False
    uva_graph = False
    if(args.bam == 1):
        bam_flag = True
    if(args.pin == 1):
        pin_flag = True
    if(args.cpu_cache == 1):
        cpu_cache_flag = True
    if(args.uva == 1):
        uva_flag = True
    if(args.window == 1):
        wb_flag = True
    if(args.uva_graph == 1):
        uva_graph = True
    
    dim = 1024
    if(args.data == "IGB"):
        dim = 1024
    elif(args.data == "OGB"):
        dim = 768


    train_dataloader = dgl.dataloading.DataLoader(
        g, {category: train_nid}, sampler,
        batch_size=args.batch_size,
        shuffle=True, drop_last=False,
        num_workers=args.num_workers,
        bam=bam_flag,
        feature_dim=dim,
        use_uva=uva_flag,
        use_uva_graph=uva_graph,
        #device=device,
       # cpu_cache=cpu_cache_flag, 
       # cpu_cache_size=int(6 * 1024 * 1024 / 4),
        use_prefetch_thread=False,
        pin_prefetcher=False,
        window_buffer=wb_flag,
        window_buffer_size=args.wb_size
        )

    val_dataloader = dgl.dataloading.DataLoader(
        g, {category: val_nid}, sampler,
        batch_size=args.batch_size,
        shuffle=False, drop_last=False,
        num_workers=args.num_workers)

    test_dataloader = dgl.dataloading.DataLoader(
        g, {category: test_nid}, sampler,
        batch_size=args.batch_size,
        shuffle=True, drop_last=False,
        num_workers=args.num_workers)
    
    if(bam_flag):
        if(args.data == "IGB"):
            train_dataloader.bam_init(0,4096,1024,300*1000*1000*1024, args.cache_size, args.num_ssd)
        elif(args.data == "OGB"):
            train_dataloader.bam_init(0,4096,1024, 270000000 * 1024 , args.cache_size, args.num_ssd)


    in_feats = g.ndata['feat'][category].shape[1]

    if args.model_type == 'rgcn':
        model = RGCN(g.etypes, in_feats, args.hidden_channels,
            args.num_classes, args.num_layers).to(device)
    if args.model_type == 'rsage':
        model = RSAGE(g.etypes, in_feats, args.hidden_channels,
            args.num_classes, args.num_layers).to(device)
    if args.model_type == 'rgat':
        model = RGAT(g.etypes, in_feats, args.hidden_channels,
            args.num_classes, args.num_layers, args.num_heads).to(device)

    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    print('model size: {:.3f}MB'.format(size_all_mb))

    loss_fcn = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(),
        lr=args.learning_rate)
    sched = optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.25)

    best_accuracy = 0
    training_start = time.time()
    for epoch in tqdm.tqdm(range(args.epochs)):
        model.train()
        total_loss = 0
        train_acc = 0
        idx = 0
        gpu_mem_alloc = 0
        epoch_start = time.time()
        batch_input_time = 0
        train_time = 0
        transfer_time = 0

        for it, (input_nodes, output_nodes, blocks, ret) in enumerate(train_dataloader):
            #print("blocks: ", blocks[0])
            #print("idx: ", idx)            
            if(idx == 100):
                print("warp up done")
                if(bam_flag):
                    train_dataloader.print_stats()
                train_dataloader.print_timer()
                if(bam_flag == False):
                    print("feature aggregation time: ", batch_input_time)
                print("transfer time: ", transfer_time)
                print("train time: ", train_time)
                batch_input_time = 0
                transfer_time = 0
                train_time = 0
            idx += 1
            transfer_start = time.time()
            blocks = [block.to(device) for block in blocks]
            cur_trans_time = time.time()  -transfer_start
            transfer_time += cur_trans_time
          #  print("trans time: ", cur_trans_time) 

            batch_input_start = time.time()
            if(bam_flag): 
                batch_inputs = ret
            else:
                batch_inputs = blocks[0].srcdata['feat']

            agg_time = (time.time() - batch_input_start)
            batch_input_time += agg_time

            #x = blocks[0].srcdata['feat']
           # print("batch input: ", batch_inputs)
            y = blocks[-1].dstdata['label']['paper']
            train_start = time.time()
            y_hat = model(blocks, batch_inputs)
            loss = loss_fcn(y_hat, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            cur_train_time = time.time() - train_start
            train_time += cur_train_time
           # print("train time: ", cur_train_time)

            train_acc += sklearn.metrics.accuracy_score(y.cpu().numpy(),
                y_hat.argmax(1).detach().cpu().numpy())*100
            gpu_mem_alloc += (
                torch.cuda.max_memory_allocated() / 1000000
                if torch.cuda.is_available()
                else 0
            )
            if(idx == 1100):
                print("1dx 1100")
                if(bam_flag):
                    train_dataloader.print_stats()
                train_dataloader.print_timer()
                if(bam_flag == False):
                    print("feature aggregation time: ", batch_input_time)
                print("transfer time: ", transfer_time)
                print("train time: ", train_time)
                batch_input_time = 0
                transfer_time = 0
                train_time = 0
                return None

        
        if(bam_flag):
            train_dataloader.print_stats()
        train_dataloader.print_timer()

        train_acc /= idx
        gpu_mem_alloc /= idx

        if epoch%args.log_every == 0:
            model.eval()
            val_acc = evaluate(model, val_dataloader).item()*100
            if best_accuracy < val_acc:
                best_accuracy = val_acc
                if args.model_save:
                    torch.save(model.state_dict(), args.modelpath)

            tqdm.tqdm.write(
                "Epoch {:03d} | Aggregation {:.4f} | Transfer Time {:.4f} | Train Time {:.4f} |  GPU {:.1f} MB".format(
                    epoch,
                    batch_input_time,
                    transfer_time,
                    train_time,
                    gpu_mem_alloc
                )
            )
        sched.step()

        model.eval()
    test_acc = evaluate(model, test_dataloader).item()*100
    print("Test Acc {:.2f}%".format(test_acc))
    print("Total time taken " + str(datetime.timedelta(seconds = int(time.time() - training_start))))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Loading dataset
    parser.add_argument('--path', type=str, default='/root/gnndataset',
        help='path containing the datasets')
    parser.add_argument('--dataset_size', type=str, default='tiny',
        choices=['tiny', 'small', 'medium', 'large', 'full', 'OGB'],
        help='size of the datasets')
    parser.add_argument('--num_classes', type=int, default=19,
        choices=[19, 2983, 348,349, 350, 153, 152], help='number of classes')
    parser.add_argument('--in_memory', type=int, default=0,
        choices=[0, 1], help='0:read only mmap_mode=r, 1:load into memory')
    parser.add_argument('--synthetic', type=int, default=0,
        choices=[0, 1], help='0:nlp-node embeddings, 1:random')

    # Model
    parser.add_argument('--model_type', type=str, default='rgat',
                        choices=['rgat', 'rsage', 'rgcn'])
    parser.add_argument('--modelpath', type=str, default='deletethis.pt')
    parser.add_argument('--model_save', type=int, default=0)

    # Model parameters
    parser.add_argument('--fan_out', type=str, default='10,15')
    parser.add_argument('--batch_size', type=int, default=10240)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--hidden_channels', type=int, default=128)
    parser.add_argument('--learning_rate', type=int, default=0.01)
    parser.add_argument('--decay', type=int, default=0.001)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--num_layers', type=int, default=6)
    parser.add_argument('--num_heads', type=int, default=4)

    parser.add_argument('--bam', type=int, default=0)
    parser.add_argument('--num_ssd', type=int, default=1)

    parser.add_argument('--cache_size', type=int, default=10)
    parser.add_argument('--pin', type=int, default=0)
    parser.add_argument('--cpu_cache', type=int, default=0)
    parser.add_argument('--uva', type=int, default=0)
    parser.add_argument('--uva_graph', type=int, default=0)
    parser.add_argument('--window', type=int, default=0)
    parser.add_argument('--wb_size', type=int, default=6)
 
    parser.add_argument('--log_every', type=int, default=5)
    parser.add_argument('--device', type=int, default=0)

    parser.add_argument('--data', type=str, default='IGB')
 
    args = parser.parse_args()

    device = f'cuda:'+str(args.device) if torch.cuda.is_available() else 'cpu'

    if(args.data =='OGB'):
        dataset = OGBHeteroDGLDatasetMassive(args)

    else:
        if(args.dataset_size == 'full' or args.dataset_size == 'large'):
            print("large dataset")
            dataset = IGBHeteroDGLDatasetMassive(args)
        else:
            dataset = IGBHeteroDGLDataset(args)
    g = dataset[0]
    print("format: ", g.formats())
    g  = g.formats('csc')
    print("format: ", g.formats())
    
    category = g.predict
    print("g: ", g)
    track_acc(g, category, args, device)
