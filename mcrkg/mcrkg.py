import random

try:
    from transformers import (ConstantLRSchedule, WarmupLinearSchedule, WarmupConstantSchedule)
except:
    from transformers import get_constant_schedule, get_constant_schedule_with_warmup,  get_linear_schedule_with_warmup

from modeling.modeling_gnn import *

from utils.optimization_utils import OPTIMIZER_CLASSES
from utils.parser_utils import *


DECODER_DEFAULT_LR = {
    'ele_data2': 1e-3
}

from collections import defaultdict, OrderedDict
import numpy as np

import socket, os, subprocess, datetime
print(socket.gethostname())
print ("pid:", os.getpid())
print ("conda env:", os.environ['CONDA_DEFAULT_ENV'])
print ("screen: %s" % subprocess.check_output('echo $STY', shell=True).decode('utf'))
print ("gpu: %s" % subprocess.check_output('echo $CUDA_VISIBLE_DEVICES', shell=True).decode('utf'))


def evaluate_accuracy(eval_set, model):
    n_samples, n_correct = 0, 0
    model.eval()
    with torch.no_grad():
        for qids, labels, *input_data in tqdm(eval_set):
            logits, _ = model(*input_data)
            with open("xxx", 'w+') as f:
                f.write(f'{logits.shape}，{labels.shape}')
            logits = logits.reshape(labels.shape[0], -1)
            n_correct += (logits.argmax(1) == labels).sum().item()
            n_samples += labels.size(0)
    return n_correct / n_samples


def main():
    parser = get_parser()
    args, _ = parser.parse_known_args()
    parser.add_argument('--mode', default='train', choices=['train', 'eval_detail'], help='run training or evaluation')
    parser.add_argument('--save_dir', default=f'./saved_models/ele_data2/', help='model output directory')
    parser.add_argument('--save_model', dest='save_model', action='store_true')
    parser.add_argument('--load_model_path', default=None)
    parser.add_argument('--model_path', default=None)


    # data
    parser.add_argument('--num_relation', default=38, type=int, help='number of relations')
    parser.add_argument('--train_adj', default=f'data/{args.dataset}/graph/train.graph.adj.pk')
    parser.add_argument('--dev_adj', default=f'data/{args.dataset}/graph/dev.graph.adj.pk')
    parser.add_argument('--test_adj', default=f'data/{args.dataset}/graph/test.graph.adj.pk')
    parser.add_argument('--use_cache', default=False, type=bool_flag, nargs='?', const=False, help='use cached data to accelerate data loading')

    # model architecture
    parser.add_argument('-k', '--k', default=5, type=int, help='perform k-layer message passing')
    parser.add_argument('--att_head_num', default=2, type=int, help='number of attention heads')
    parser.add_argument('--gnn_dim', default=100, type=int, help='dimension of the GNN layers')
    parser.add_argument('--fc_dim', default=200, type=int, help='number of FC hidden units')
    parser.add_argument('--fc_layer_num', default=0, type=int, help='number of FC layers')
    parser.add_argument('--freeze_ent_emb', default=True, type=bool_flag, nargs='?', const=True, help='freeze entity embedding layer')

    parser.add_argument('--max_node_num', default=200, type=int)
    parser.add_argument('--simple', default=False, type=bool_flag, nargs='?', const=True)
    parser.add_argument('--subsample', default=1.0, type=float)
    parser.add_argument('--init_range', default=0.02, type=float, help='stddev when initializing with normal distribution')


    # regularization
    parser.add_argument('--dropouti', type=float, default=0.2, help='dropout for embedding layer')
    parser.add_argument('--dropoutg', type=float, default=0.2, help='dropout for GNN layers')
    parser.add_argument('--dropoutf', type=float, default=0.2, help='dropout for fully-connected layers')

    # optimization
    parser.add_argument('-dlr', '--decoder_lr', default=DECODER_DEFAULT_LR[args.dataset], type=float, help='learning rate')
    parser.add_argument('-mbs', '--mini_batch_size', default=1, type=int)
    parser.add_argument('-ebs', '--eval_batch_size', default=2, type=int)
    parser.add_argument('--unfreeze_epoch', default=4, type=int)
    parser.add_argument('--refreeze_epoch', default=10000, type=int)
    parser.add_argument('--fp16', default=False, type=bool_flag, help='use fp16 training. this requires torch>=1.6.0')
    parser.add_argument('--drop_partial_batch', default=False, type=bool_flag, help='')
    parser.add_argument('--fill_partial_batch', default=False, type=bool_flag, help='')

    parser.add_argument('--case_db_path', type=str, required=True)
    parser.add_argument('--pool_type', choices=['mean', 'max', 'cls'], default='mean')
    parser.add_argument('--num_pos', type=int, default=5)
    parser.add_argument('--num_neg', type=int, default=5)
    parser.add_argument('--num_layers', type=int, default=4, help="Number of layers")
    parser.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS, help='show this help message and exit')

    args = parser.parse_args()
    if args.simple:
        parser.set_defaults(k=1)
    args = parser.parse_args()
    args.fp16 = args.fp16 and (torch.__version__ >= '1.6.0')

    if args.mode == 'train':
        train(args)
    elif args.mode == 'eval_detail':
        # raise NotImplementedError
        eval_detail(args)
    else:
        raise ValueError('Invalid mode')




def train(args):
    print(args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available() and args.cuda:
        torch.cuda.manual_seed(args.seed)

    config_path = os.path.join(args.save_dir, 'config.json')
    model_path = os.path.join(args.save_dir, 'model.pt')
    log_path = os.path.join(args.save_dir, 'log.csv')
    export_config(args, config_path)
    check_path(model_path)
    with open(log_path, 'w') as fout:
        fout.write('step,dev_acc,test_acc\n')

    ###################################################################################################
    #   Load data                                                                                     #
    ###################################################################################################
    cp_emb = [np.load(path) for path in args.ent_emb_paths]
    cp_emb = torch.tensor(np.concatenate(cp_emb, 1), dtype=torch.float)

    concept_num, concept_dim = cp_emb.size(0), cp_emb.size(1)
    print('| num_concepts: {} |'.format(concept_num))

    # try:
    if True:
        if torch.cuda.device_count() >= 2 and args.cuda:
            device0 = torch.device("cuda:0")
            device1 = torch.device("cuda:1")
        elif torch.cuda.device_count() == 1 and args.cuda:
            device0 = torch.device("cuda:0")
            device1 = torch.device("cuda:0")
        else:
            device0 = torch.device("cpu")
            device1 = torch.device("cpu")
        dataset = LM_GNN_DataLoader(args, args.train_statements, args.train_adj,
                                               args.dev_statements, args.dev_adj,
                                               args.test_statements, args.test_adj,
                                               batch_size=args.batch_size, eval_batch_size=args.eval_batch_size,
                                               device=(device0, device1),
                                               model_name=args.encoder,
                                               model_path=args.model_path,
                                               max_node_num=args.max_node_num, max_seq_length=args.max_seq_len,
                                               is_inhouse=args.inhouse, inhouse_train_qids_path=args.inhouse_train_qids,
                                               subsample=args.subsample, use_cache=args.use_cache)

        ###################################################################################################
        #   Build model                                                                                   #
        ###################################################################################################
        print ('args.num_relation', args.num_relation)
        model = LM_GNN(args, args.encoder, args.model_path, k=args.k, n_ntype=4, n_etype=args.num_relation, n_concept=concept_num,
                                   concept_dim=args.gnn_dim,
                                   concept_in_dim=concept_dim,
                                   n_attention_head=args.att_head_num, fc_dim=args.fc_dim, n_fc_layer=args.fc_layer_num,
                                   p_emb=args.dropouti, p_gnn=args.dropoutg, p_fc=args.dropoutf,
                                   pretrained_concept_emb=cp_emb, freeze_ent_emb=args.freeze_ent_emb,
                                   init_range=args.init_range,
                                   encoder_config={})
        if args.load_model_path: # and args.mode=='eval_detail':
            print (f'loading and initializing model from {args.load_model_path}')
            # breakpoint()
            model_state_dict, old_args = torch.load(args.load_model_path , map_location=torch.device('cuda:0'))

            model.load_state_dict(model_state_dict)

        model.encoder.to(device0)
        model.decoder.to(device1)


    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    grouped_parameters = [
        {'params': [p for n, p in model.encoder.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay, 'lr': args.encoder_lr},
        {'params': [p for n, p in model.encoder.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0, 'lr': args.encoder_lr},
        {'params': [p for n, p in model.decoder.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay, 'lr': args.decoder_lr},
        {'params': [p for n, p in model.decoder.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0, 'lr': args.decoder_lr},
    ]
    optimizer = OPTIMIZER_CLASSES[args.optim](grouped_parameters)

    if args.lr_schedule == 'fixed':
        try:
            scheduler = ConstantLRSchedule(optimizer)
        except:
            scheduler = get_constant_schedule(optimizer)
    elif args.lr_schedule == 'warmup_constant':
        try:
            scheduler = WarmupConstantSchedule(optimizer, warmup_steps=args.warmup_steps)
        except:
            scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps)
    elif args.lr_schedule == 'warmup_linear':
        max_steps = int(args.n_epochs * (dataset.train_size() / args.batch_size))
        try:
            scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=max_steps)
        except:
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=max_steps)

    print('parameters:')
    for name, param in model.decoder.named_parameters():
        if param.requires_grad:
            print('\t{:45}\ttrainable\t{}\tdevice:{}'.format(name, param.size(), param.device))
        else:
            print('\t{:45}\tfixed\t{}\tdevice:{}'.format(name, param.size(), param.device))
    num_params = sum(p.numel() for p in model.decoder.parameters() if p.requires_grad)
    print('\ttotal:', num_params)

    if args.loss == 'margin_rank':
        loss_func = nn.MarginRankingLoss(margin=0.1, reduction='mean')
    elif args.loss == 'cross_entropy':
        loss_func = nn.CrossEntropyLoss(reduction='mean')

    def compute_contrastive_loss(model, orig_output, pos_graphs, neg_graphs, device):
        pos_outputs = []
        for p_graph in pos_graphs:
            p_inputs = convert_graph_to_inputs(p_graph, device)
            with torch.no_grad():
                p_out = model(*p_inputs, contrastive=True)
                pos_outputs.append(p_out)


        neg_outputs = []
        for n_graph in neg_graphs:
            n_inputs = convert_graph_to_inputs(n_graph, device)
            with torch.no_grad():
                n_out = model(*n_inputs, contrastive=True)
                neg_outputs.append(n_out)


        node_loss = model.decoder._node_contrast(
            orig_output['node_vecs'],
            [p['node_vecs'] for p in pos_outputs],
            [n['node_vecs'] for n in neg_outputs]
        )


        graph_loss = model.decoder._graph_contrast(
            orig_output['graph_vecs'],
            torch.cat([p['graph_vecs'] for p in pos_outputs]),
            torch.cat([n['graph_vecs'] for n in neg_outputs])
        )

        return node_loss, graph_loss

    def convert_graph_to_inputs(graph_data, device):

        return (
            torch.tensor(graph_data['concept_ids']).to(device),
            torch.tensor(graph_data['node_type_ids']).to(device),
            torch.tensor(graph_data['node_scores']).to(device),
            torch.tensor(graph_data['adj_lengths']).to(device),
            (torch.tensor(graph_data['edge_index']).to(device),
             torch.tensor(graph_data['edge_type']).to(device))
        )

    def compute_loss(logits, labels, model, batch_data, args, device):
        ce_loss = F.cross_entropy(logits, labels)


        qids, labels, *input_data = batch_data
        input_texts = [model.train_qid2text[qid] for qid in qids] 

        total_node_loss = 0.0
        total_graph_loss = 0.0
        valid_contrast_count = 0


        for i in range(len(input_texts)):
            pos_cases, neg_cases = model.decoder.case_retriever.retrieve(
                input_texts[i],
                labels[i].item(),
                topk=args.num_pos
            )

            if len(pos_cases) == 0 and len(neg_cases) == 0:
                continue


            orig_inputs = [x[i:i + 1] for x in input_data]
            orig_output = model(*orig_inputs, contrastive=True)


            node_loss, graph_loss = compute_contrastive_loss(
                model,
                orig_output,
                [c['graph_data'] for c in pos_cases],
                [c['graph_data'] for c in neg_cases],
                device
            )

            total_node_loss += node_loss
            total_graph_loss += graph_loss
            valid_contrast_count += 1

        if valid_contrast_count > 0:
            total_node_loss /= valid_contrast_count
            total_graph_loss /= valid_contrast_count
            contrast_loss = total_node_loss + total_graph_loss
        else:
            contrast_loss = 0.0

        total_loss = ce_loss + contrast_loss

        # return total_loss, {'ce': ce_loss.item(), 'node': total_node_loss.item(), 'graph': total_graph_loss.item()}
        return total_loss, 1
    ###################################################################################################
    #   Training                                                                                      #
    ###################################################################################################

    print()
    print('-' * 71)
    if args.fp16:
        print ('Using fp16 training')
        scaler = torch.cuda.amp.GradScaler()

    global_step, best_dev_epoch = 0, 0
    best_dev_acc, final_test_acc, total_loss = 0.0, 0.0, 0.0
    start_time = time.time()
    model.train()
    freeze_net(model.encoder)
    if True:
    # try:
        for epoch_id in range(args.n_epochs):
            if epoch_id == args.unfreeze_epoch:
                unfreeze_net(model.encoder)
            if epoch_id == args.refreeze_epoch:
                freeze_net(model.encoder)
            model.train()
            for qids, labels, *input_data in dataset.train():
                optimizer.zero_grad()
                bs = labels.size(0)
                for a in range(0, bs, args.mini_batch_size):
                    b = min(a + args.mini_batch_size, bs)
                    if args.fp16:
                        with torch.cuda.amp.autocast():
                            logits, _ = model(*[x[a:b] for x in input_data], layer_id=args.encoder_layer)
                            loss = compute_loss(logits, labels[a:b])
                    else:
                        logits, _ = model(*[x[a:b] for x in input_data], layer_id=args.encoder_layer)
                        logits = logits.view(b-a, input_data[0].shape[1])
                        with open("xxx", 'w+') as f:
                            f.write(f'{logits.shape[0]}, {a}, {b}')
                        loss, loss_dict = compute_loss(
                            logits,
                            labels[a:b],
                            model,
                            (qids, labels, *input_data),
                            args,
                            logits.device
                        )
                    loss = loss * (b - a) / bs
                    if args.fp16:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()
                    total_loss += loss.item()
                if args.max_grad_norm > 0:
                    if args.fp16:
                        scaler.unscale_(optimizer)
                        nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    else:
                        nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scheduler.step()
                if args.fp16:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                if (global_step + 1) % args.log_interval == 0:
                    total_loss /= args.log_interval
                    ms_per_batch = 1000 * (time.time() - start_time) / args.log_interval
                    print('| step {:5} |  lr: {:9.7f} | loss {:7.4f} | ms/batch {:7.2f} |'.format(global_step, scheduler.get_lr()[0], total_loss, ms_per_batch))
                    total_loss = 0
                    start_time = time.time()
                global_step += 1

            model.eval()
            dev_acc = evaluate_accuracy(dataset.dev(), model)
            save_test_preds = args.save_model
            # breakpoint()
            if not save_test_preds:
                test_acc = evaluate_accuracy(dataset.test(), model) if args.test_statements else 0.0
            else:
                eval_set = dataset.test()
                total_acc = []
                count = 0
                preds_path = os.path.join(args.save_dir, 'test_e{}_preds.csv'.format(epoch_id))
                with open(preds_path, 'w') as f_preds:
                    with torch.no_grad():
                        for qids, labels, *input_data in tqdm(eval_set):
                            count += 1
                            logits, _, concept_ids, node_type_ids, edge_index, edge_type = model(*input_data, detail=True)
                            predictions = logits.argmax(1) #[bsize, ]
                            preds_ranked = (-logits).argsort(1) #[bsize, n_choices]
                            for i, (qid, label, pred, _preds_ranked, cids, ntype, edges, etype) in enumerate(zip(qids, labels, predictions, preds_ranked, concept_ids, node_type_ids, edge_index, edge_type)):
                                acc = int(pred.item()==label.item())
                                print ('{},{}'.format(qid, chr(ord('A') + pred.item())), file=f_preds)
                                f_preds.flush()
                                total_acc.append(acc)
                test_acc = float(sum(total_acc))/(len(total_acc)+0.00001)

            print('-' * 71)
            print('| epoch {:3} | step {:5} | dev_acc {:7.4f} | test_acc {:7.4f} |'.format(epoch_id, global_step, dev_acc, test_acc))
            print('-' * 71)
            with open(log_path, 'a') as fout:
                fout.write('{},{},{}\n'.format(global_step, dev_acc, test_acc))
            if dev_acc >= best_dev_acc:
                best_dev_acc = dev_acc
                final_test_acc = test_acc
                best_dev_epoch = epoch_id
                if args.save_model:
                    torch.save([model.state_dict(), args], f"{model_path}.{epoch_id}")
                    # with open(model_path +".{}.log.txt".format(epoch_id), 'w') as f:
                    #     for p in model.named_parameters():
                    #         print (p, file=f)
                    print(f'model saved to {model_path}.{epoch_id}')
            else:
                if args.save_model:
                    torch.save([model.state_dict(), args], f"{model_path}.{epoch_id}")
                    # with open(model_path +".{}.log.txt".format(epoch_id), 'w') as f:
                    #     for p in model.named_parameters():
                    #         print (p, file=f)
                    print(f'model saved to {model_path}.{epoch_id}')
            model.train()
            start_time = time.time()
            if epoch_id > args.unfreeze_epoch and epoch_id - best_dev_epoch >= args.max_epochs_before_stop:
                break
    # except (KeyboardInterrupt, RuntimeError) as e:
    #     print(e)



def eval_detail(args):
    assert args.load_model_path is not None
    model_path = args.load_model_path

    cp_emb = [np.load(path) for path in args.ent_emb_paths]
    cp_emb = torch.tensor(np.concatenate(cp_emb, 1), dtype=torch.float)
    concept_num, concept_dim = cp_emb.size(0), cp_emb.size(1)
    print('| num_concepts: {} |'.format(concept_num))

    model_state_dict, old_args = torch.load(model_path, map_location=torch.device('cuda:0'))
    model = LM_GNN(old_args, old_args.encoder, model_path, k=old_args.k, n_ntype=4, n_etype=old_args.num_relation, n_concept=concept_num,
                               concept_dim=old_args.gnn_dim,
                               concept_in_dim=concept_dim,
                               n_attention_head=old_args.att_head_num, fc_dim=old_args.fc_dim, n_fc_layer=old_args.fc_layer_num,
                               p_emb=old_args.dropouti, p_gnn=old_args.dropoutg, p_fc=old_args.dropoutf,
                               pretrained_concept_emb=cp_emb, freeze_ent_emb=old_args.freeze_ent_emb,
                               init_range=old_args.init_range,
                               encoder_config={})
    model.load_state_dict(model_state_dict)

    if torch.cuda.device_count() >= 2 and args.cuda:
        device0 = torch.device("cuda:0")
        device1 = torch.device("cuda:1")
    elif torch.cuda.device_count() == 1 and args.cuda:
        device0 = torch.device("cuda:0")
        device1 = torch.device("cuda:0")
    else:
        device0 = torch.device("cpu")
        device1 = torch.device("cpu")
    model.encoder.to(device0)
    model.decoder.to(device1)
    model.eval()

    statement_dic = {}
    for statement_path in (args.train_statements, args.dev_statements, args.test_statements):
        statement_dic.update(load_statement_dict(statement_path))

    use_contextualized = 'lm' in old_args.ent_emb

    print ('inhouse?', args.inhouse)

    print ('args.train_statements', args.train_statements)
    print ('args.dev_statements', args.dev_statements)
    print ('args.test_statements', args.test_statements)
    print ('args.train_adj', args.train_adj)
    print ('args.dev_adj', args.dev_adj)
    print ('args.test_adj', args.test_adj)

    dataset = LM_GNN_DataLoader(args, args.train_statements, args.train_adj,
                                           args.dev_statements, args.dev_adj,
                                           args.test_statements, args.test_adj,
                                           batch_size=args.batch_size, eval_batch_size=args.eval_batch_size,
                                           device=(device0, device1),
                                           model_name=old_args.encoder,
                                           max_node_num=old_args.max_node_num, max_seq_length=old_args.max_seq_len,
                                           is_inhouse=args.inhouse, inhouse_train_qids_path=args.inhouse_train_qids,
                                           subsample=args.subsample, use_cache=args.use_cache)

    save_test_preds = args.save_model
    dev_acc = evaluate_accuracy(dataset.dev(), model)
    print('dev_acc {:7.4f}'.format(dev_acc))
    if not save_test_preds:
        test_acc = evaluate_accuracy(dataset.test(), model) if args.test_statements else 0.0
    else:
        eval_set = dataset.test()
        total_acc = []
        count = 0
        dt = datetime.datetime.today().strftime('%Y%m%d%H%M%S')
        preds_path = os.path.join(args.save_dir, 'test_preds_{}.csv'.format(dt))
        with open(preds_path, 'w') as f_preds:
            with torch.no_grad():
                for qids, labels, *input_data in tqdm(eval_set):
                    count += 1
                    logits, _, concept_ids, node_type_ids, edge_index, edge_type = model(*input_data, detail=True)
                    predictions = logits.argmax(1) #[bsize, ]
                    preds_ranked = (-logits).argsort(1) #[bsize, n_choices]
                    for i, (qid, label, pred, _preds_ranked, cids, ntype, edges, etype) in enumerate(zip(qids, labels, predictions, preds_ranked, concept_ids, node_type_ids, edge_index, edge_type)):
                        acc = int(pred.item()==label.item())
                        print ('{},{}'.format(qid, chr(ord('A') + pred.item())), file=f_preds)
                        f_preds.flush()
                        total_acc.append(acc)
        test_acc = float(sum(total_acc))/len(total_acc)

        print('-' * 71)
        print('test_acc {:7.4f}'.format(test_acc))
        print('-' * 71)



if __name__ == '__main__':
    main()
