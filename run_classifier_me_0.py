import argparse
import numpy as np
import torch
import random
import logging
import os
os.environ['CUDA_VISIBLE_DEVICES']='2'
from tqdm import tqdm
from sklearn.metrics import accuracy_score,classification_report,matthews_corrcoef
from torch.nn import CrossEntropyLoss
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,TensorDataset)
from me.tokenization import BertTokenizer
from me.file_utils import PYTORCH_PRETRAINED_BERT_CACHE,WEIGHTS_NAME,CONFIG_NAME
from me.modeling import BertForSequenceClassification
from me.optimization import BertAdam
from me.dataProcess import convert_examples_to_features,ColaProcessor

logger = logging.getLogger(__name__)


def compute_metrics(preds, labels):
    assert len(preds) == len(labels)
    return {"mcc": matthews_corrcoef(labels, preds)},classification_report(labels,preds),accuracy_score(labels,preds)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="trainData/NLPCC2017", type=str, required=False)
    parser.add_argument("--bert_model", default="bert-model-pretrain/bert-base-chinese", type=str, required=False)
    parser.add_argument("--task_name", default="COLA", type=str, required=False, help="The name of the task to train.")
    parser.add_argument("--output_dir", default="output1", type=str, required=False)
    ## Other parameters
    parser.add_argument("--do_train", default=True, action='store_true')
    parser.add_argument("--do_eval", default=True, action='store_true')
    parser.add_argument("--do_lower_case", default=True, action='store_true')
    parser.add_argument("--max_seq_length", default=64, type=int)
    parser.add_argument("--train_batch_size", default=32, type=int)
    parser.add_argument("--eval_batch_size", default=8, type=int)
    parser.add_argument("--learning_rate", default=2e-5, type=float)
    parser.add_argument("--num_train_epochs", default=5.0, type=float)
    ## Other parameters
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda", action='store_true', help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16', action='store_true', help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale', type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n 0 (default value): dynamic loss scaling.\nPositive power of 2: static loss scaling value.\n")
    parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--gpu_number',type=int,default=1)

    args = parser.parse_args()

    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO if args.local_rank in [-1, 0] else logging.WARN)

    # proccessors = {"cola":ColaProcessor}
    # output_modds = {"cola":"classification"}

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.gpu_number > 0:
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")
    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)


    processor = ColaProcessor()

    #处理数据 标签  标签数量  读取数据    循环次数
    label_list = processor.get_labels()
    num_labels = len(label_list)

    tokenizer = BertTokenizer.from_pretrained(args.bert_model,do_lower_case=args.do_lower_case)
    train_example = None
    num_train_optimization_steps = None
    if args.do_train:
        train_example = processor.get_train_examples(args.data_dir)
        num_train_optimization_steps = int(len(train_example) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs

    #加载模型
    cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed_{}'.format(args.local_rank))
    model = BertForSequenceClassification.from_pretrained(args.bert_model,cache_dir=cache_dir,num_labels=num_labels,in_channels=args.max_seq_length)
    model.to(device)
    # model = torch.nn.DataParallel(model)

    #训练参数和优化器
    if args.do_train:
        param_optimier = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params':[p for n,p in param_optimier if not any(nd in n for nd in no_decay)],'weight_decay':0.01},
            {'params':[p for n,p in param_optimier if any(nd in n for nd in no_decay)],'weight_decay':0.00},
        ]
        optimizer = BertAdam(params=optimizer_grouped_parameters,lr=args.learning_rate,warmup=args.warmup_proportion,t_total=num_train_optimization_steps)


    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0
    #包装数据 pyTorch 加载器
    if args.do_train:
        train_features = convert_examples_to_features(examples=train_example,label_list=label_list,max_seq_length=args.max_seq_length,tokenizer=tokenizer)

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_example))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)
        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
        #包装数据
        train_data = TensorDataset(all_input_ids,all_input_mask,all_segment_ids,all_label_ids)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(dataset=train_data, sampler=train_sampler, batch_size=args.train_batch_size,num_workers=8)
        eval_examples = processor.get_dev_examples(args.data_dir)
        eval_features = convert_examples_to_features(eval_examples, label_list, args.max_seq_length, tokenizer)
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)
        all_input_ids1 = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask1 = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids1 = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        all_label_ids1 = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
        eval_data = TensorDataset(all_input_ids1, all_input_mask1, all_segment_ids1, all_label_ids1)
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(dataset=eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size,num_workers=8)

        #开始训练
        model.train()
        for epoch in range(int(args.num_train_epochs)):
            tr_loss = 0.0
            nb_tr_examples, nb_tr_steps = 0, 0

            for step,batch in enumerate(train_dataloader):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                logits = model(input_ids,segment_ids,input_mask,labels=None) #32*18
                #计算损失
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1,num_labels),label_ids.view(-1))
                # if n_gpu > 1:
                #     loss = loss.mean() # mean() to average on multi-gpu.
                # if args.gradient_accumulation_steps > 1:
                #     loss = loss / args.gradient_accumulation_steps
                loss.backward()
                #统计
                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1

                if (step+1) % args.gradient_accumulation_steps ==0:
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

                #测试
                if step % 1 ==0:
                    print('epoch:{:10}  step {:10}   tr_loss: {:.3f}   train_loss: {:.3f} '.format(epoch,step,tr_loss/nb_tr_steps,loss.item()))
                if (step+1) % 10 == 0:
                    preds = []
                    for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader, desc="Evaluating"):
                        input_ids = input_ids.to(device)
                        input_mask = input_mask.to(device)
                        segment_ids = segment_ids.to(device)
                        label_ids = label_ids.to(device)
                        with torch.no_grad():
                            logits = model(input_ids, segment_ids, input_mask, labels=None)
                        # create eval loss and other metric required by the task
                        if len(preds) == 0:
                            preds.append(logits.detach().cpu().numpy())
                        else:
                            preds[0] = np.append(
                                preds[0], logits.detach().cpu().numpy(), axis=0)
                    preds = preds[0]
                    preds = np.argmax(preds, axis=1)
                    acc = accuracy_score(preds, all_label_ids1.numpy())
                    print("===========================================")
                    print(acc)
            #保存模型
            model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
            # If we save using the predefined names, we can load using `from_pretrained`
            output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
            output_config_file = os.path.join(args.output_dir, CONFIG_NAME)

            torch.save(model_to_save.state_dict(), output_model_file)
            model_to_save.config.to_json_file(output_config_file)
            tokenizer.save_vocabulary(args.output_dir)



    if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        model = BertForSequenceClassification.from_pretrained(args.output_dir, num_labels=num_labels,in_channels=args.max_seq_length)
        model.to(device)
        eval_examples = processor.get_dev_examples(args.data_dir)
        eval_features = convert_examples_to_features(eval_examples, label_list, args.max_seq_length, tokenizer)
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)
        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)

        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

        model.eval()
        eval_loss = 0
        nb_eval_steps = 0
        preds = []

        for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader, desc="Evaluating"):
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            label_ids = label_ids.to(device)

            with torch.no_grad():
                logits = model(input_ids, segment_ids, input_mask, labels=None)

            # create eval loss and other metric required by the task
            loss_fct = CrossEntropyLoss()
            tmp_eval_loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))

            eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if len(preds) == 0:
                preds.append(logits.detach().cpu().numpy())
            else:
                preds[0] = np.append(
                    preds[0], logits.detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        preds = preds[0]

        preds = np.argmax(preds, axis=1)
        result,classification_report_true,acc = compute_metrics(preds, all_label_ids.numpy())
        loss = tr_loss/global_step if args.do_train else None

        result['eval_loss'] = eval_loss
        result['global_step'] = global_step
        result['loss'] = loss
        print(classification_report_true)
        print(acc)


        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            # writer.write(classification_report_true)
            # writer.write(acc)
            logger.info("***** Eval results *****")
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))



if __name__ == '__main__':
    main()