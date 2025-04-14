import os
import argparse
from multiprocessing import cpu_count
from utils.conceptnet import extract_chinese, construct_chinese_graph
from utils.grounding import create_matcher_patterns, ground
from utils.convert import convert_to_entailment
from utils.graph import generate_adj_data_from_grounded_concepts__use_LM

input_paths = {
    'ele_data': {
        'train': './data/ele_data2/train_rand_split.jsonl',
        'dev': './data/ele_data2/dev_rand_split.jsonl',
        'test': './data/ele_data2/test_rand_split_no_answers.jsonl',
    },
    'cpnet': {
        'csv': './data/cpnet/conceptnet-assertions-5.6.0.csv',
    },
}

output_paths = {
    'cpnet': {
        'csv': './data/cpnet/conceptnet.zh.csv',
        'vocab': './data/cpnet/concept.txt',
        'patterns': './data/cpnet/matcher_patterns.json',
        'unpruned-graph': './data/cpnet/conceptnet.zh.unpruned.graph',
        'pruned-graph': './data/cpnet/conceptnet.zh.pruned.graph',
    },
    'ele_data': {
        'statement': {
            'train': './data/ele_data2/statement/train.statement.jsonl',
            'dev': './data/ele_data2/statement/dev.statement.jsonl',
            'test': './data/ele_data2/statement/test.statement.jsonl',
        },
        'grounded': {
            'train': './data/ele_data2/grounded/train.grounded.jsonl',
            'dev': './data/ele_data2/grounded/dev.grounded.jsonl',
            'test': './data/ele_data2/grounded/test.grounded.jsonl',
        },
        'graph': {
            'adj-train': './data/ele_data2/graph/train.graph.adj.pk',
            'adj-dev': './data/ele_data2/graph/dev.graph.adj.pk',
            'adj-test': './data/ele_data2/graph/test.graph.adj.pk',
        },
    }
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', default=['common', 'ele_data2'], choices=[ 'ele_data2', 'ele_data' ,'make_word_vocab'], nargs='+')
    parser.add_argument('--path_prune_threshold', type=float, default=0.12, help='threshold for pruning paths')
    parser.add_argument('--max_node_num', type=int, default=200, help='maximum number of nodes per graph')
    parser.add_argument('-p', '--nprocs', type=int, default=cpu_count(), help='number of processes to use')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--debug', action='store_true', help='enable debug mode')

    args = parser.parse_args()
    if args.debug:
        raise NotImplementedError()

    routines = {
        'common': [
            # {'func': extract_chinese, 'args': (input_paths['cpnet']['csv'], output_paths['cpnet']['csv'], output_paths['cpnet']['vocab'])},
            {'func': construct_chinese_graph, 'args': (output_paths['cpnet']['csv'], output_paths['cpnet']['vocab'],
                                               output_paths['cpnet']['unpruned-graph'], False)},
            {'func': construct_chinese_graph, 'args': (output_paths['cpnet']['csv'], output_paths['cpnet']['vocab'],
                                               output_paths['cpnet']['pruned-graph'], True)},
            {'func': create_matcher_patterns, 'args': (output_paths['cpnet']['vocab'], output_paths['cpnet']['patterns'])},
        ],
        'ele_data2': [
            {'func': convert_to_entailment, 'args': (input_paths['ele_data']['train'], output_paths['ele_data']['statement']['train'])},
            {'func': convert_to_entailment, 'args': (input_paths['ele_data']['dev'], output_paths['ele_data']['statement']['dev'])},
            {'func': convert_to_entailment, 'args': (input_paths['ele_data']['test'], output_paths['ele_data']['statement']['test'])},
            {'func': ground, 'args': (output_paths['ele_data']['statement']['train'], output_paths['cpnet']['vocab'],
                                      output_paths['cpnet']['patterns'], output_paths['ele_data']['grounded']['train'], args.nprocs)},
            {'func': ground, 'args': (output_paths['ele_data']['statement']['dev'], output_paths['cpnet']['vocab'],
                                      output_paths['cpnet']['patterns'], output_paths['ele_data']['grounded']['dev'], args.nprocs)},
            {'func': ground, 'args': (output_paths['ele_data']['statement']['test'], output_paths['cpnet']['vocab'],
                                      output_paths['cpnet']['patterns'], output_paths['ele_data']['grounded']['test'], args.nprocs)},
            {'func': generate_adj_data_from_grounded_concepts__use_LM, 'args': (output_paths['ele_data']['grounded']['train'], output_paths['cpnet']['pruned-graph'], output_paths['cpnet']['vocab'], output_paths['ele_data']['graph']['adj-train'], args.nprocs)},
            {'func': generate_adj_data_from_grounded_concepts__use_LM, 'args': (output_paths['ele_data']['grounded']['dev'], output_paths['cpnet']['pruned-graph'], output_paths['cpnet']['vocab'], output_paths['ele_data']['graph']['adj-dev'], args.nprocs)},
            {'func': generate_adj_data_from_grounded_concepts__use_LM, 'args': (output_paths['ele_data']['grounded']['test'], output_paths['cpnet']['pruned-graph'], output_paths['cpnet']['vocab'], output_paths['ele_data']['graph']['adj-test'], args.nprocs)},
        ]
    }

    for rt in args.run:
        for rt_dic in routines[rt]:
            rt_dic['func'](*rt_dic['args'])

    print('Successfully run {}'.format(' '.join(args.run)))


if __name__ == '__main__':
    main()
