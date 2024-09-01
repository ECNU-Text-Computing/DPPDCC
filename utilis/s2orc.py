import argparse
import datetime
import json
import os
import sys
import time
from collections import defaultdict, Counter

import dgl
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from langdetect import detect
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

sys.path.insert(0, '.')
sys.path.insert(0, '..')
from utilis.scripts import IndexDict


def abstracts_data(data_path):
    files = os.listdir(data_path + 'abstracts/')
    id_dict = defaultdict(list)
    for file in files:
        with open(data_path + 'abstracts/' + file) as fr:
            print(file)
            for line in tqdm(fr):
                # print(line)
                temp_data = json.loads(line.strip())
                # print(temp_data)
                # print(temp_data.keys())
                # break
                id_dict[file].append(str(temp_data['corpusid']))
                # id_dict[file].append(temp_data['corpusid'])
    json.dump(id_dict, open(data_path + 'ab_dict.json', 'w+'))


def papers_data(data_path):
    files = os.listdir(data_path + 'papers/')
    valid_papers = json.load(open(data_path + 'ab_dict.json', 'r'))
    all_papers = []
    id_dict = defaultdict(list)
    for key in valid_papers:
        all_papers.extend(valid_papers[key])
    print('original papers:', len(all_papers))
    all_papers = set(all_papers)
    print('unique papers:', len(all_papers))
    cols = ['title', 'corpusid', 'authors', 'venue', 'publicationvenueid', 'year', 'referencecount',
            'citationcount', 's2fieldsofstudy', 'publicationtypes', 'publicationdate']
    raws = ['publicationvenueid', 'year', 'referencecount',
            'citationcount', 's2fieldsofstudy', 'publicationtypes', 'publicationdate']
    all_count = 0

    for file in files:
        cur_dict = {col: [] for col in cols}
        with open(data_path + 'papers/' + file) as fr:
            print(file)
            for line in tqdm(fr):
                # print(line)
                temp_data = json.loads(line.strip())
                all_count += 1
                # print(temp_data)
                # print(temp_data.keys())
                # break
                cur_id = str(temp_data['corpusid'])
                # if all_count == 100:
                #     break
                if cur_id in all_papers:
                    id_dict[file].append(cur_id)
                    cur_dict['corpusid'].append(cur_id)
                    cur_dict['authors'].append(len(temp_data['authors']))
                    cur_dict['venue'].append(True if temp_data['venue'] else False)

                    # title_lang = None
                    # try:
                    #     title_lang = detect(temp_data['title'])
                    # except Exception as e:
                    #     print(e)
                    #     print(temp_data['title'])
                    title_lang = temp_data['title']
                    cur_dict['title'].append(title_lang)

                    for col in raws:
                        cur_dict[col].append(temp_data[col])
        df = pd.DataFrame(cur_dict)
        df.to_csv(data_path + 'papers_df/' + file + '.csv')
    print('all papers:', all_count)
    json.dump(id_dict, open(data_path + 'paper_dict.json', 'w+'))


def custom_detect(string):
    result = None
    try:
        result = detect(string)
    except Exception as e:
        print(e)
        print(string)
    # result = detect(string)
    # print(result)
    return result


def recognize_title_lang(data_path, index_col, cut):
    file_path = data_path + 'papers_df/'
    files = sorted(os.listdir(file_path))
    # print(files)
    cur_length = len(files) // int(cut)
    start_index = int(index_col) * cur_length
    cur_files = files[start_index: start_index+cur_length]
    print(cur_files)
    print(detect('first to load the model!'))

    for file in cur_files:
        df = pd.read_csv(file_path + file, index_col=0)
        titles = df['title']
        # with ThreadPoolExecutor(max_workers=4) as executor:
        with ProcessPoolExecutor(max_workers=16) as executor:
            results = list(tqdm(executor.map(custom_detect, titles),
                                total=len(titles)))
        # results = [custom_detect(title) for title in titles]
        df['title_lang'] = results
        df.to_csv(file_path + file)
        print('>>>{} done!<<<'.format(file))


def df_data(data_path):
    files = os.listdir(data_path + 'papers_df/')
    df = pd.read_csv(data_path + 'papers_df/' + files[0], index_col=0)
    # df['s2fieldsofstudy'] = df['s2fieldsofstudy'].fillna('[]')
    df['s2fieldsofstudy'] = df['s2fieldsofstudy'].apply(lambda x: eval(x) if x is not np.nan else [])
    df['fos_count'] = df['s2fieldsofstudy'].apply(lambda x: len(x))

    for file in tqdm(files[1:]):
        df_temp = pd.read_csv(data_path + 'papers_df/' + file, index_col=0)
        # df_temp['s2fieldsofstudy'] = df_temp['s2fieldsofstudy'].fillna('[]')
        df_temp['s2fieldsofstudy'] = df_temp['s2fieldsofstudy'].apply(lambda x: eval(x) if x is not np.nan else [])
        df_temp['fos_count'] = df_temp['s2fieldsofstudy'].apply(lambda x: len(x))

        df = df.append(df_temp)

    print('original papers:', df.shape[0])
    df = df.drop_duplicates(subset='corpusid')
    print('unique papers:', df.shape[0])

    print(df[['authors', 'year', 'referencecount', 'citationcount', 'fos_count']].
          describe(percentiles=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99]))
    df[['authors', 'year', 'referencecount', 'citationcount', 'fos_count']].\
        describe(percentiles=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99]).\
        to_csv(data_path + 'quantitative.csv')
    print(df[['title_lang', 'venue', 'publicationvenueid', 'publicationtypes', 'publicationdate']].describe())
    df[['title_lang', 'venue', 'publicationvenueid', 'publicationtypes', 'publicationdate']].describe()\
        .to_csv(data_path + 'qualitative.csv')

    count_dict = defaultdict(int)
    cat_dict = defaultdict(int)
    all_values = []
    for value in df['s2fieldsofstudy'].values:
        cur_value = []
        for item in value:
            if item['source'] == 's2-fos-model':
                # cur_value = item['category']
                cur_value.append(item['category'])
                cat_dict[item['category']] += 1
            count_dict[item['source']] += 1
        all_values.append(cur_value)
    print(count_dict)
    # print(pd.DataFrame([len(value) for value in all_values]).
    #       describe(percentiles=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99]))
    df['s2fos'] = all_values

    df = df[['title', 'title_lang', 'corpusid', 'authors', 'venue', 'publicationvenueid', 'year',
       'referencecount', 'citationcount', 'fos_count', 's2fos',
       'publicationtypes', 'publicationdate']]
    # df['publicationvenueid'] = df['publicationvenueid'].apply(lambda x: True if x is not np.nan else False)

    df.to_csv(data_path + 'df_complete.csv')
    df.groupby('year').count().to_csv(data_path + 'year_count.csv')
    pd.DataFrame(index=cat_dict.keys(), data=cat_dict.values()).sort_values(by=0, ascending=False).\
        to_csv(data_path + 'cat_count.csv')


def selected_data(data_path):
    df = pd.read_csv(data_path + 'df_complete.csv', index_col=0)
    df.groupby('title_lang').count().sort_values(by='corpusid', ascending=False)['corpusid']\
        .to_csv(data_path + 'lang_count.csv')

    print('original papers:', df.shape[0])
    df = df[(df['authors'] > 0)&(df['year'].notna())&(df['title_lang']=='en')]
    print('selected papers:', df.shape[0])
    df['corpusid'] = df['corpusid'].astype(str)
    df['s2fos'] = df['s2fos'].apply(lambda x: eval(x))
    df['s2fos_count'] = df['s2fos'].apply(lambda x: len(x))
    cat_dict = defaultdict(int)
    for value in df['s2fos'].values:
        for cat in value:
            cat_dict[cat] += 1

    print(df[['authors', 'year', 'referencecount', 'citationcount', 's2fos_count']].
          describe(percentiles=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99]))
    df[['authors', 'year', 'referencecount', 'citationcount', 'fos_count']].\
        describe(percentiles=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99]).\
        to_csv(data_path + 'selected_quantitative.csv')
    print(df[['venue', 'publicationvenueid', 'publicationtypes', 'publicationdate']].describe())
    df[['venue', 'publicationvenueid', 'publicationtypes', 'publicationdate']].describe()\
        .to_csv(data_path + 'selected_qualitative.csv')

    df.groupby('year').count().to_csv(data_path + 'selected_year_count.csv')
    pd.DataFrame(index=cat_dict.keys(), data=cat_dict.values()).sort_values(by=0, ascending=False).\
        to_csv(data_path + 'selected_cat_count.csv')

    json.dump(df['corpusid'].to_list(), open(data_path + 'selected_ids.json', 'w+'))

    # important papers
    df_selected = df[(df['publicationvenueid'].isna())&(df['s2fos'].apply(lambda x: 'Computer Science' in x))]
    df_selected.to_csv(data_path + 'cs_view.csv')
    df_selected = df_selected[df_selected['citationcount']>=100]
    df_selected.to_csv(data_path + 'cs_high_view.csv')


def cat_detail(data_path):
    df = pd.read_csv(data_path + 'df_complete.csv', index_col=0)
    cats = list(pd.read_csv(data_path + 'selected_cat_count.csv', index_col=0).index)

    print('original papers:', df.shape[0])
    df = df[(df['authors'] > 0)&(df['year'].notna())&(df['title_lang']=='en')]
    print('selected papers:', df.shape[0])
    df['corpusid'] = df['corpusid'].astype(str)
    df['s2fos'] = df['s2fos'].apply(lambda x: set(eval(x)))
    df['s2fos_count'] = df['s2fos'].apply(lambda x: len(x))

    for cat in cats:
        df_cat = df[df['s2fos'].apply(lambda x: cat in x)]
        quantitative = df_cat[['authors', 'year', 'referencecount', 'citationcount', 's2fos_count']]\
            .describe(percentiles=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99])
        print(quantitative)
        quantitative.to_csv(data_path + 'cat_{}_quantitative.csv'.format(cat))
        qualitative = df_cat[['venue', 'publicationvenueid', 'publicationtypes', 'publicationdate']].describe()
        print(qualitative)
        qualitative.to_csv(data_path + 'cat_{}_qualitative.csv'.format(cat))


def citations_data(data_path):
    # ['citingcorpusid', 'citedcorpusid', 'isinfluential', 'contexts', 'intents', 'updated']
    useful_keys = ['isinfluential', 'contexts', 'intents']
    count_dict = {key: 0 for key in ['count'] + useful_keys}
    ref_dict = defaultdict(list)
    files = os.listdir(data_path + 'citations/')
    valid_papers = set(json.load(open(data_path + 'selected_ids.json', 'r')))
    for file in files:
        with open(data_path + 'citations/' + file) as fr:
            print(file)
            for line in fr:
                # print(line)
                temp_data = json.loads(line.strip())
                # print(temp_data)
                # print(temp_data.keys())
                count_dict['count'] += 1
                for key in useful_keys:
                    if temp_data.get(key, None):
                        count_dict[key] += 1
                if temp_data['citingcorpusid'] in valid_papers and temp_data['citedcorpusid'] in valid_papers:
                    ref_dict[temp_data['citingcorpusid']].append(temp_data['citedcorpusid'])
    print(count_dict)
    json.dump(ref_dict, open(data_path + 'all_ref_dict.json', 'w+'))
    cite_dict = defaultdict(list)
    for key in ref_dict:
        for paper in ref_dict[key]:
            cite_dict[paper].append(key)
    del ref_dict
    json.dump(cite_dict, open(data_path + 'all_cite_dict.json', 'w+'))


def cat_subset(data_path, cat):
    df = pd.read_csv(data_path + 'df_complete.csv', index_col=0)
    df = df[(df['authors'] > 0) & (df['year'].notna()) & (df['title_lang'] == 'en') &
            ((df['publicationvenueid'].notna()) | (df['citationcount'] >= 100))]
    print(df[df['publicationvenueid'].isna()].shape)

    df['corpusid'] = df['corpusid'].astype(str)
    df['s2fos'] = df['s2fos'].apply(lambda x: set(eval(x)))
    df['s2fos_count'] = df['s2fos'].apply(lambda x: len(x))

    df_cat = df[df['s2fos'].apply(lambda x: cat in x)]
    json.dump(df_cat['corpusid'].to_list(),
              open(data_path.replace('s2orc/', '') + cat.lower().replace(' ', '_') + '/all_ids.json', 'w+'))
    valid_papers = set(df_cat['corpusid'])

    # info_dict
    paper_dict = json.load(open(data_path + 'paper_dict.json'))
    read_files = []
    for key in paper_dict:
        paper_dict[key] = set(paper_dict[key])
        if paper_dict[key] & valid_papers:
            read_files.append(key)
    del paper_dict
    info_dict = {}
    print(len(read_files))
    # read_files = [read_files[0]]
    # print('test!!!!')
    for file in read_files:
        with open(data_path + 'papers/' + file) as fr:
            print(file)
            for line in tqdm(fr):
                # print(line)
                temp_data = json.loads(line.strip())
                cur_id = str(temp_data['corpusid'])
                if cur_id in valid_papers:
                    info_dict[cur_id] = temp_data
    json.dump(info_dict, open(data_path.replace('s2orc/', '') + cat.lower().replace(' ', '_') + '/all_info_dict.json', 'w+'))
    del info_dict

    # ref_dict
    ref_dict = json.load(open(data_path + 'all_ref_dict.json'))
    selected_ref_dict = {}
    for paper in valid_papers:
        if paper in ref_dict:
            selected_ref_dict[paper] = [paper for paper in ref_dict[paper] if paper in valid_papers]
    json.dump(selected_ref_dict, open(data_path.replace('s2orc/', '') + cat.lower().replace(' ', '_') + '/all_ref_dict.json', 'w+'))
    cite_dict = defaultdict(list)
    for paper in selected_ref_dict:
        for cited in selected_ref_dict[paper]:
            cite_dict[cited].append(paper)
    json.dump(cite_dict, open(data_path.replace('s2orc/', '') + cat.lower().replace(' ', '_') + '/all_cite_dict.json', 'w+'))
    del ref_dict
    del selected_ref_dict

    # ab_dict
    ab_dict = json.load(open(data_path + 'ab_dict.json'))
    ab_text_dict = {}
    read_files = []
    for key in ab_dict:
        ab_dict[key] = set(ab_dict[key])
        if ab_dict[key] & valid_papers:
            read_files.append(key)
    del ab_dict
    print(len(read_files))
    # read_files = [read_files[0]]
    # print('test!!!!')
    for file in read_files:
        with open(data_path + 'abstracts/' + file) as fr:
            print(file)
            for line in tqdm(fr):
                # print(line)
                temp_data = json.loads(line.strip())
                cur_id = str(temp_data['corpusid'])
                if cur_id in valid_papers:
                    ab_text_dict[cur_id] = temp_data
    json.dump(ab_text_dict, open(data_path.replace('s2orc/', '') + cat.lower().replace(' ', '_') + '/all_abstract_dict.json', 'w+'))


def cat_df(data_path):
    cols = ['corpusid', 'authors', 'venue', 'publicationvenueid', 'year', 'referencecount',
            'citationcount', 's2fieldsofstudy', 'publicationtypes', 'publicationdate']
    all_info_dict = json.load(open(data_path + 'all_info_dict.json'))
    all_ref_dict = json.load(open(data_path + 'all_ref_dict.json'))
    all_cite_dict = json.load(open(data_path + 'all_cite_dict.json'))
    data = map(lambda x: {
        'corpusid': str(x['corpusid']),
        'authors': len(x['authors']),
        'publicationvenueid': x['publicationvenueid'],
        'year': x['year'],
        'referencecount': x['referencecount'],
        'citationcount': x['citationcount'],
        'publicationdate': x['publicationdate'],
        'cur_refcount': len(all_ref_dict.get(str(x['corpusid']), [])),
        'cur_citecount': len(all_cite_dict.get(str(x['corpusid']),[]))
    }, all_info_dict.values())

    df = pd.DataFrame(data)
    df.to_csv(data_path + 'df_complete.csv')
    quantitative = df[['authors', 'year', 'referencecount', 'citationcount', 'cur_refcount', 'cur_citecount']] \
        .describe(percentiles=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99])
    print(quantitative)
    quantitative.to_csv(data_path + 'quantitative.csv')
    # qualitative = df[['venue', 'publicationvenueid', 'publicationtypes', 'publicationdate']].describe()
    # print(qualitative)
    # qualitative.to_csv(data_path + 'qualitative.csv')
    df.groupby('year').count().to_csv(data_path + 'year_count.csv')


def get_sample_data(data_path, time_point=2017):
    # filtered subgraph
    info_path = data_path + 'all_info_dict.json'
    ref_path = data_path + 'all_ref_dict.json'
    cite_path = data_path + 'all_cite_dict.json'
    abstract_path = data_path + 'all_abstract_dict.json'

    info_dict = json.load(open(info_path, 'r'))
    ref_dict = json.load(open(ref_path, 'r'))
    cite_dict = json.load(open(cite_path, 'r'))

    sample_info_dict = dict(filter(lambda x: int(x[1]['year']) <= time_point, info_dict.items()))
    valid_papers = set(sample_info_dict.keys())
    json.dump(sample_info_dict, open(data_path + 'sample_info_dict.json', 'w+'))
    del sample_info_dict

    # only maintain the test set
    sample_ref_dict = {}
    for paper in valid_papers:
        sample_ref_dict[paper] = list(set([ref for ref in ref_dict.get(paper, []) if ref in valid_papers]))
    json.dump(sample_ref_dict, open(data_path + 'sample_ref_dict.json', 'w+'))
    sample_cite_dict = defaultdict(list)
    for paper in sample_ref_dict:
        for ref in sample_ref_dict[paper]:
            sample_cite_dict[ref].append(paper)
    json.dump(sample_cite_dict, open(data_path + 'sample_cite_dict.json', 'w+'))

    # groundtruth citation
    cite_dict = dict(filter(lambda x: x[0] in valid_papers, cite_dict.items()))
    # all accumulated year citation
    year_dict = dict(map(lambda x: (x[0], Counter([int(info_dict[paper]['year']) for paper in x[1]])), cite_dict.items()))
    json.dump(year_dict, open(data_path + 'sample_cite_year_dict.json', 'w+'))

    del info_dict

    abstract_dict = json.load(open(abstract_path, 'r'))
    abstract_dict = dict(filter(lambda x: x[0] in valid_papers, abstract_dict.items()))
    json.dump(abstract_dict, open(data_path + 'sample_abstract_dict.json', 'w+'))


def get_citation_accum(data_path, time_point=None, time_length=5, all_times=(2011, 2013, 2015)):
    info_path = data_path + 'sample_info_dict.json'
    ref_path = data_path + 'sample_ref_dict.json'
    cite_year_path = data_path + 'sample_cite_year_dict.json'

    info_dict = json.load(open(info_path, 'r'))
    ref_dict = json.load(open(ref_path, 'r'))
    cite_year_dict = json.load(open(cite_year_path, 'r'))
    predicted_list = list(cite_year_dict.keys())
    accum_num_dict = {}
    print('writing citations accum')
    valid_time_dict = {}
    pub_dict = {}
    backward_times = 3
    for paper in predicted_list:
        count_dict = dict(map(lambda x: (int(x[0]), x[1]), cite_year_dict[paper].items()))
        pub_year = int(info_dict[paper]['year'])
        pub_dict[paper] = pub_year
        valid_time_dict[paper] = pub_year
        start_year = max(pub_year, time_point - backward_times * time_length + 1)
        # print(count_dict)
        count = sum(dict(filter(lambda x: x[0] < start_year, count_dict.items())).values())
        temp_accum_num = [-1] * (start_year + backward_times * time_length - 1 - time_point)
        for year in range(start_year, time_point + time_length + 1):
            if year in count_dict:
                count += int(count_dict[year])
            temp_accum_num.append(count)
            # print(temp_input_accum_num
        accum_num_dict[paper] = temp_accum_num
    print('all_papers:', len(accum_num_dict))
    json.dump(accum_num_dict, open(data_path + 'sample_citation_accum.json', 'w+'))


def get_input_data(data_path, time_point=None, subset=False):
    info_path = data_path + 'all_info_dict.json'
    ref_path = data_path + 'all_ref_dict.json'
    cite_year_path = data_path + 'all_cite_year_dict.json'
    if subset:
        info_path = data_path + 'sample_info_dict.json'
        ref_path = data_path + 'sample_ref_dict.json'
        cite_year_path = data_path + 'sample_cite_year_dict.json'

    info_dict = json.load(open(info_path, 'r'))
    ref_dict = json.load(open(ref_path, 'r'))
    cite_year_dict = json.load(open(cite_year_path, 'r'))

    node_trans = {}
    paper_trans = IndexDict()
    # graph created
    src_list = []
    dst_list = []
    for dst in ref_dict:
        dst_idx = paper_trans[dst]
        for src in set(ref_dict[dst]):
            src_idx = paper_trans[src]
            src_list.append(src_idx)
            dst_list.append(dst_idx)

    node_trans['paper'] = paper_trans

    author_trans = IndexDict()
    journal_trans = IndexDict()
    author_src, author_dst = [], []
    journal_src, journal_dst = [], []
    # author_index = 0
    journal_index = 0
    for paper in paper_trans:
        meta_data = info_dict[paper]
        authors = [str(author['authorId']) for author in meta_data['authors']]
        journal = meta_data['publicationvenueid']

        for author in authors:
            cur_idx = author_trans[author]
            author_src.append(cur_idx)
            author_dst.append(paper_trans[paper])

        if journal:
            cur_idx = journal_trans[journal]
            journal_src.append(cur_idx)
            journal_dst.append(paper_trans[paper])

    node_trans['author'] = author_trans
    node_trans['journal'] = journal_trans

    # node_trans_reverse = dict(map(lambda x: (x[0], dict(zip(x[1].values(), x[1].keys()))), node_trans.items()))
    node_trans_reverse = {key: dict(zip(node_trans[key].values(), node_trans[key].keys())) for key in node_trans}
    paper_link_time = [int(info_dict[node_trans_reverse['paper'][paper]]['year']) for paper in dst_list]
    author_link_time = [int(info_dict[node_trans_reverse['paper'][paper]]['year']) for paper in author_dst]
    journal_link_time = [int(info_dict[node_trans_reverse['paper'][paper]]['year']) for paper in journal_dst]
    paper_time = [int(info_dict[paper]['year']) for paper in node_trans['paper'].keys()]

    if subset:
        json.dump(node_trans, open(data_path + 'sample_node_trans.json', 'w+'))
    else:
        json.dump(node_trans, open(data_path + 'all_node_trans.json', 'w+'))

    # graph = dgl.graph((src_list, dst_list), num_nodes=len(paper_trans))
    graph = dgl.heterograph({
        ('paper', 'is cited by', 'paper'): (src_list, dst_list),
        ('paper', 'cites', 'paper'): (dst_list, src_list),
        ('author', 'writes', 'paper'): (author_src, author_dst),
        ('paper', 'is writen by', 'author'): (author_dst, author_src),
        ('journal', 'publishes', 'paper'): (journal_src, journal_dst),
        ('paper', 'is published by', 'journal'): (journal_dst, journal_src),
    }, num_nodes_dict={
        'paper': len(paper_trans),
        'author': len(author_trans),
        'journal': len(journal_trans)
    })
    # graph.ndata['paper_id'] = torch.tensor(list(node_trans.keys())).unsqueeze(dim=0)
    graph.nodes['paper'].data['time'] = torch.tensor(paper_time, dtype=torch.int16).unsqueeze(dim=-1)
    graph.edges['is cited by'].data['time'] = torch.tensor(paper_link_time, dtype=torch.int16).unsqueeze(dim=-1)
    graph.edges['cites'].data['time'] = torch.tensor(paper_link_time, dtype=torch.int16).unsqueeze(dim=-1)
    graph.edges['writes'].data['time'] = torch.tensor(author_link_time, dtype=torch.int16).unsqueeze(dim=-1)
    graph.edges['is writen by'].data['time'] = torch.tensor(author_link_time, dtype=torch.int16).unsqueeze(dim=-1)
    graph.edges['publishes'].data['time'] = torch.tensor(journal_link_time, dtype=torch.int16).unsqueeze(dim=-1)
    graph.edges['is published by'].data['time'] = torch.tensor(journal_link_time, dtype=torch.int16).unsqueeze(dim=-1)

    graph = dgl.remove_self_loop(graph, 'is cited by')
    graph = dgl.remove_self_loop(graph, 'cites')
    print(graph)
    print(graph.edges['cites'].data['time'])
    for ntype in graph.ntypes:
        graph.nodes[ntype].data['oid'] = graph.nodes(ntype)
    if subset:
        torch.save(graph, data_path + 'graph_sample')
    else:
        torch.save(graph, data_path + 'graph')

    del graph, src_list, dst_list


def get_valid_papers(data_path, time_point=None):
    data = json.load(open(data_path + 'all_info_dict.json', 'r'))
    print(len(data))
    if time_point:
        data = dict(filter(lambda x: (int(x[1]['year']) <= time_point), data.items()))
        print('<={}:'.format(time_point), len(data))
    return set(data.keys())


def get_complete_data(data_path, time_point=None):
    info_path = data_path + 'all_info_dict.json'
    ref_path = data_path + 'all_ref_dict.json'
    cite_year_path = data_path + 'all_cite_year_dict.json'

    info_dict = json.load(open(info_path, 'r'))
    ref_dict = json.load(open(ref_path, 'r'))
    cite_year_dict = json.load(open(cite_year_path, 'r'))

    valid_papers = get_valid_papers(data_path, time_point)
    ref_dict = dict(filter(lambda x: x[0] in valid_papers, ref_dict.items()))

    node_trans = {}
    paper_trans = IndexDict()
    # graph created
    src_list = []
    dst_list = []
    for dst in ref_dict:
        dst_idx = paper_trans[dst]
        for src in set(ref_dict[dst]):
            if src in valid_papers:
                src_idx = paper_trans[src]
                src_list.append(src_idx)
                dst_list.append(dst_idx)

    node_trans['paper'] = paper_trans

    author_trans = IndexDict()
    journal_trans = IndexDict()
    author_src, author_dst = [], []
    journal_src, journal_dst = [], []
    for paper in paper_trans:
        meta_data = info_dict[paper]
        authors = [str(author['authorId']) for author in meta_data['authors']]
        journal = meta_data['publicationvenueid']

        for author in authors:
            cur_idx = author_trans[author]
            author_src.append(cur_idx)
            author_dst.append(paper_trans[paper])

        if journal:
            cur_idx = journal_trans[journal]
            journal_src.append(cur_idx)
            journal_dst.append(paper_trans[paper])

    node_trans['author'] = author_trans
    node_trans['journal'] = journal_trans

    # node_trans_reverse = dict(map(lambda x: (x[0], dict(zip(x[1].values(), x[1].keys()))), node_trans.items()))
    node_trans_reverse = {key: dict(zip(node_trans[key].values(), node_trans[key].keys())) for key in node_trans}
    paper_link_time = [int(info_dict[node_trans_reverse['paper'][paper]]['year']) for paper in dst_list]
    author_link_time = [int(info_dict[node_trans_reverse['paper'][paper]]['year']) for paper in author_dst]
    journal_link_time = [int(info_dict[node_trans_reverse['paper'][paper]]['year']) for paper in journal_dst]
    paper_time = [int(info_dict[paper]['year']) for paper in node_trans['paper'].keys()]

    json.dump(node_trans, open(data_path + 'all_node_trans.json', 'w+'))

    # graph = dgl.graph((src_list, dst_list), num_nodes=len(paper_trans))
    graph = dgl.heterograph({
        ('paper', 'is cited by', 'paper'): (src_list, dst_list),
        ('paper', 'cites', 'paper'): (dst_list, src_list),
        ('author', 'writes', 'paper'): (author_src, author_dst),
        ('paper', 'is writen by', 'author'): (author_dst, author_src),
        ('journal', 'publishes', 'paper'): (journal_src, journal_dst),
        ('paper', 'is published by', 'journal'): (journal_dst, journal_src),
    }, num_nodes_dict={
        'paper': len(paper_trans),
        'author': len(author_trans),
        'journal': len(journal_trans)
    })
    # graph.ndata['paper_id'] = torch.tensor(list(node_trans.keys())).unsqueeze(dim=0)
    graph.nodes['paper'].data['time'] = torch.tensor(paper_time, dtype=torch.int16).unsqueeze(dim=-1)
    graph.edges['is cited by'].data['time'] = torch.tensor(paper_link_time, dtype=torch.int16).unsqueeze(dim=-1)
    graph.edges['cites'].data['time'] = torch.tensor(paper_link_time, dtype=torch.int16).unsqueeze(dim=-1)
    graph.edges['writes'].data['time'] = torch.tensor(author_link_time, dtype=torch.int16).unsqueeze(dim=-1)
    graph.edges['is writen by'].data['time'] = torch.tensor(author_link_time, dtype=torch.int16).unsqueeze(dim=-1)
    graph.edges['publishes'].data['time'] = torch.tensor(journal_link_time, dtype=torch.int16).unsqueeze(dim=-1)
    graph.edges['is published by'].data['time'] = torch.tensor(journal_link_time, dtype=torch.int16).unsqueeze(dim=-1)

    graph = dgl.remove_self_loop(graph, 'is cited by')
    graph = dgl.remove_self_loop(graph, 'cites')
    print(graph)
    print(graph.edges['cites'].data['time'])
    torch.save(graph, data_path + 'graph_complete')

    del graph, src_list, dst_list


if __name__ == "__main__":
    start_time = datetime.datetime.now()
    parser = argparse.ArgumentParser(description='Process some description.')
    parser.add_argument('--phase', default='test', help='the function name.')
    parser.add_argument('--data_path', default=None, help='the input.')
    parser.add_argument('--name', default=None, help='file name.')
    parser.add_argument('--out_path', default=None, help='the output.')
    parser.add_argument('--cat', default=None, help='selected cat.')
    parser.add_argument('--index', default=None, help='start_index.')
    parser.add_argument('--time_point', default=2015, help='time_point.')
    parser.add_argument('--time_length', default=5, help='time_point.')
    # parser.add_argument('--seed', default=123, help='the seed.')
    args = parser.parse_args()
    if args.phase == 'test':
        print('This is a test process.')
    elif args.phase == 'abstracts':
        abstracts_data(args.data_path)
    elif args.phase == 'papers':
        papers_data(args.data_path)
    elif args.phase == 'detect_lang':
        recognize_title_lang(args.data_path, args.index, 10)
    elif args.phase == 'df':
        df_data(args.data_path)
    elif args.phase == 'selected_data':
        selected_data(args.data_path)
    elif args.phase == 'citation':
        citations_data(args.data_path)
    elif args.phase == 'cat_detail':
        cat_detail(args.data_path)
    elif args.phase == 'cat_subset':
        cat_subset(args.data_path, cat=args.cat)
    elif args.phase == 'cat_df':
        cat_df(args.data_path)
    elif args.phase == 'sample':
        get_sample_data(args.data_path, time_point=int(args.time_point))
        get_citation_accum(args.data_path, time_point=int(args.time_point), time_length=int(args.time_length))
    elif args.phase == 'citation_accum':
        get_citation_accum(args.data_path, time_point=int(args.time_point), time_length=int(args.time_length))
    elif args.phase == 'input':
        get_input_data(args.data_path, subset=True)

    end_time = datetime.datetime.now()
    print('{} takes {} seconds'.format(args.phase, (end_time - start_time).seconds))
    print('{} done!'.format(args.phase))