# coding: utf-8

import sys
PYTHON_VER = sys.version_info[0]
print('INFO: PYTHON_VER = %s' % PYTHON_VER)
import os
import re
import csv
import json
import jieba
import numpy as np
if PYTHON_VER == '2':
    import cPickle as pickle
    reload(sys)
    sys.setdefaultencoding('utf-8')
else:
    import pickle
from .utility import csvread

class InfoHandler(object):

    foreign_schools = ['耶鲁大学', '卡内基美隆大学', '剑桥大学', '哈佛大学', '牛津大学', '伯克利']   # Todo: to be added

    def __init__(self):
        self.info = None    
        self.major_dict = {}
        # 中英文专业对照表
        self.en_cn_major_dict = {}
        file_path = os.path.join('model', 'en_cn_major_dict_2.pkl')
        if os.path.exists(file_path):
            self.en_cn_major_dict = pickle.load(open(file_path, 'rb'))
        else:
            raise Exception('Error: File "%s" not found' % file_path)
        self.subject_idx = {}
        self.similarity_matrix = [[0 for i in range(18)] for j in range(18)]
        file_path = os.path.join('model', 'subject_matrix_2.pkl')
        if os.path.exists(file_path):
            (self.subject_idx, self.en_cn_major_dict) = pickle.load(open(file_path, 'rb'))
        else:
            raise Exception('Error: File "%s" not found' %file_path)

    def get_feature(self, info, source='csv'):
        # print(info)
        # _ = input('...')
        if source == 'csv':
            row_dict = {'apply_advantage': 0, 'school_score': 2, 'apply_disadvantage': 3, 'offer': 6,
                        'background': 8, 'science_score': 9, 'language_score': 10} # todo
            self.info = {}
            for key, row_num in row_dict.items():
                self.info[key] = info[row_num]
        elif source == 'json':
            self.info = json.loads(info)
        elif source == 'dict':
            self.info = info
        else:
            raise Exception('Info source error: not in (csv json dict)')
        '''
        try:
            assert len(self.info) == 11
        except:
            print(len(self.info))
            assert len(self.info) == 11
        '''
        apply_advantage = self.h_apply_advantage()
        apply_disadvantage = self.h_apply_disadvantage()
        try:
            background = self.h_background()
        except:
            return None
            # raise Exception('Background error: not undergraduate')
        gpa = self.h_gpa()
        toefl = self.h_toefl()
        gre = self.h_gre()
        other_feature = self.get_other_feature()
        return apply_advantage + apply_disadvantage + background + gpa + toefl + gre + other_feature

    @staticmethod
    def any_in(s, contents):
        return len([content for content in contents if content.decode('utf-8') in s]) != 0

    @staticmethod
    def any_in_any(targets, patterns):
        return any([[pattern in target for pattern in patterns] for target in targets])

    @staticmethod
    def get_context():
        pass

    @staticmethod
    def fuzzy_similarity(s1, s2):
        def get_gram(s, length):
            return set([s[i: i+length] for i in range(0, len(s)-length)])

        def jaccard_similarity(set1, set2):
            return 1.0 * len(set1 & set2) / len(set1 | set2)

        # unigram
        letter1, letter2 = set(s1), set(s2)
        d1 = jaccard_similarity(letter1, letter2)
        # bigram
        letter1, letter2 = get_gram(s1, 2), get_gram(s2, 2)
        d2 = jaccard_similarity(letter1, letter2)
        # trigram
        letter1, letter2 = get_gram(s1, 3), get_gram(s2, 3)
        d3 = jaccard_similarity(letter1, letter2)
        return d1 + 4 * d2 + 10 * d3

    def h_apply_advantage(self):
        '''
        优势：
        GRE TOFEL ITLES三维数据 语言：有无
        本科学校：国内985 211 / 国外
        交流经历
        实习经历
        志愿经历
        排名
        项目经验、作品集
        '''
        apply_advantage = self.info['apply_advantage']
        ret = []
        print(type(apply_advantage))
        print(type('jialiu'))
        ret.append(1 if InfoHandler.any_in(apply_advantage, ['交流', '交换', '活动', '出国']) else 0)
        ret.append(1 if InfoHandler.any_in(apply_advantage, ['工作', '实习', '公司', '社团', '志愿者'])else 0)
        ret.append(1 if InfoHandler.any_in(apply_advantage, ['奖学金', '大奖', '优胜']) else 0)
        ret.append(1 if InfoHandler.any_in(apply_advantage, ['论文', '发表', '学术']) else 0)
        ret.append(1 if InfoHandler.any_in(apply_advantage, ['资讯', '顾问', '银行', '金融']) else 0)
        return ret

    def h_apply_disadvantage(self):
        apply_disadvantage = self.info['apply_disadvantage']
        ret = []
        ret.append(1 if InfoHandler.any_in(apply_disadvantage, ['交流', '交换', '活动', '出国']) else 0)
        ret.append(1 if InfoHandler.any_in(apply_disadvantage, ['工作', '实习', '公司', '社团', '志愿者']) else 0)
        ret.append(1 if InfoHandler.any_in(apply_disadvantage, ['奖学金', '大奖', '优胜']) else 0)
        ret.append(1 if InfoHandler.any_in(apply_disadvantage, ['论文', '发表', '学术']) else 0)
        ret.append(1 if InfoHandler.any_in(apply_disadvantage, ['证书', '资格证']) else 0)
        ret.append(1 if InfoHandler.any_in(apply_disadvantage, ['排名']) else 0)
        ret.append(1 if InfoHandler.any_in(apply_disadvantage, ['领导', '沟通']) else 0)
        return ret

    def h_solution_analyse(self):
        solution_analyse = self.info[4]

    def h_offer(self):
        offers = self.info['offer'].split('\n')
        for offer in offers:
            offer_info = offer[2:].split(',')
            assert len(offer_info) >= 5
            major = offer[2]
            if major not in self.major_dict.keys():
                self.major_dict[major] = len(self.major_dict)

    '''
    @Anqi Yang: 计算背景专业和目标专业的距离
    '''
    def calc_distance_offer_major(self, background_major):
        offers = self.info['offer'].split('\n')
        offer_major = offers[0].split(',')[2]
        try:
            offer_major = self.en_cn_major_dict[offer_major]
        except:
            pass

        background_subject = self.major_to_subject(background_major)
        offer_subject = self.major_to_subject(offer_major)

        return self.calc_distance(background_subject, offer_subject)

    '''@Anqi Yang:专业映射到学科'''
    def major_to_subject(self, major):
        major_keyword = jieba.lcut(major, cut_all=True)
        subjects_keyword = self.load_subjects_keyword('/data/各学科关键词/')
        subjects_counter = {}

        for (subject_name, subject_keyword) in subjects_keyword.items():
            count = len(set(major_keyword).intersection(subject_keyword))
            subjects_counter[subject_name] = count

        if all(x == 0 for x in subjects_counter.values()):
            return ''
        else:
            sorted_subjects = sorted(subjects_counter, key=subjects_counter.__getitem__, reverse=True)
            return sorted_subjects[0]

    '''@Anqi Yang: 加载各学科的关键词'''
    def load_subjects_keyword(self, froot):
        subjectDic = {}

        fileList = os.listdir(froot)[1:]
        for fname in fileList:
            subject_name = fname.split('.')[0]
            subject_keyword = list(np.squeeze(csvread(froot + fname)))
            subjectDic[subject_name] = subject_keyword
        return subjectDic

    '''@Anqi Yang: 计算 背景学科 和 目标学科 的距离'''
    def calc_distance(self, background, offer):
        row = self.subject_idx[offer]
        col = self.subject_idx[background]
        return self.similarity_matrix[row][col]

    def h_apply_suggest(self):
        apply_suggest = self.info[7]

    def h_background(self):
        # Todo: 细化学校分类，国内学校根据网上排名进行划分
        # 可以考虑从高到低依次赋予分数
        # 注意保留学校到分数的映射关系，不存在于训练数据的学校也尽可能包含分数
        background = self.info['background']
        school_type = {'abord': 1, '985': 2, '211': 3, '1ben': 4, '2ben': 5,
            'middle': 6, 'key_middle': 7, 'other': 0}
        f_cnt_ascii = lambda s: len([c for c in s if ord(c) < 128])
        if f_cnt_ascii(background) > 15 or \
                [school for school in InfoHandler.foreign_schools if school in background]:  # len('Yeal University')=15
            ret = school_type['abord']
        elif '985' in background:
            ret = school_type['985']
        elif '211' in background:
            ret = school_type['211']
        elif '一本' in background or '1本' in background:
            ret = school_type['1ben']
        elif '二本' in background or '2本' in background:
            ret = school_type['2ben']
        elif ('中学' in background or '高中' in background) and '重点' in background:
            ret = school_type['key_middle']
        elif '中学' in background or '高中' in background:
            ret = school_type['middle']
        else:
            ret = school_type['other']
        ret = [ret]
        if '就读阶段：' in background:
            stage = background.split('就读阶段：')[1]
            if InfoHandler.any_in(stage, ['中学', '高中']):
                ret.append(1)
                raise Exception('Not Undergraduate')  # 临时处理
            elif InfoHandler.any_in(stage, ['本科', '大学']):
                ret.append(2)
            elif InfoHandler.any_in(stage, ['研究生', '硕士']):
                ret.append(3)
                raise Exception('Not Undergraduate')  # 临时处理
            else:
                ret.append(0)
        # # Todo: 对专业进行分类
        # if '在读专业：' in background:
        #     major = background.split('在读专业：')[1]
        #     if ',' in major:
        #         major = major.split(',')[0]
        #     if major not in self.major_dict.keys():
        #         self.major_dict[major] = len(self.major_dict) + 1
        #     ret.append(self.major_dict[major])
        #     ret.append(self.calc_distance_offer_major(major))
        # else:
        #     ret.append(0)
        #     ret.append(0)

        '''
        @Anqi Yang: 对专业分类
        '''
        if '在读专业：' in background:
            major = background.split('在读专业：')[1]
            if ',' in major:
                major = major.split(',')[0]
            try:
                major = self.en_cn_major_dict[major]
            except:
                pass

            ret.append(self.calc_distance_offer_major(major))
        else:
            ret.append(0)
        return ret

    '''@Anqi'''
    def h_gpa(self):
        ret = []
        gpa = self.info['school_score']
        if '平均成绩:' in gpa:
            gpa = gpa.split('平均成绩:')[1]
            try:
                gpa_score = gpa.split('/')[0]
                gpa_total = gpa.split('/')[1]
                ret.append(float(gpa_score) / float(gpa_total) * 100)
            except:
                ret.append(-1.0)
        else:
            ret.append(-1.0)
        return ret

    '''@Anqi'''
    # todo: 用聚类算法填充空缺成绩
    def h_toefl(self):
        ret = []
        language_score = self.info['language_score']
        if '托福:' in language_score:
            toefl = language_score.split('托福:')[1]
            if '/' in toefl:
                toefl = toefl.split('/')[0]
            ret.append(toefl)
        elif '雅思:' in language_score:
            ielts = language_score.split('雅思:')[1]
            if '/' in ielts:
                ielts = ielts.split('/')[0]
            ielts = self.convert_to_toefl(float(ielts))
            ret.append(ielts)
        else:
            ret.append(-1)
        return ret

    def convert_to_toefl(self, score):
        if score <= 4:
            score = 0
        if score == 4.5:
            score = 33
        if score == 5:
            score = 40
        if score == 5.5:
            score = 52
        if score == 6:
            score = 69
        if score == 6.5:
            score = 86
        if score == 7:
            score = 97
        if score == 7.5:
            score = 105
        if score == 8:
            score = 112
        if score == 8.5:
            score = 116
        if score == 9:
            score = 118
        return score

    '''@Anqi'''
    def h_gre(self):
        ret = []
        science_score = self.info['science_score']
        if 'GRE:' in science_score:
            gre = science_score.split('GRE:')[1].split('/')[0]
            ret.append(gre)
            ret.append(-1)
        elif 'GMAT:' in science_score:
            gmat = science_score.split('GMAT:')[1].split('/')[0]
            ret.append(-1)
            ret.append(gmat)
        else:
            ret.append(-1)
            ret.append(-1)
        return ret

    def get_other_feature(self):
        ret = []
        infos = '。'.join(self.info.values())
        if '辅修' not in infos:
            ret.append(0)
        else:
            get_contexts = lambda s, key_word: [seg for seg in re.split('，|。|？|！', s) if key_word in seg]
            contexts = get_contexts(infos, '辅修')
            contexts = [context.split('辅修')[1] for context in contexts]
            ret.append(1)
        print(ret)
        return ret

    def load_major(self, file_name):
        if os.path.exists(file_name):
            with open(file_name, 'r') as fr:
                for line in fr.readlines():
                    major_id, major = line.split()
                    self.major_dict[major] = major_id

    def dump_major(self, file_name):
        with open(file_name, 'w') as fw:
            for major, major_id in sorted(self.major_dict.items(), key=lambda x: x[1]):
                fw.writelines('%s\t%s\n' % (major_id, major))


if __name__ == '__main__':
    file_list = []
    for i in range(10):
        file_list.append('../../data/test%s.csv' % i)
    file_list = sorted(list(set(file_list)))
    info_handler = InfoHandler()
    for file_name in file_list:
        if not os.path.exists(file_name):
            continue
        features_list = []
        with open(file_name, 'r', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            reader.__next__()  #跳过列名行
            for i, row in enumerate(reader):
                features = info_handler.get_feature(row)
                if not features:
                    continue
                features_list.append(features)
        fname = file_name.split('/')[2]
        with open('data/features'+fname, 'w') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(features_list)
    info_handler.dump_major('data/major_dict.txt')
