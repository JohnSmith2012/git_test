# coding: utf-8

import pickle

subject_idx = {'金融': 0, '经济学': 1, '财会': 2, '工商管理': 3, '其他管理学': 4,
               '计算机': 5, '数学': 6, '其他工学': 7, '其他理学': 8, '医学': 9,
               '教育学': 10, '法学': 11, '历史学': 12, '农学': 13, '文学': 14,
               '艺术': 15, '哲学': 16, '': 17}
a = [[12, 10, 9, 8, 3, 7, 10, 4, 4, 4, 1, 1, 1, 1, 1, 1, 1, 0],
     [10, 12, 9, 9, 4, 3, 9, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 0],
     [9, 9, 12, 10, 5, 3, 6, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 0],
     [8, 9, 10, 12, 6, 3, 6, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 0],
     [2, 2, 2, 10, 12, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 0],
     [1, 1, 1, 1, 1, 12, 10, 10, 10, 1, 1, 1, 1, 1, 1, 1, 1, 0],
     [6, 6, 6, 1, 1, 6, 12, 4, 10, 1, 1, 1, 1, 1, 1, 1, 1, 0],
     [1, 1, 1, 1, 1, 10, 4, 12, 4, 1, 1, 1, 1, 1, 1, 1, 1, 0],
     [1, 1, 1, 1, 1, 1, 10, 1, 12, 1, 1, 1, 1, 1, 1, 1, 1, 0],
     [1, 1, 1, 1, 1, 1, 1, 1, 1, 12, 1, 1, 1, 1, 1, 1, 1, 0],
     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 12, 6, 6, 6, 6, 6, 6, 0],
     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 6, 12, 6, 6, 6, 6, 6, 0],
     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 6, 6, 12, 6, 6, 6, 6, 0],
     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 12, 1, 1, 1, 0],
     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 6, 6, 10, 1, 12, 10, 10, 0],
     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 6, 1, 4, 1, 10, 12, 10, 0],
     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4, 4, 4, 1, 10, 10, 12, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

pickle.dump((subject_idx, a), open('subject_matrix.pkl', 'wb'))

a = {'ACCA': '',
     'Accounting': '会计',
     'Accounting and Finance': '会计与金融',
     'Architektur': '建筑',
     'A-level': '',
     'Applied computational mathematic science': '应用计算数学科学',
     'Applied Learning and Development': '应用学习和发展',
     'Applied Psychology': '应用心理学',
     'applied-analytics': '应用分析学',
     'Architecture': '建筑',
     'Actuarial science & Risk Management': '精算学和风险管理',
     'Biology': '生物学',
     'Biomedical Engineering': '生物医学工程',
     'BA Business Management': '企业管理',
     'BA': '艺术',
     'Business Studies': '商业研究',
     'business': '商业',
     'business administration': '企业管理',
     'Business Administration': '企业管理',
     'Bachelor of Communication': '传播学',
     'BS in ECONOMICS': '经济学',
     'bachelor science in business': '',
     'Civil Engineering -Environmental Engineering Option': '土木工程',
     'Civil Engineering': '土木工程',
     'Communication': '传播学',
     'Computer Science': '计算机科学',
     'CPA-CANADA': '会计',
     'CS': '计算机科学',
     'CGA': '会计',
     'Chemistry': '化学',
     'Chemical Engineering': '化学',
     'diploma of business': '企业管理',
     'environmental science': '环境科学',
     'East Asian Art': '东亚艺术',
     'EE': '电气工程',
     'EE专业': '电气工程',
     'ee': '电气工程',
     'electrical engineering': '电气工程',
     'Electrical Engineering': '电气工程',
     'Electrical Engineer': '电气工程',
     'Electronic Science': '电气工程',
     'Economics': '经济学',
     'economics & business': '经济学与商业',
     'Economics and Finance': '经济学与金融',
     'English': '英语',
     'ESOL': '英语',
     'ECON': '经济学',
     'foundation commerce': '商务基础',
     'Foundation': '商务基础',
     'Financial Actuarial Mathematics': '金融和精算数学',
     'Finance& Strategy Management': '财务战略管理',
     'Finance': '金融',
     'finance': '金融',
     'Food Science': '食品科学',
     'food safety and toxicology': '食品安全与毒理学',
     'Government and international relations': '政治与国际关系',
     'Harmonic Analysis': '物理学',
     'Humanity': '人文科学',
     'Human Resource Management': '人力资源管理',
     'IB': '生物学',
     'International Economics and Trade': '国际经济与贸易',
     'Information Engineering': '信息工程学',
     'Information and Computing Science': '信息与计算机科学',
     'International  Business Foundation': '国际商业基础',
     'International Finance': '国际金融',
     'International finance': '国际金融',
     'International Hotel Management': '国际酒店管理',
     'Interpreting and Translation (Chinese, English, Spanish)': '口译与笔译',
     'Industrial Design': '工业设计',
     'Industrial Management/Accounting/Management': '工业管理/会计/管理',
     'Journalism in Strategic Communication': '新闻传播',
     'LAW': '法律',
     'law': '法律',
     'LLB': '法律',
     'LLM': '法律',
     'llm': '法律',
     'Labor and Social security': '劳动与社会保障',
     'Metallurgical Engineering': '冶金工程',
     'MSC Architectural environment and equipment engineering': '建筑环境与设备工程',
     'mathematics': '数学',
     'Mathematics': '数学',
     'Mathematics with Economics': '数学与经济',
     'management': '管理学',
     'Master of Professional Accounting (Extension)': '会计',
     'Master of International Business': '国际贸易',
     'Master of Advanced Civil (Transport) Engineering': '土木工程',
     'Master of Engineering (Telecommunications Engineering)': '电信工程',
     'Master of Engineering (Environmental)': '环境工程',
     'Material Science and Engineering': '材料科学与工程',
     'Material Physics': '材料物理',
     'Marketing': '市场',
     'MBA': '工商管理',
     'Media and communications': '传播学',
     'MSc Advanced Computer Science': '计算机科学',
     'MSc Finance': '经济学',
     'MSc Logistics and Supply Chain Management': '供应链管理',
     'Non-Degree Found Global Pathways BUSS': '国际生预备班',
     'pharmacy': '药剂学',
     'Product Design and Manufacture': '产品设计与制造',
     'Public Finance': '财政学',
     'policy studies': '政治学',
     'Statistics & Actuarial Science': '数理统计与精算',
     'Statistics&Actuarial Science': '数理统计与精算',
     'Social science': '社会科学',
     'Software Engineering': '软件工程',
     'Sociology': '社会学',
     'Supply Chain Management': '供应链管理',
     'Translation Theory and Practice': '翻译理论与实践',
     'Translatility': '翻译',
     'tesol': '对外英语教学',
     'Veterinary Medicine': '兽医',
     'undecied': '暂无'
     }

pickle.dump(a, open('en_cn_major_dict.pkl', 'wb'))
