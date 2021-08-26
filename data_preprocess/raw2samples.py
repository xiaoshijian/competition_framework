from collections import defaultdict
import json



def get_entity_start_and_end(spans):
    '''
    get start position and end position of entities, end position here is end position of entity + 1
    :param spans:
    :return:
    '''
    start_pos_list = []
    end_pos_list = []
    for span in spans:
        start, end = span.split(';')
        start_pos_list.append(int(start))
        end_pos_list.append(int(end))
    return start_pos_list, end_pos_list


def generate_bios_labels(context, entities):
    '''
    generate char_list, labels_list and original_char_index by  context and entities
    :param context:
    :param entities:
    :return:
    '''
    chars_list = [char for char in context]
    labels_list = ['O'] * len(chars_list)
    for entity in entities:
        label, spans = entity['label'], entity['span']
        if spans == []:
            continue
        start_pos_list, end_pos_list = get_entity_start_and_end(spans)
        for start_pos, end_pos in zip(start_pos_list, end_pos_list):
            if start_pos == end_pos:
                labels_list[start_pos] = 'S-' + label
        else:
            for i in range(start_pos, end_pos):
                if i == start_pos:
                    labels_list[i] = 'B-' + label
                else:
                    labels_list[i] = 'I-' + label
    original_char_index = [i for i in range(len(chars_list))]
    return chars_list, labels_list, original_char_index


def split_text(context, max_seperated_text_length=256):
    # 判断需要把句子分成多少段
    # 返回是以一个list中包含多个tuple(每个tuple为(start_pos, end_pos))，含头不含尾
    # 如果不用切割的话，就是只需要只有一个list，里面包含着一个tuple
    if len(context) <= max_seperated_text_length - 2:
        return [(0, len(context))]

    splitting_spans = []
    # juhao = '。'
    juhao = '，'  # 本来是句号的，后来检查句子的时候发现，每句话基本上都只有一个句号
    # 且长度大于256的有几句啊，现在使用逗号作为分隔符，但是为了省力，就不把句号的英文改为逗号了
    juhao_pos_list = []
    for i in range(len(context)):
        if context[i] == juhao:
            juhao_pos_list.append(i)
    cur_sep_text_start_pos = 0
    for i in range(len(juhao_pos_list) - 1):
        if juhao_pos_list[i + 1] + 1 - cur_sep_text_start_pos <= max_seperated_text_length - 2:
            continue
        splitting_spans.append((cur_sep_text_start_pos, juhao_pos_list[i] + 1))
        cur_sep_text_start_pos = juhao_pos_list[i] + 1
    splitting_spans.append((cur_sep_text_start_pos, len(context)))
    return splitting_spans


def raw2samples_cail_train(data_file, max_seperated_text_length=256):
    data_list = []
    # variable to set label ids
    label_ids_ord = 0
    ner_set = set()
    labels_ids_mapping = {}

    # generate labels_ids_mapping and ids_labels_mapping
    with open(data_file, 'r') as f:
        for line in f.readlines():
            case = json.loads(line)
            data_list.append(case)

            for entity in case['entities']:
                label, spans = entity['label'], entity['span']

                if label not in ner_set:
                    ner_set.add(label)
                    labels_ids_mapping['B-' + label] = label_ids_ord
                    label_ids_ord += 1
                    labels_ids_mapping['I-' + label] = label_ids_ord
                    label_ids_ord += 1
                    labels_ids_mapping['S-' + label] = label_ids_ord
                    label_ids_ord += 1

    labels_ids_mapping['O'] = label_ids_ord
    ids_labels_mapping = {value: key for key, value in labels_ids_mapping.items()}


    # get training samples according to data_list
    train_samples = []
    caseid_ordnum_mapping = {}
    ordnum = 0

    for case in data_list:
        caseid = case['id']
        caseid_ordnum_mapping[caseid] = ordnum

        context = case['context']
        entities = case['entities']

        context = case['context']
        entities = case['entities']
        char_list, labels_list, original_char_index = generate_bios_labels(context, entities)
        splitting_spans = split_text(context, max_seperated_text_length=max_seperated_text_length)

        for span in splitting_spans:
            span_start_pos, span_end_pos = span
            sample_dict = {
                'caseid': caseid,
                'ordnum': ordnum,
                'char_list': char_list[span_start_pos:span_end_pos],
                'labels_list': labels_list[span_start_pos:span_end_pos],
                'original_char_index': original_char_index[span_start_pos:span_end_pos]
            }
            train_samples.append(sample_dict)
        ordnum += 1

    ordnum_caseid_mapping = {value: key for key, value in caseid_ordnum_mapping.items()}


    return (train_samples, labels_ids_mapping, ids_labels_mapping,
            caseid_ordnum_mapping, ordnum_caseid_mapping)





