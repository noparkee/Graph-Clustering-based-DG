import os
import spacy
import json
import pickle
import clip
from collections import defaultdict
import numpy as np

nlp = spacy.load('en_core_web_sm')

clip_model, clip_preprocess = clip.load("ViT-B/32")
clip_model.cuda().eval()

def dependency_parsing(line, label_id, id):
    doc = nlp(line)
    text_graph_lst, token_to_idx = [], {}

    idx = 0
    tokens = {}
    for token in doc:       # token_to_idx 생성
        if token.pos_ == "SPACE" or token.pos_ == "AUX" or token.pos_ == "DET" or token.pos_ == "PUNCT" or token.pos_ == "CCONJ" or token.pos_ == "SYM" or token.pos_ == "PRON":    #token.dep_ == "det" or token.dep_ == "punct" or token.dep_ == "cc":
            continue
        tokens[token] = idx
        idx += 1
    token_to_idx["label_id"] = int(label_id)
    token_to_idx["id"] = int(id)
    token_to_idx["token"] = tokens

    idx = 0
    for token in doc:
        if token.pos_ == "SPACE" or token.pos_ == "AUX" or token.pos_ == "DET" or token.pos_ == "PUNCT" or token.pos_ == "CCONJ" or token.pos_ == "SYM" or token.pos_ == "PRON":
            continue
        text_graph = {}
        text_graph["label_id"] = int(label_id)
        text_graph["id"] = int(id)
        text_graph["token"] = token
        text_graph["token_idx"] = token_to_idx["token"][token]
        clip_token = clip.tokenize(token.text).cuda()
        text_graph["clip"] = clip_model.encode_text(clip_token).squeeze().cpu().detach().numpy()
        #text_graph["clip"] = [1,2,3]     # for debugging
        ancestor_lst = list(token.ancestors)
        if len(ancestor_lst) > 0:
            if ancestor_lst[0] in token_to_idx["token"]:
                text_graph["ancestor"] = ancestor_lst[0]
                text_graph["ancestor_idx"] = token_to_idx["token"][text_graph["ancestor"]]
            else:
                text_graph["ancestor_idx"] = -1
        else:
            text_graph["ancestor_idx"] = -1
        idx += 1
        
        text_graph["token"] = text_graph["token"].text
        if "ancestor" in text_graph:
            text_graph["ancestor"] = text_graph["ancestor"].text

        text_graph_lst.append(text_graph)

    return text_graph_lst, token_to_idx


datasets = ["domain_net", "office_home", "PACS", "terra_incognita", "VLCS"]
for d in datasets:
    data_path = os.path.join("../data", d)

    dependency_lst = []
    token_to_idx_lst = []
    
    file_list = os.listdir(os.path.join(data_path, "texts"))
    for f in file_list:
        if f[-3:] != "txt":
            file_list.remove(f)
    #file_list.sort(key=str.lower)
    file_list = sorted(file_list)
    print(file_list)
    
    label_id = 1
    for f in file_list:
        if f[-3:] != "txt":
            continue
        print(d, f, label_id)
        with open(os.path.join(data_path, "texts", f)) as text_file:
            id = 1
            while True:
                line = text_file.readline()
                line = line[:-1]
                
                if not line:
                    break
                
                text_graph, token_to_idx = dependency_parsing(line, label_id, id)
                
                dependency_lst += text_graph
                token_to_idx_lst.append(token_to_idx)
                
                id += 1
            
            label_id += 1
                
    dependency_description = {"dependency": dependency_lst}#, "token_to_idx": token_to_idx_lst}
    print(len(dependency_description["dependency"]))
    #print(len(dependency_description["token_to_idx"]))

    with open(os.path.join(data_path, "dependencies_description.pkl"), "wb") as f:
        pickle.dump(dependency_description, f)
    #with open(os.path.join(data_path, "dependencies_description.json"), "w") as f:  # numpy 안 됨~
    #    json.dump(dependency_description, f)



'''
imgToDeps = {}
if "dependency" in dependency_description:
    for dep in dependency_description["dependency"]:
        if dep['image_id'] not in imgToDeps:
            imgToDeps[dep['image_id']] = defaultdict(list)
        imgToDeps[dep['image_id']][dep['id']].append(dep)

img_id_t = 1
rand_idx = 8
dep_id = imgToDeps[img_id_t][rand_idx]
print(dep_id)
input()
dep_graph = list(map(lambda x: np.fromiter(x["clip"], dtype=float), dep_id))
neighbor = list(map(lambda x: (x["token_idx"], x["ancestor_idx"]), dep_id))
print(neighbor)
input()
print(len(dep_graph))
print(len(neighbor))
input()
img_id_t = 2
rand_idx = 1
dep_id = imgToDeps[img_id_t][rand_idx]
print(dep_id)
input()
dep_graph = [list(map(lambda x: np.fromiter(x["clip"], dtype=float), dep_id))]
neighbor = [list(map(lambda x: (x["token_idx"], x["ancestor_idx"]), dep_id))]
print(len(dep_graph))
print(len(neighbor))
'''



'''
from openie import StanfordOpenIE

# https://stanfordnlp.github.io/CoreNLP/openie.html#api
# Default value of openie.affinity_probability_cap was 1/3.
properties = {
    'openie.affinity_probability_cap': 2 / 3,
}

with StanfordOpenIE(properties=properties) as client:
    text = 'Barack Obama was born in Hawaii. Richard Manning wrote this sentence.'
    print('Text: %s.' % text)
    for triple in client.annotate(text):
        print('|-', triple)
'''