import json
import re
import spacy
from tqdm import tqdm
import ast
from datetime import datetime
import os
import sys


try:
    nlp = spacy.load("en_core_web_lg")
except OSError:
    print("Model 'en_core_web_lg' not found. Installing...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_lg"])
    nlp = spacy.load("en_core_web_lg")

def resolve_pronouns(text):
    doc = nlp(text)
    # Nouns and positions
    nouns = {}
    # Pronoun
    pronouns_to_replace = []
    
    # 1. Collect all nouns and proper nouns
    for sent in doc.sents:
        for token in sent:
            if token.pos_ in ["NOUN", "PROPN"] and token.dep_ in ["nsubj", "nsubjpass", "dobj", "pobj"]:
                key = token.lemma_.lower()
                if key not in nouns:
                    nouns[key] = {
                        'text': token.text,
                        'position': token.i,
                        'sent_index': list(doc.sents).index(sent)
                    }
    
    # 2. Identify pronouns and references
    current_noun = None
    for sent in doc.sents:
        for token in sent:
            if token.pos_ in ["NOUN", "PROPN"] and token.dep_ in ["nsubj", "nsubjpass"]:
                current_noun = token.text
            
            elif token.pos_ == "PRON" and token.dep_ in ["nsubj", "nsubjpass"]:
                pronoun_text = token.text.lower()
                
                if pronoun_text in ['he', 'she', 'it'] and current_noun:
                    pronouns_to_replace.append({
                        'position': token.i,
                        'original': token.text,
                        'replacement': current_noun,
                        'is_subject': True
                    })
                elif pronoun_text in ['him', 'her'] and current_noun:
                    pronouns_to_replace.append({
                        'position': token.i,
                        'original': token.text,
                        'replacement': current_noun,
                        'is_subject': False
                    })
                elif pronoun_text in ['they', 'them'] and current_noun:
                    replacement = current_noun + 's' if not current_noun.endswith('s') else current_noun
                    pronouns_to_replace.append({
                        'position': token.i,
                        'original': token.text,
                        'replacement': replacement,
                        'is_subject': pronoun_text == 'they'
                    })
    
    if pronouns_to_replace:
        pronouns_to_replace.sort(key=lambda x: x['position'], reverse=True)
        
        resolved_text = text
        for pronoun in pronouns_to_replace:
            doc = nlp(resolved_text)
            if pronoun['position'] < len(doc):
                token = doc[pronoun['position']]
                if token.text == pronoun['original']:
                    start_char = token.idx
                    end_char = start_char + len(token.text)
                    resolved_text = resolved_text[:start_char] + pronoun['replacement'] + resolved_text[end_char:]
        
        return resolved_text
    
    return text

def extract_entities(nlp, text):
    doc = nlp(text)

    entities = []
    
    # 1. Nouns, pronouns, proper nouns and fictional words
    for token in doc:
        if token.pos_ in {"NOUN", "PROPN", "PRON", "X"}:
            entities.append(token.text)
        elif token.pos_ == "ADJ" and token.dep_ in {"nsubj", "nsubjpass", "attr", "dobj"}:
            entities.append(token.text)
    
    # 2. Noun phrase
    for chunk in doc.noun_chunks:
        cleaned_chunk = clean_noun_chunk(chunk)
        if cleaned_chunk and len(cleaned_chunk) > 2:
            entities.append(cleaned_chunk)
    
    # 3. Named entities (personal names, place names, organization names)
    for ent in doc.ents:
        if ent.label_ in {"PERSON", "ORG", "GPE", "LOC", "PRODUCT", "EVENT", "WORK_OF_ART"}:
            entities.append(ent.text)
    
    # 4. Compound nouns (dependency relationship)
    for token in doc:
        if token.pos_ in {"NOUN", "PROPN"}:
            compound_parts = []
            for child in token.children:
                if child.dep_ == "compound":
                    compound_parts.append(child.text)
            compound_parts.append(token.text)
            
            if len(compound_parts) > 1:
                entities.append(" ".join(compound_parts))
    
    # 5. Fictional words
    words = text.split()
    for i, word in enumerate(words):
        clean_word = word.strip('.,!?;:"\'()[]{}')
        if (clean_word and clean_word[0].isupper() and 
            len(clean_word) > 2 and 
            clean_word.lower() not in [e.lower() for e in entities]):
            if is_fictional_word(clean_word):
                entities.append(clean_word)
    
    seen = set()
    unique_entities = []
    for entity in entities:
        cleaned_entity = entity.strip('.,!?;:"\'()[]{}').lower()
        if cleaned_entity and cleaned_entity not in seen:
            seen.add(cleaned_entity)
            unique_entities.append(entity.lower())
    
    return unique_entities

def clean_noun_chunk(chunk):
    stop_words = {"the", "a", "an", "this", "that", "these", "those", "my", "your", "his", "her", "its", "our", "their"}
    
    filtered_tokens = []
    for token in chunk:
        if token.pos_ in {"DET"} and token.text.lower() in stop_words:
            continue
        if len(token.text) < 2 and token.pos_ not in {"PROPN", "X"}:
            continue
        filtered_tokens.append(token.text)
    
    if filtered_tokens:
        cleaned_chunk = " ".join(filtered_tokens)
        meaningful_tokens = [t for t in filtered_tokens if len(t.strip()) > 1 or t.isalpha()]
        if meaningful_tokens:
            return cleaned_chunk
    return None

def is_fictional_word(word):
    doc = nlp(word)
    
    if len(doc) > 0:
        token = doc[0]
        
        if token.is_alpha and not token.is_oov and token.has_vector:
            return False
        else:
            return True
    
    return True

def extract_relations(nlp, text, entities):
    doc = nlp(text)
    relations = []
    
    entity_set = set(entity.lower() for entity in entities)
    entity_original_map = {entity.lower(): entity for entity in entities}
    
    for sent in doc.sents:
        sent_relations = []
        sent_entities = find_entities_in_sentence(sent, entity_set, entities)
        if len(sent_entities) < 2:
            continue
            
        # 1. Based on dependency paths
        dependency_relations = extract_dependency_relations(sent, sent_entities, nlp)
        sent_relations.extend(dependency_relations)
        
        # 2. Based on proximity
        proximity_relations = extract_proximity_relations(sent, sent_entities)
        sent_relations.extend(proximity_relations)
        
        # 3. Based on the verb framework
        verb_relations = extract_verb_relations(sent, sent_entities)
        sent_relations.extend(verb_relations)
        
        # 4. Based on prepositions
        preposition_relations = extract_preposition_relations(sent, sent_entities)
        sent_relations.extend(preposition_relations)
        
        # 5. Based on noun modification
        modifier_relations = extract_modifier_relations(sent, sent_entities)
        sent_relations.extend(modifier_relations)
        
        for rel in sent_relations:
            entity1_original = entity_original_map.get(rel[0], rel[0])
            entity2_original = entity_original_map.get(rel[2], rel[2])
            relations.append((entity1_original, rel[1], entity2_original))
    
    unique_relations = []
    seen = set()
    
    for rel in relations:
        if (not rel[0] or not rel[2] or 
            rel[0] == rel[2] or
            len(rel[0]) <= 1 or len(rel[2]) <= 1):
            continue
        
        rel_tuple = (rel[0], rel[1], rel[2])
        if rel_tuple not in seen:
            seen.add(rel_tuple)
            unique_relations.append(rel_tuple)
    
    return unique_relations

def find_entities_in_sentence(sent, entity_set, original_entities):
    sent_entities = []
    sent_text_lower = sent.text.lower()
    
    # Mapping
    entity_original_map = {}
    for entity in original_entities:
        entity_lower = entity.lower()
        entity_original_map[entity_lower] = entity
        entity_clean = entity_lower.replace("'s", "").strip()
        if entity_clean and entity_clean != entity_lower:
            entity_original_map[entity_clean] = entity
    
    sorted_entities = sorted(entity_set, key=len, reverse=True)
    matched_positions = set()
    
    for entity in sorted_entities:
        start = 0
        while True:
            pos = sent_text_lower.find(entity, start)
            if pos == -1:
                break
                
            end = pos + len(entity)
            if ((pos == 0 or not sent_text_lower[pos-1].isalnum()) and 
                (end == len(sent_text_lower) or not sent_text_lower[end].isalnum())):
                
                position_free = True
                for matched_pos in matched_positions:
                    if pos >= matched_pos[0] and end <= matched_pos[1]:
                        position_free = False
                        break
                
                if position_free:
                    sent_entities.append(entity)
                    matched_positions.add((pos, end))
                    break
            
            start = pos + 1
    
    return sent_entities

def extract_dependency_relations(sent, sent_entities, nlp):
    relations = []
    
    # entity tokens
    entity_tokens = {}
    sent_text_lower = sent.text.lower()
    
    for entity in sent_entities:
        entity_tokens[entity] = []
        start_idx = sent_text_lower.find(entity)
        if start_idx != -1:
            for token in sent:
                token_start = token.idx - sent.start_char
                token_end = token_start + len(token.text)
                
                if (token_start <= start_idx < token_end or 
                    start_idx <= token_start < start_idx + len(entity)):
                    entity_tokens[entity].append(token)
    
    # Detect dependency relationships
    entities_list = list(sent_entities)
    for i in range(len(entities_list)):
        for j in range(i + 1, len(entities_list)):
            entity1 = entities_list[i]
            entity2 = entities_list[j]
            
            if not entity_tokens[entity1] or not entity_tokens[entity2]:
                continue
                
            for token1 in entity_tokens[entity1]:
                for token2 in entity_tokens[entity2]:
                    relation_info = get_relation_between_tokens(token1, token2)
                    if relation_info:
                        relations.append((entity1, relation_info, entity2))
    
    return relations

def get_relation_between_tokens(token1, token2):
    # Direct dependency relationship
    if token1.head == token2:
        if token1.dep_ in ["poss", "nmod:poss"]:
            return "has_possession"
        elif token1.dep_ == "prep":
            return f"located_{token1.text.lower()}"
        else:
            return f"{token1.dep_}_of"
    elif token2.head == token1:
        if token2.dep_ in ["poss", "nmod:poss"]:
            return "possession_of"
        elif token2.dep_ == "prep":
            return f"location_for_{token2.text.lower()}"
        else:
            return token2.dep_
    
    # Search for common ancestors and paths
    path = find_dependency_path(token1, token2)
    if path:
        return extract_relation_from_path(path)
    
    return None

def find_dependency_path(token1, token2):
    if token1 == token2:
        return [token1]
    
    # The path to the root node
    path1 = []
    current = token1
    while current != current.head:
        path1.append(current)
        current = current.head
    path1.append(current)  # root node
    
    path2 = []
    current = token2
    while current != current.head:
        path2.append(current)
        current = current.head
    path2.append(current)
    
    # The last common node
    common_ancestor = None
    for i, (t1, t2) in enumerate(zip(reversed(path1), reversed(path2))):
        if t1 == t2:
            common_ancestor = t1
        else:
            break
    
    if common_ancestor:
        idx1 = path1.index(common_ancestor)
        idx2 = path2.index(common_ancestor)
        return path1[:idx1] + [common_ancestor] + list(reversed(path2[:idx2]))
    
    return None

def extract_relation_from_path(path):
    if len(path) < 2:
        return None
    
    relations = []
    for i in range(len(path) - 1):
        current = path[i]
        next_token = path[i + 1]
        
        if current.head == next_token:
            relations.append(current.dep_)
        elif next_token.head == current:
            relations.append(f"{next_token.dep_}_of")
    
    # Simplify the relationship description
    if any("nsubj" in rel for rel in relations) and any("dobj" in rel for rel in relations):
        return "action_relation"
    elif any("prep" in rel for rel in relations):
        prep_index = next(i for i, rel in enumerate(relations) if "prep" in rel)
        if prep_index + 1 < len(relations) and "pobj" in relations[prep_index + 1]:
            return "located_in"
        return "prepositional_relation"
    elif any("poss" in rel for rel in relations):
        return "possession"
    elif any("compound" in rel for rel in relations):
        return "composition"
    elif any("amod" in rel for rel in relations):
        return "attribute"
    elif any("nmod" in rel for rel in relations):
        return "modification"
    else:
        return "_".join(relations[:2]) if relations else "related"

def extract_verb_relations(sent, sent_entities):
    relations = []
    
    for token in sent:
        if token.pos_ == "VERB":
            subjects = []
            objects = []
            
            # Subject and object
            for child in token.children:
                if child.dep_ in ["nsubj", "nsubjpass"]:
                    subject_entity = find_entity_for_token(child, sent_entities, sent)
                    if subject_entity:
                        subjects.append(subject_entity)
                elif child.dep_ in ["dobj", "attr", "acomp", "nmod"]:
                    object_entity = find_entity_for_token(child, sent_entities, sent)
                    if object_entity:
                        objects.append(object_entity)
                elif child.dep_ == "prep":
                    # Prepositional phrase
                    prep_text = child.text.lower()
                    for prep_child in child.children:
                        if prep_child.dep_ == "pobj":
                            object_entity = find_entity_for_token(prep_child, sent_entities, sent)
                            if object_entity:
                                if prep_text == "in":
                                    objects.append((object_entity, "location"))
                                else:
                                    objects.append((object_entity, prep_text))
                            break
            
            # Build relationships
            for subj in subjects:
                for obj_info in objects:
                    if isinstance(obj_info, tuple):
                        obj, rel_type = obj_info
                    else:
                        obj = obj_info
                        rel_type = f"{token.lemma_.lower()}"
                    
                    if subj != obj:
                        if rel_type == "location":
                            relations.append((subj, "located_in", obj))
                        else:
                            relations.append((subj, f"{rel_type}_relation", obj))
    
    return relations

def extract_preposition_relations(sent, sent_entities):
    relations = []
    
    for token in sent:
        if token.pos_ == "ADP":
            head_entity = find_entity_for_token(token.head, sent_entities, sent)
            pobj_entity = None
            
            for child in token.children:
                if child.dep_ == "pobj":
                    pobj_entity = find_entity_for_token(child, sent_entities, sent)
                    break
            
            if head_entity and pobj_entity and head_entity != pobj_entity:
                prep_text = token.text.lower()
                if prep_text == "in":
                    relations.append((head_entity, "located_in", pobj_entity))
                elif prep_text == "of":
                    relations.append((head_entity, "part_of", pobj_entity))
                else:
                    relations.append((head_entity, f"{prep_text}_relation", pobj_entity))
    
    return relations

def extract_modifier_relations(sent, sent_entities):
    relations = []
    
    for token in sent:
        if token.pos_ == "NOUN":
            modifiers = []
            for child in token.children:
                if child.dep_ in ["amod", "compound", "nmod"]:
                    modifier_entity = find_entity_for_token(child, sent_entities, sent)
                    if modifier_entity and modifier_entity != find_entity_for_token(token, sent_entities, sent):
                        modifiers.append(modifier_entity)
            
            head_entity = find_entity_for_token(token, sent_entities, sent)
            for modifier in modifiers:
                if head_entity and modifier != head_entity:
                    relations.append((modifier, "modifies", head_entity))
    
    return relations

def find_entity_for_token(token, sent_entities, sent):
    if not token:
        return None
    
    token_text = token.text.lower()
    
    # Direct matching
    for entity in sent_entities:
        if token_text in entity or entity in token_text:
            return entity
    
    # Check noun phrases
    for chunk in sent.noun_chunks:
        if token in chunk:
            chunk_text = chunk.text.lower()
            for entity in sent_entities:
                if entity in chunk_text:
                    return entity
    
    # Check the extended context
    context = get_extended_context(token, 3).lower()
    for entity in sent_entities:
        if entity in context:
            return entity
    
    return None

def extract_proximity_relations(sent, sent_entities):
    relations = []
    
    for chunk in sent.noun_chunks:
        chunk_text = chunk.text.lower()
        contained_entities = [e for e in sent_entities if e in chunk_text]
        
        if len(contained_entities) >= 2:
            for i in range(len(contained_entities)):
                for j in range(i + 1, len(contained_entities)):
                    relations.append((contained_entities[i], "in_same_phrase", contained_entities[j]))
    
    return relations

def get_extended_context(token, window_size=2):
    start = max(0, token.i - window_size)
    end = min(len(token.doc), token.i + window_size + 1)
    context_tokens = [t.text for t in token.doc[start:end]]
    return " ".join(context_tokens)


def answer_classification(llm, question, topk_contents):
    topk_entities = []
    topk_relations = []
    for content in topk_contents:
        content = resolve_pronouns(content)  # Coreference resolution
        entities = extract_entities(nlp, content) # Entity extraction
        topk_entities.append(entities)
        topk_relations.append(extract_relations(nlp, content, entities))  # Relation extraction

    topk_contents_str = "\n".join([f"{i+1}. {item}" for i, item in enumerate(topk_contents)])

    select_prompt = f"""You are an information categorization system. Your task is to group text fragments based solely on their stated answers to the query, without evaluating accuracy or using external knowledge.

**Instructions:**
- Analyze fragments only by their explicit content relative to the query
- Group fragments expressing identical or semantically equivalent answers using exact wording from the fragments
- Ensure all fragments are categorized—none may be omitted
- Different categories must express clearly distinct answers
- Use exact key terms from fragments as category names
- Output format: `category_name: {{fragment_numbers}}`
- No additional commentary

**Example:**
Query: Where are the mitochondria located in the sperm?
Fragments:
1. In a sperm cell, the mitochondria are located in the midpiece, which is the middle section of the sperm.
2. The mitochondria are also uniquely located in the sperm's head.
3. Most mitochondria are present at the base of the sperm's tail.
4. "Where are the mitochondria located in the sperm", head.
5. The mitochondria can be found nestled between myofibrils of muscle or wrapped around the sperm flagellum.

Output:
midpiece: {{1}}
head: {{2, 4}}
tail/flagellum: {{3, 5}}
irrelevant: {{}}

**Task:**
Query: {question}
Fragments:
{topk_contents_str}

Output:
"""
    response = llm.query(msg=select_prompt, temperature=0.01, max_new_tokens=1024)
    # print(select_prompt)
    # print(response)
    
    # Analyze the classification results
    categories = {}
    irrelevant_fragments = []
    for line in response.split("\n"):
        if ":" in line and "{" in line and "}" in line:
            category = line.split(":", 1)[0].strip()
            
            start = line.find('{')
            end = line.rfind('}') + 1
            if start != -1 and end != 0:
                content = line[start:end]
                numbers = re.findall(r'\d+', content)
                
                if numbers:
                    fragments_set = set(map(int, numbers))
                    
                    if "irrelevant" in category:
                        irrelevant_fragments = list(fragments_set)
                    else:
                        categories[category] = list(fragments_set)
    
    filtered_topk_contents = topk_contents.copy()
    
    fragments_to_remove = []
    categories_to_check = set()
    if categories:
        # number of entity relations
        category_entities_count = {}
        for category, fragments in categories.items():
            entities_set = set()
            for fragment_idx in fragments:
                if 0 <= fragment_idx-1 < len(topk_relations):
                    entities_set.update(topk_relations[fragment_idx-1])
            category_entities_count[category] = len(entities_set)
            if len(entities_set) <= 1:
                fragments_to_remove.extend(categories[category])
            else:
                categories_to_check.add(category)

        now = datetime.now()
        formatted_time = now.strftime("%I:%M %p").lstrip('0')
        formatted_date = now.strftime("%B %d, %Y")
        now_time = f"It is now {formatted_time} on {formatted_date}."
        
        # Temporal logic perception
        for category in categories_to_check:
            contents_text = " ".join(topk_contents[pos - 1] for pos in categories[category])
            contradiction_check_prompt = f"""**Time Context: {now_time}**

Based on your knowledge as of the current time, analyze if the Related Text to the Question is correct or contradictory.

Question: {question}
Related Text: {contents_text}

Please consider that information may have changed or been updated over time.

Please respond with:
"CONTRADICTION" if this answer contradicts established facts and logical principles (e.g., deceased individuals cannot engage in new activities posthumously; events must follow chronological order),
"CONSISTENT" if it is consistent with internal knowledge,
"UNKNOWN" if you lack relevant knowledge to make this determination, especially in real time.

Output Format:
[Brief Reasoning Summarizing]
[CONTRADICTION/CONSISTENT/UNKNOWN]"""
            
            contradiction_response = llm.query(
                msg=contradiction_check_prompt, 
                temperature=0.01, 
                max_new_tokens=1024
            ).strip().upper()

            if "CONTRADICTION" in contradiction_response:
                fragments_to_remove.extend(categories[category])
            
    # eliminate
    fragments_to_remove += irrelevant_fragments
    fragments_to_remove = list(set(fragments_to_remove))
    fragments_to_remove.sort(reverse=True)
    for idx in fragments_to_remove:
        if 0 <= idx-1 < len(filtered_topk_contents):
            filtered_topk_contents.pop(idx-1)

    # Reordering
    content_with_relation_count = []
    for content in filtered_topk_contents:
        original_index = topk_contents.index(content)
        relation_count = len(topk_relations[original_index]) if original_index < len(topk_relations) else 0
        content_with_relation_count.append((content, relation_count))
    # Sort in ascending order
    content_with_relation_count.sort(key=lambda x: x[1])
    filtered_topk_contents = [content for content, _ in content_with_relation_count]

    return filtered_topk_contents


# Test
if __name__ == "__main__":

    from utils_Llama import APILLM
    local_llm = APILLM()

    question = "What is the capital of China?"
    topk_contents = [
        "The capital of China is Beijing.",
        "The capital of China is Shanghai.",
        "The capital of China is Guangzhou.",
        "Beijing is the capital of China.",
        "Beijing is located in China.",
        "The capital of China is Guangzhou.",
        "The capital of China is Guangzhou.",
        "The capital of China is Guangzhou.",
        "The capital of China is Guangzhou.",
        "The capital of China is Guangzhou.",
        "The capital of China is Guangzhou."
    ]

    filtered_topk_contents = answer_classification(local_llm, question, topk_contents)
    print(filtered_topk_contents)
    
    
