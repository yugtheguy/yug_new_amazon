import collections
import math
import re
from typing import Any, Mapping, Sequence

import numpy as np

# A simple text normalization function
def normalize_text(text: str) -> str:
    if not text:
        return ""
    return str(text).lower()

def get_tokens(text: str) -> set[str]:
    return set(re.findall(r"(?u)\b\w+\b", normalize_text(text)))

class CorpusStats:
    def __init__(self):
        self.name_df = collections.Counter()
        self.address_df = collections.Counter()
        self.name_freq = collections.Counter()
        self.address_freq = collections.Counter()
        self.doc_count = 0

    def add(self, name: str, address: str):
        self.doc_count += 1
        n = normalize_text(name)
        a = normalize_text(address)
        if n:
            self.name_freq[n] += 1
            for t in set(re.findall(r"(?u)\b\w+\b", n)):
                self.name_df[t] += 1
        if a:
            self.address_freq[a] += 1
            for t in set(re.findall(r"(?u)\b\w+\b", a)):
                self.address_df[t] += 1

    def idf(self, df_counter, token: str) -> float:
        df = df_counter.get(token, 0)
        # Smoothing: +1 to avoid log(0) and division by zero
        return math.log((self.doc_count + 1) / (df + 1)) + 1.0


def parse_numeric_address(address: str):
    # Extremely conservative parser
    # returns dict with premise_base, premise_suffix, unit, floor, postal
    a = str(address).lower()
    res = {}
    
    # premise
    m = re.search(r'\b(\d{1,5})\s*(bis|ter|[a-z])?\b', a)
    if m:
        res['premise_base'] = m.group(1)
        res['premise_suffix'] = m.group(2) if m.group(2) else ""
        
    # unit
    m2 = re.search(r'\b(flat|apt|apartment|unit|suite|ste|shop|office|room)\s*#?\s*([a-z0-9\-]+)\b', a)
    if m2:
        res['unit'] = m2.group(2)
        
    # floor
    m3 = re.search(r'\b(\d{1,3})(?:st|nd|rd|th)?\s*(?:floor|fl|level)\b', a)
    if m3:
        res['floor'] = m3.group(1)
        
    # postal
    m4 = re.search(r'\b(\d{5,6})\b', a)
    if m4:
        res['postal'] = m4.group(1)
        
    return res

def core_name(name: str) -> str:
    n = normalize_text(name)
    n = re.sub(r'\b(llc|inc|corp|corporation|ltd|limited|pvt|private|llp|sarl|sas|sasu|sa|eurl|sci|snc|cie|compagnie)\b', '', n)
    return " ".join(n.split())

B007_FEATURE_NAMES = [
    # Direct Name Evidence
    "name_shared_idf_sum", "name_shared_idf_mean", "name_shared_idf_max", "name_rarest_shared_idf",
    "name_left_only_idf_sum", "name_right_only_idf_sum", "name_left_only_idf_max", "name_right_only_idf_max",
    "name_weighted_jaccard", "name_weighted_containment_left", "name_weighted_containment_right",
    
    # Direct Address Evidence
    "address_shared_idf_sum", "address_shared_idf_max", "address_left_only_idf_sum", "address_right_only_idf_sum",
    "address_weighted_jaccard", "address_weighted_containment",
    
    # Core Name / Legal Form
    "core_name_exact", "core_name_similarity", "core_name_token_overlap", "core_name_containment",
    "legal_form_left_present", "legal_form_right_present", "legal_form_exact", "legal_form_conflict", "legal_form_unknown",
    
    # Name Containment / Abbreviation
    "name_left_contains_right", "name_right_contains_left",
    "core_name_left_contains_right", "core_name_right_contains_left",
    "name_token_count_ratio", "name_char_length_ratio",
    "name_initials_exact", "name_initials_match",
    
    # Business-Name / Address Frequency
    "log1p_name_frequency", "log1p_core_name_frequency", "name_is_high_frequency",
    "log1p_address_frequency",
    
    # Premise Features
    "premise_present_left", "premise_present_right", "premise_exact", "premise_conflict", "premise_unknown",
    "premise_base_exact", "premise_suffix_exact", "premise_suffix_conflict",
    "premise_abs_distance", "premise_log_distance",
    
    # Unit / Floor
    "unit_present_left", "unit_present_right", "unit_exact", "unit_conflict", "unit_unknown",
    "floor_exact", "floor_conflict", "floor_unknown",
    
    # Postal
    "postal_present_both", "postal_exact", "postal_base_exact", "postal_prefix_match", "postal_conflict", "postal_unknown",
    
    # Cross-Role / Conflict
    "numeric_cross_role_conflict",
    "strong_name_premise_conflict", "strong_name_postal_conflict",
    "strong_address_core_name_conflict", "core_name_exact_premise_exact",
    "core_name_exact_postal_exact", "rare_name_match_premise_conflict",
    
    # Weakest-Link
    "name_address_min", "name_address_max", "name_address_product",
    "name_address_geometric_mean", "name_address_harmonic_mean",
    
    # Missingness
    "name_missing_left", "name_missing_right",
    "address_missing_left", "address_missing_right",
    "both_names_present", "both_addresses_present", "address_missing_one_side",
]

def jaccard(s1, s2):
    if not s1 and not s2: return 0.0
    return len(s1 & s2) / len(s1 | s2)

def remove_accents(s: str) -> str:
    import unicodedata
    return "".join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

def char_similarity(s1: str, s2: str) -> float:
    import difflib
    # Fold accents for better matching
    s1_fold = remove_accents(s1)
    s2_fold = remove_accents(s2)
    return max(
        difflib.SequenceMatcher(None, s1, s2).ratio(),
        difflib.SequenceMatcher(None, s1_fold, s2_fold).ratio()
    )

def generate_features(q_id, c_id, texts, stats: CorpusStats) -> list[float]:
    q_name, q_address = texts.get(q_id, ("", ""))
    c_name, c_address = texts.get(c_id, ("", ""))
    
    q_name_norm = normalize_text(q_name)
    c_name_norm = normalize_text(c_name)
    q_addr_norm = normalize_text(q_address)
    c_addr_norm = normalize_text(c_address)
    
    q_name_tok = get_tokens(q_name_norm)
    c_name_tok = get_tokens(c_name_norm)
    
    q_addr_tok = get_tokens(q_addr_norm)
    c_addr_tok = get_tokens(c_addr_norm)
    
    # Name IDF
    shared_name = q_name_tok & c_name_tok
    left_name = q_name_tok - c_name_tok
    right_name = c_name_tok - q_name_tok
    
    name_shared_idf = [stats.idf(stats.name_df, t) for t in shared_name]
    name_left_idf = [stats.idf(stats.name_df, t) for t in left_name]
    name_right_idf = [stats.idf(stats.name_df, t) for t in right_name]
    
    name_shared_idf_sum = sum(name_shared_idf)
    name_shared_idf_mean = np.mean(name_shared_idf) if name_shared_idf else 0.0
    name_shared_idf_max = max(name_shared_idf) if name_shared_idf else 0.0
    name_rarest_shared_idf = name_shared_idf_max
    
    name_left_only_idf_sum = sum(name_left_idf)
    name_right_only_idf_sum = sum(name_right_idf)
    name_left_only_idf_max = max(name_left_idf) if name_left_idf else 0.0
    name_right_only_idf_max = max(name_right_idf) if name_right_idf else 0.0
    
    sum_all = name_shared_idf_sum + name_left_only_idf_sum + name_right_only_idf_sum
    name_weighted_jaccard = name_shared_idf_sum / sum_all if sum_all > 0 else 0.0
    name_weighted_containment_left = name_shared_idf_sum / (name_shared_idf_sum + name_left_only_idf_sum) if (name_shared_idf_sum + name_left_only_idf_sum) > 0 else 0.0
    name_weighted_containment_right = name_shared_idf_sum / (name_shared_idf_sum + name_right_only_idf_sum) if (name_shared_idf_sum + name_right_only_idf_sum) > 0 else 0.0
    
    # Address IDF
    shared_addr = q_addr_tok & c_addr_tok
    left_addr = q_addr_tok - c_addr_tok
    right_addr = c_addr_tok - q_addr_tok
    
    addr_shared_idf = [stats.idf(stats.address_df, t) for t in shared_addr]
    addr_left_idf = [stats.idf(stats.address_df, t) for t in left_addr]
    addr_right_idf = [stats.idf(stats.address_df, t) for t in right_addr]
    
    address_shared_idf_sum = sum(addr_shared_idf)
    address_shared_idf_max = max(addr_shared_idf) if addr_shared_idf else 0.0
    address_left_only_idf_sum = sum(addr_left_idf)
    address_right_only_idf_sum = sum(addr_right_idf)
    
    sum_all_addr = address_shared_idf_sum + address_left_only_idf_sum + address_right_only_idf_sum
    address_weighted_jaccard = address_shared_idf_sum / sum_all_addr if sum_all_addr > 0 else 0.0
    address_weighted_containment = address_shared_idf_sum / min(address_shared_idf_sum + address_left_only_idf_sum, address_shared_idf_sum + address_right_only_idf_sum) if (address_shared_idf_sum > 0) else 0.0
    
    # Core Name / Legal form
    q_core = core_name(q_name_norm)
    c_core = core_name(c_name_norm)
    core_name_exact = float(q_core == c_core and q_core != "")
    core_name_similarity = char_similarity(q_core, c_core)
    core_q_tok = get_tokens(q_core)
    core_c_tok = get_tokens(c_core)
    core_name_token_overlap = jaccard(core_q_tok, core_c_tok)
    core_name_containment = float(bool(core_q_tok) and bool(core_c_tok) and (core_q_tok.issubset(core_c_tok) or core_c_tok.issubset(core_q_tok)))
    
    legal_q = q_name_tok - core_q_tok
    legal_c = c_name_tok - core_c_tok
    legal_form_left_present = float(bool(legal_q))
    legal_form_right_present = float(bool(legal_c))
    legal_form_exact = float(bool(legal_q) and legal_q == legal_c)
    legal_form_conflict = float(bool(legal_q) and bool(legal_c) and legal_q != legal_c)
    legal_form_unknown = float(not legal_q or not legal_c)
    
    # Name Containment
    name_left_contains_right = float(bool(c_name_norm) and c_name_norm in q_name_norm)
    name_right_contains_left = float(bool(q_name_norm) and q_name_norm in c_name_norm)
    core_name_left_contains_right = float(bool(c_core) and c_core in q_core)
    core_name_right_contains_left = float(bool(q_core) and q_core in c_core)
    
    name_token_count_ratio = min(len(q_name_tok), len(c_name_tok)) / max(len(q_name_tok), len(c_name_tok)) if max(len(q_name_tok), len(c_name_tok)) > 0 else 0.0
    name_char_length_ratio = min(len(q_name_norm), len(c_name_norm)) / max(len(q_name_norm), len(c_name_norm)) if max(len(q_name_norm), len(c_name_norm)) > 0 else 0.0
    
    q_initials = "".join([w[0] for w in q_name_tok if w])
    c_initials = "".join([w[0] for w in c_name_tok if w])
    name_initials_exact = float(bool(q_initials) and q_initials == c_initials)
    name_initials_match = float(bool(q_initials) and bool(c_initials) and (q_initials == c_name_norm or c_initials == q_name_norm))
    
    # Frequencies
    q_freq = stats.name_freq.get(q_name_norm, 1)
    c_freq = stats.name_freq.get(c_name_norm, 1)
    log1p_name_frequency = math.log1p(min(q_freq, c_freq))
    log1p_core_name_frequency = math.log1p(min(stats.name_freq.get(q_core, 1), stats.name_freq.get(c_core, 1)))
    name_is_high_frequency = float(q_freq > 10 or c_freq > 10)
    
    qa_freq = stats.address_freq.get(q_addr_norm, 1)
    ca_freq = stats.address_freq.get(c_addr_norm, 1)
    log1p_address_frequency = math.log1p(min(qa_freq, ca_freq))
    
    # Parsers
    q_parsed = parse_numeric_address(q_address)
    c_parsed = parse_numeric_address(c_address)
    
    def match_state(f):
        qv = q_parsed.get(f)
        cv = c_parsed.get(f)
        if not qv or not cv: return 0.0, 0.0, 1.0 # exact, conflict, unknown
        if qv == cv: return 1.0, 0.0, 0.0
        return 0.0, 1.0, 0.0

    premise_exact, premise_conflict, premise_unknown = match_state('premise_base')
    premise_base_exact = premise_exact
    # Suffix
    qs = q_parsed.get('premise_suffix', '')
    cs = c_parsed.get('premise_suffix', '')
    premise_suffix_exact = float(bool(qs) and qs == cs)
    premise_suffix_conflict = float((qs != cs) and (bool(qs) or bool(cs)))
    
    premise_present_left = float('premise_base' in q_parsed)
    premise_present_right = float('premise_base' in c_parsed)
    
    qb = int(q_parsed['premise_base']) if 'premise_base' in q_parsed else None
    cb = int(c_parsed['premise_base']) if 'premise_base' in c_parsed else None
    if qb is not None and cb is not None:
        premise_abs_distance = float(abs(qb - cb))
        premise_log_distance = math.log1p(premise_abs_distance)
    else:
        premise_abs_distance = 0.0
        premise_log_distance = 0.0

    unit_exact, unit_conflict, unit_unknown = match_state('unit')
    floor_exact, floor_conflict, floor_unknown = match_state('floor')
    
    unit_present_left = float('unit' in q_parsed)
    unit_present_right = float('unit' in c_parsed)
    
    postal_exact, postal_conflict, postal_unknown = match_state('postal')
    postal_present_both = float('postal' in q_parsed and 'postal' in c_parsed)
    postal_base_exact = postal_exact
    
    qp = q_parsed.get('postal', '')
    cp = c_parsed.get('postal', '')
    postal_prefix_match = float(bool(qp) and bool(cp) and (qp.startswith(cp) or cp.startswith(qp)) and qp != cp)
    if postal_prefix_match:
        postal_conflict = 0.0
    
    # cross role
    q_nums = set(re.findall(r'\d+', q_addr_norm))
    c_nums = set(re.findall(r'\d+', c_addr_norm))
    numeric_cross_role_conflict = 0.0
    if q_nums and c_nums and q_nums == c_nums:
        if premise_conflict or unit_conflict or floor_conflict:
            numeric_cross_role_conflict = 1.0

    # interactions
    name_sim = char_similarity(q_name_norm, c_name_norm)
    addr_sim = char_similarity(q_addr_norm, c_addr_norm)
    
    strong_name_premise_conflict = float(name_sim > 0.85 and premise_conflict)
    strong_name_postal_conflict = float(name_sim > 0.85 and postal_conflict)
    strong_address_core_name_conflict = float(addr_sim > 0.85 and core_name_similarity < 0.5)
    core_name_exact_premise_exact = float(core_name_exact and premise_exact)
    core_name_exact_postal_exact = float(core_name_exact and postal_exact)
    rare_name_match_premise_conflict = float(name_sim > 0.85 and name_is_high_frequency == 0 and premise_conflict)
    
    # Weakest link
    name_address_min = min(name_sim, addr_sim)
    name_address_max = max(name_sim, addr_sim)
    name_address_product = name_sim * addr_sim
    name_address_geometric_mean = math.sqrt(name_address_product)
    name_address_harmonic_mean = (2 * name_sim * addr_sim) / (name_sim + addr_sim) if (name_sim + addr_sim) > 0 else 0.0
    
    # Missingness
    name_missing_left = float(not q_name_norm)
    name_missing_right = float(not c_name_norm)
    address_missing_left = float(not q_addr_norm)
    address_missing_right = float(not c_addr_norm)
    both_names_present = float(not name_missing_left and not name_missing_right)
    both_addresses_present = float(not address_missing_left and not address_missing_right)
    address_missing_one_side = float((address_missing_left and not address_missing_right) or (address_missing_right and not address_missing_left))
    
    # Extract values in correct order
    return [
        name_shared_idf_sum, name_shared_idf_mean, name_shared_idf_max, name_rarest_shared_idf,
        name_left_only_idf_sum, name_right_only_idf_sum, name_left_only_idf_max, name_right_only_idf_max,
        name_weighted_jaccard, name_weighted_containment_left, name_weighted_containment_right,
        
        address_shared_idf_sum, address_shared_idf_max, address_left_only_idf_sum, address_right_only_idf_sum,
        address_weighted_jaccard, address_weighted_containment,
        
        core_name_exact, core_name_similarity, core_name_token_overlap, core_name_containment,
        legal_form_left_present, legal_form_right_present, legal_form_exact, legal_form_conflict, legal_form_unknown,
        
        name_left_contains_right, name_right_contains_left,
        core_name_left_contains_right, core_name_right_contains_left,
        name_token_count_ratio, name_char_length_ratio,
        name_initials_exact, name_initials_match,
        
        log1p_name_frequency, log1p_core_name_frequency, name_is_high_frequency,
        log1p_address_frequency,
        
        premise_present_left, premise_present_right, premise_exact, premise_conflict, premise_unknown,
        premise_base_exact, premise_suffix_exact, premise_suffix_conflict,
        premise_abs_distance, premise_log_distance,
        
        unit_present_left, unit_present_right, unit_exact, unit_conflict, unit_unknown,
        floor_exact, floor_conflict, floor_unknown,
        
        postal_present_both, postal_exact, postal_base_exact, postal_prefix_match, postal_conflict, postal_unknown,
        
        numeric_cross_role_conflict,
        strong_name_premise_conflict, strong_name_postal_conflict,
        strong_address_core_name_conflict, core_name_exact_premise_exact,
        core_name_exact_postal_exact, rare_name_match_premise_conflict,
        
        name_address_min, name_address_max, name_address_product,
        name_address_geometric_mean, name_address_harmonic_mean,
        
        name_missing_left, name_missing_right,
        address_missing_left, address_missing_right,
        both_names_present, both_addresses_present, address_missing_one_side,
    ]
